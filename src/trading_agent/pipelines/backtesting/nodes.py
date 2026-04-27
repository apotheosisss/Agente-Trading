# src/trading_agent/pipelines/backtesting/nodes.py

import logging

import numpy as np
import pandas as pd

from trading_agent.pipelines.shared_utils import score_row

logger = logging.getLogger(__name__)


def ejecutar_backtest(
    feature_vector: pd.DataFrame,
    vix_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Portfolio trend-following con tres mecanismos de protección:

    1. **Score mínimo de entrada** (``min_entry_score``): solo se abren
       posiciones cuando el score cuantitativo supera el umbral.  El score
       ya incorpora EMA-200 (trend filter), MACD, RSI y alineación de EMAs,
       por lo que en mercados bajistas el score cae de forma natural y se
       evita comprar sin necesitar datos externos como VIX.

    2. **Circuit breaker de portfolio** (``max_drawdown_circuit``): si el
       portfolio cae más del umbral configurado desde su máximo histórico,
       se liquidan TODAS las posiciones y se activa un período de espera
       (``circuit_break_cooldown`` días) antes de permitir nuevas entradas.
       Proporciona un límite de MaxDD hard sin depender del VIX.

    3. **VIX crisis** (``vix_crisis_threshold``): reserva para pánico
       extremo (VIX > 40).  Liquida todo inmediatamente.  Caso extremo
       adicional al circuit breaker.

    Nota: stop-loss fijo en entrada (no trailing) y peso igualitario
    entre posiciones.  Ambas simplificaciones son deliberadas: el trailing
    stop genera whipsaw en mercados tendenciales y el sizing inverso a
    volatilidad reduce la exposición a activos ganadores de alta volatilidad.

    Retorna DataFrame con columnas: equity, cash, trade_type, tickers_held,
    n_positions, buys_today, exits_today, vix, in_cooldown.
    """
    bt = parameters["backtesting"]
    initial_capital = float(bt["initial_capital"])
    commission = float(bt["commission"])
    max_positions = int(parameters["risk"]["max_positions"])
    stop_loss_atr_mult = float(parameters["risk"]["stop_loss_atr_mult"])
    rebalance_dow      = int(bt.get("rebalance_day", 0))
    rebalance_interval = int(bt.get("rebalance_interval", 0))  # 0 = usar rebalance_dow

    risk = parameters["risk"]
    min_entry_score = float(risk.get("min_entry_score", 1.5))
    circuit_pct = float(risk.get("max_drawdown_circuit", 0.12))
    cooldown_total = int(risk.get("circuit_break_cooldown", 21))
    vix_crisis = float(risk.get("vix_crisis_threshold", 40))

    # ── Preparar serie VIX ────────────────────────────────────────────────────
    if not vix_data.empty and "vix" in vix_data.columns:
        vix_series: pd.Series = vix_data["vix"].sort_index()
    else:
        vix_series = pd.Series(dtype=float)

    def _vix_for(date: pd.Timestamp) -> float:
        if vix_series.empty:
            return 20.0
        try:
            v = vix_series.asof(date)
            return 20.0 if pd.isna(v) else float(v)
        except Exception:
            return 20.0

    cash = initial_capital
    open_positions: dict = {}   # {ticker: {shares, entry_price, stop_loss}}
    peak_equity = initial_capital
    cooldown_remaining = 0      # días de espera tras circuit breaker
    daily_records = []
    days_since_rebalance = 0   # contador para rebalance_interval

    dates = sorted(feature_vector.index.unique())

    for date in dates:
        today = feature_vector[feature_vector.index == date]

        price_map: dict[str, float] = {}
        atr_map: dict[str, float] = {}
        for _, row in today.iterrows():
            t = str(row["ticker"])
            price_map[t] = float(row["close"])
            atr_map[t] = float(row["atr"])

        trade_events: list[str] = []
        vix_today = _vix_for(date)

        # ── Calcular equity actual (antes de operar) ──────────────────────────
        pos_value_pre = sum(
            pos["shares"] * price_map.get(t, pos["entry_price"])
            for t, pos in open_positions.items()
        )
        equity_pre = cash + pos_value_pre
        peak_equity = max(peak_equity, equity_pre)

        # ── 1. CIRCUIT BREAKER: caída desde pico ─────────────────────────────
        drawdown_from_peak = (equity_pre - peak_equity) / peak_equity  # <= 0
        if drawdown_from_peak <= -circuit_pct and open_positions:
            for ticker, pos in list(open_positions.items()):
                price = price_map.get(ticker, pos["entry_price"])
                cash += pos["shares"] * price * (1.0 - commission)
                trade_events.append("CIRCUIT_BREAK")
            open_positions.clear()
            # CRÍTICO: reset del pico al nivel de liquidación.
            # Sin reset, cada re-entrada seguiría midiendo DD contra el pico
            # original (2021), causando triggers repetidos en toda la recuperación.
            peak_equity = equity_pre
            cooldown_remaining = cooldown_total
            logger.warning(
                "Circuit breaker activado: DD=%.1f%% | nuevo pico=$%.0f | "
                "enfriamiento=%d dias",
                drawdown_from_peak * 100, equity_pre, cooldown_total,
            )

        # ── 2. VIX CRISIS EXTREMO: liquidar todo ─────────────────────────────
        elif vix_today > vix_crisis and open_positions:
            for ticker, pos in list(open_positions.items()):
                price = price_map.get(ticker, pos["entry_price"])
                cash += pos["shares"] * price * (1.0 - commission)
                trade_events.append("VIX_LIQUIDATE")
            open_positions.clear()
            logger.warning(
                "VIX crisis extremo (%.1f > %.1f): liquidando todo",
                vix_today, vix_crisis,
            )

        else:
            # ── 3. STOP-LOSS DIARIO (fijo en entrada) ────────────────────────
            stopped: list[str] = []
            for ticker, pos in open_positions.items():
                price = price_map.get(ticker)
                if price is None:
                    continue
                if price <= pos["stop_loss"]:
                    cash += pos["shares"] * price * (1.0 - commission)
                    stopped.append(ticker)
                    trade_events.append("STOP_LOSS")
                    logger.debug(
                        "Stop-loss: %s @ $%.2f (stop=%.2f entry=%.2f)",
                        ticker, price, pos["stop_loss"], pos["entry_price"],
                    )
            for t in stopped:
                del open_positions[t]

        # ── Gestionar cooldown ────────────────────────────────────────────────
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        in_cooldown = cooldown_remaining > 0

        # ── 4. REBALANCEO (semanal por dia-semana O cada N dias) ──────────────
        days_since_rebalance += 1
        if rebalance_interval > 0:
            do_rebalance = (days_since_rebalance >= rebalance_interval)
            if do_rebalance:
                days_since_rebalance = 0
        else:
            do_rebalance = (date.isoweekday() == rebalance_dow + 1)
        if do_rebalance:
            # Score con filtro de score mínimo para nuevas entradas
            scores: dict[str, float] = {}
            for _, row in today.iterrows():
                s = score_row(row)
                if s > -999.0:          # pasa filtro EMA-200
                    scores[str(row["ticker"])] = s

            # Solo los top-N con score MÍNIMO (filtro de condición de mercado)
            qualified = {t: s for t, s in scores.items() if s >= min_entry_score}
            target = set(
                sorted(qualified, key=lambda k: qualified[k], reverse=True)[:max_positions]
            )

            # Vender posiciones fuera del target
            to_sell = [t for t in list(open_positions.keys()) if t not in target]
            for ticker in to_sell:
                pos = open_positions.pop(ticker)
                price = price_map.get(ticker, pos["entry_price"])
                cash += pos["shares"] * price * (1.0 - commission)
                trade_events.append("SELL")

            # Comprar solo si no estamos en cooldown NI en crisis VIX
            if not in_cooldown and vix_today <= vix_crisis and target:
                pos_value = sum(
                    open_positions[t]["shares"]
                    * price_map.get(t, open_positions[t]["entry_price"])
                    for t in open_positions
                )
                total_value = cash + pos_value
                alloc = total_value / max_positions  # peso igualitario

                new_buys = [t for t in target if t not in open_positions]
                for ticker in new_buys:
                    price = price_map.get(ticker)
                    atr = atr_map.get(ticker, 0.0)
                    if not price or price <= 0.0:
                        continue
                    budget = min(alloc, cash)
                    if budget <= 0.0:
                        continue
                    shares = budget * (1.0 - commission) / price
                    stop_price = price - atr * stop_loss_atr_mult
                    cash -= shares * price * (1.0 + commission)
                    open_positions[ticker] = {
                        "shares": shares,
                        "entry_price": price,
                        "stop_loss": stop_price,
                    }
                    trade_events.append("BUY")
                    logger.debug(
                        "BUY %s shares=%.4f @ $%.2f score=%.2f (cooldown=%s)",
                        ticker, shares, price,
                        qualified.get(ticker, 0), in_cooldown,
                    )
            elif in_cooldown and target:
                logger.debug(
                    "Cooldown activo (%d dias restantes): sin nuevas compras",
                    cooldown_remaining,
                )
            elif vix_today > vix_crisis and target:
                new_buys = [t for t in target if t not in open_positions]
                if new_buys:
                    logger.debug(
                        "VIX crisis (%.1f > %.1f): sin nuevas compras el lunes",
                        vix_today, vix_crisis,
                    )

        # ── 5. SNAPSHOT DIARIO ────────────────────────────────────────────────
        pos_value = sum(
            pos["shares"] * price_map.get(t, pos["entry_price"])
            for t, pos in open_positions.items()
        )
        equity = cash + pos_value

        if "CIRCUIT_BREAK" in trade_events:
            trade_type = "CIRCUIT_BREAK"
        elif "VIX_LIQUIDATE" in trade_events:
            trade_type = "VIX_CRISIS"
        elif "STOP_LOSS" in trade_events:
            trade_type = "STOP_LOSS"
        elif "BUY" in trade_events:
            trade_type = "BUY"
        elif "SELL" in trade_events:
            trade_type = "SELL"
        else:
            trade_type = "HOLD"

        daily_records.append(
            {
                "date": date,
                "equity": equity,
                "cash": cash,
                "trade_type": trade_type,
                "tickers_held": ",".join(sorted(open_positions.keys())),
                "n_positions": len(open_positions),
                "buys_today": trade_events.count("BUY"),
                "exits_today": (
                    trade_events.count("SELL")
                    + trade_events.count("STOP_LOSS")
                    + trade_events.count("CIRCUIT_BREAK")
                    + trade_events.count("VIX_LIQUIDATE")
                ),
                "vix": vix_today,
                "in_cooldown": in_cooldown,
            }
        )

    portfolio = pd.DataFrame(daily_records).set_index("date")

    total_buys = int(portfolio["buys_today"].sum())
    circuit_events = int((portfolio["trade_type"] == "CIRCUIT_BREAK").sum())
    logger.info(
        "Backtest completado: %d periodos | %d compras | %d circuit-breaks | "
        "%d posiciones abiertas al cierre",
        len(dates), total_buys, circuit_events, len(open_positions),
    )
    return portfolio


def calcular_benchmark(feature_vector: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Benchmark buy-and-hold sobre SPY (o primer ticker disponible)."""
    initial_capital = float(parameters["backtesting"]["initial_capital"])
    commission = float(parameters["backtesting"]["commission"])
    benchmark_ticker = str(parameters.get("ticker", "SPY"))

    spy = feature_vector[feature_vector["ticker"] == benchmark_ticker].sort_index()
    if spy.empty:
        first_ticker = str(feature_vector["ticker"].iloc[0])
        spy = feature_vector[feature_vector["ticker"] == first_ticker].sort_index()
        logger.warning(
            "%s no disponible — usando %s como benchmark", benchmark_ticker, first_ticker
        )

    spy = spy[~spy.index.duplicated(keep="first")]

    first_price = float(spy["close"].iloc[0])
    shares = initial_capital * (1.0 - commission) / first_price
    spy_equity = spy["close"] * shares

    total_return = float(spy_equity.iloc[-1]) / initial_capital - 1.0
    logger.info("Benchmark %s: retorno total %.1f%%", benchmark_ticker, total_return * 100)

    return pd.DataFrame({"date": spy.index, "equity": spy_equity.values})


def calcular_metricas(portfolio: pd.DataFrame, parameters: dict) -> tuple:
    """Calcula métricas de rendimiento y genera la curva de equity.

    Retorna tupla ``(metrics_df, equity_df)`` mapeada a
    ``(backtest_metrics, equity_curve)``.
    """
    initial_capital = float(parameters["backtesting"]["initial_capital"])
    equity = portfolio["equity"].dropna()

    returns = equity.pct_change().dropna()

    if returns.std() > 0:
        sharpe = float(returns.mean() * 252 / (returns.std() * np.sqrt(252)))
    else:
        sharpe = 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    n_years = len(equity) / 252
    final_equity = float(equity.iloc[-1])
    cagr = (
        (final_equity / initial_capital) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    )

    n_buys = int(portfolio["buys_today"].sum())
    n_exits = int(portfolio["exits_today"].sum())
    n_trades = min(n_buys, n_exits)
    trades_per_year = round(n_trades / n_years, 1) if n_years > 0 else 0.0

    pos_days = int((returns > 0).sum())
    win_rate = pos_days / len(returns) if len(returns) > 0 else 0.0

    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    profit_factor = (
        (gains / losses)
        if losses > 0
        else (float("inf") if gains > 0 else 0.0)
    )

    circuit_events = int(
        (portfolio["trade_type"] == "CIRCUIT_BREAK").sum()
    ) if "trade_type" in portfolio.columns else 0
    vix_crisis_days = int(
        (portfolio["trade_type"] == "VIX_CRISIS").sum()
    ) if "trade_type" in portfolio.columns else 0

    metrics_df = pd.DataFrame(
        [
            {
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "win_rate_pct": round(win_rate * 100, 2),
                "profit_factor": round(profit_factor, 4),
                "cagr_pct": round(cagr * 100, 2),
                "final_equity_usd": round(final_equity, 2),
                "total_return_pct": round((final_equity / initial_capital - 1) * 100, 2),
                "n_trades": n_trades,
                "trades_per_year": trades_per_year,
                "circuit_break_events": circuit_events,
                "vix_crisis_days": vix_crisis_days,
            }
        ]
    )

    logger.info(
        "Metricas: Sharpe=%.2f | MaxDD=%.1f%% | CAGR=%.1f%% | "
        "Trades/yr=%.1f | WinRate=%.1f%% | CircuitBreaks=%d",
        sharpe,
        max_drawdown * 100,
        cagr * 100,
        trades_per_year,
        win_rate * 100,
        circuit_events,
    )

    equity_df = equity.reset_index()
    equity_df.columns = ["date", "equity"]
    return metrics_df, equity_df


def calcular_walk_forward(
    backtest_portfolio: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Valida robustez de la estrategia comparando in-sample vs out-of-sample.

    Divide la curva de equity en la fecha ``walk_forward_split`` y calcula
    Sharpe, MaxDD y CAGR para cada ventana.

    Un sistema robusto muestra métricas similares en ambas ventanas.
    Si el out-of-sample es muy inferior, la estrategia está sobre-ajustada
    y no es apta para dinero real.

    Retorna DataFrame con una fila por periodo.
    """
    split_date_str = str(
        parameters["backtesting"].get("walk_forward_split", "2022-01-01")
    )
    split_date = pd.Timestamp(split_date_str)

    eq = backtest_portfolio["equity"].sort_index()

    def _metrics(series: pd.Series, label: str) -> dict:
        if len(series) < 5:
            return {
                "periodo": label, "sharpe": 0.0,
                "max_drawdown_pct": 0.0, "cagr_pct": 0.0, "n_dias": 0,
            }
        rets = series.pct_change().dropna()
        sharpe = (
            float(rets.mean() * 252 / (rets.std() * np.sqrt(252)))
            if rets.std() > 0 else 0.0
        )
        rolling_max = series.cummax()
        max_dd = float(((series - rolling_max) / rolling_max).min()) * 100
        n_years = len(series) / 252
        start_val = float(series.iloc[0])
        end_val = float(series.iloc[-1])
        cagr = (
            ((end_val / start_val) ** (1.0 / n_years) - 1.0) * 100 if n_years > 0 else 0.0
        )
        return {
            "periodo": label,
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 1),
            "cagr_pct": round(cagr, 1),
            "n_dias": len(series),
        }

    in_sample = eq[eq.index < split_date]
    out_sample = eq[eq.index >= split_date]

    rows = [
        _metrics(in_sample, f"In-sample (hasta {split_date_str})"),
        _metrics(out_sample, f"Out-of-sample (desde {split_date_str})"),
        _metrics(eq, "Completo"),
    ]

    result = pd.DataFrame(rows)
    logger.info(
        "Walk-forward | In-sample Sharpe=%.2f CAGR=%.1f%% | "
        "Out-of-sample Sharpe=%.2f CAGR=%.1f%%",
        rows[0]["sharpe"], rows[0]["cagr_pct"],
        rows[1]["sharpe"], rows[1]["cagr_pct"],
    )
    return result
