# src/trading_agent/pipelines/backtesting/nodes.py

import logging

import numpy as np
import pandas as pd

from trading_agent.pipelines.shared_utils import compute_scores, score_row

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
    slippage = float(bt.get("slippage", 0.0))
    execute_next_open = bool(bt.get("execute_next_open", False))
    # Coste total por lado de operación: comisión + deslizamiento.
    cost = commission + slippage
    max_positions = int(parameters["risk"]["max_positions"])
    stop_loss_atr_mult = float(parameters["risk"]["stop_loss_atr_mult"])
    rebalance_dow      = int(bt.get("rebalance_day", 0))
    rebalance_interval = int(bt.get("rebalance_interval", 0))  # 0 = usar rebalance_dow

    # ── Integración Polymarket (opcional, para ablation A/B/C) ────────────────
    # use_polymarket=false → estrategia base (variante B).
    # use_polymarket=true  → suma poly_score histórico por (fecha,ticker) (var. A).
    # polymarket_shuffle=true → baraja los poly_score (variante C: ¿es ruido?).
    use_poly = bool(bt.get("use_polymarket", False))
    poly_shuffle = bool(bt.get("polymarket_shuffle", False))
    poly_lookup: dict = {}
    poly_overlap_days = 0
    if use_poly:
        from pathlib import Path
        _p = Path("data/01_raw/polymarket_history.parquet")
        if _p.exists():
            _ph = pd.read_parquet(_p)
            if poly_shuffle and not _ph.empty:
                _ph = _ph.copy()
                _ph["poly_score"] = (
                    _ph["poly_score"].sample(frac=1.0, random_state=42).to_numpy()
                )
            _bt_dates = {pd.Timestamp(d).normalize() for d in feature_vector.index.unique()}
            for _, r in _ph.iterrows():
                d = pd.Timestamp(r["date"]).normalize()
                poly_lookup[(d, str(r["ticker"]))] = float(r["poly_score"])
                if d in _bt_dates:
                    poly_overlap_days += 1
        logger.info(
            "Backtest Polymarket: use=%s shuffle=%s | %d filas históricas | "
            "%d con solape en ventana de backtest",
            use_poly, poly_shuffle, len(poly_lookup), poly_overlap_days,
        )

    # ── Estrategia de señal (Fase B) ──────────────────────────────────────────
    signal_strategy = str(parameters.get("signal_strategy", "legacy"))

    risk = parameters["risk"]
    min_entry_score = float(risk.get("min_entry_score", 1.5))
    # Las estrategias nuevas devuelven z-scores (~N(0,1)); el umbral legacy (2.0)
    # no aplica. Usar signal_min_score (en desviaciones estándar; 0 = sobre la media).
    if signal_strategy != "legacy":
        min_entry_score = float(parameters.get("signal_min_score", 0.0))
    circuit_pct = float(risk.get("max_drawdown_circuit", 0.12))
    cooldown_total = int(risk.get("circuit_break_cooldown", 21))
    vix_crisis = float(risk.get("vix_crisis_threshold", 40))
    # ── Fase A: re-armado por RÉGIMEN de mercado (no por equity) ──────────────
    # El circuit breaker liquida en caídas, pero la re-entrada NO se decide por
    # la recuperación del portfolio (que en cash nunca llega), sino por el régimen
    # de mercado: re-entra cuando el ticker de referencia vuelve por encima de su
    # EMA-200. Esto evita (a) componer caídas re-entrando 25% más abajo cada vez
    # y (b) perderse la recuperación quedándose en cash para siempre.
    circuit_rearm_regime = bool(risk.get("circuit_rearm_regime", True))
    regime_ticker = str(risk.get("regime_ticker", "SPY"))
    # ── Trailing stop (Fase A) ────────────────────────────────────────────────
    trailing_stop = bool(risk.get("trailing_stop", False))
    # ── Sizing por volatilidad (Fase A) ───────────────────────────────────────
    vol_sizing = bool(risk.get("vol_sizing", False))

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
    circuit_active = False      # True = liquidado, esperando re-armado por régimen
    daily_records = []
    days_since_rebalance = 0

    # ── Realismo de ejecución: open del día SIGUIENTE por ticker ──────────────
    # La señal se calcula con el cierre del día D; la orden se ejecuta a la
    # apertura del día D+1. Evita el look-ahead de operar al mismo cierre que
    # generó la señal. shift(-1) dentro de cada ticker (frame ordenado por fecha).
    feature_vector = feature_vector.sort_index().copy()
    if execute_next_open and "open" in feature_vector.columns:
        feature_vector["_next_open"] = (
            feature_vector.groupby("ticker")["open"].shift(-1)
        )
    else:
        feature_vector["_next_open"] = np.nan

    dates = sorted(feature_vector.index.unique())

    for date in dates:
        today = feature_vector[feature_vector.index == date]

        price_map: dict[str, float] = {}
        atr_map: dict[str, float] = {}
        fill_map: dict[str, float] = {}   # precio de ejecución (next-open o close)
        regime_close = None
        regime_ema200 = None
        for _, row in today.iterrows():
            t = str(row["ticker"])
            close = float(row["close"])
            price_map[t] = close
            atr_map[t] = float(row["atr"])
            nxt = row.get("_next_open", np.nan)
            fill_map[t] = float(nxt) if (execute_next_open and pd.notna(nxt) and nxt > 0) else close
            if t == regime_ticker:
                regime_close = close
                regime_ema200 = float(row.get("ema_200", row.get("ema_50", close)))

        # Régimen risk-on: ticker de referencia por encima de su EMA-200.
        # Si el ticker de régimen no cotiza hoy, se asume risk-on (no bloquea).
        regime_risk_on = (
            True if regime_close is None or regime_ema200 is None
            else regime_close > regime_ema200
        )

        trade_events: list[str] = []
        vix_today = _vix_for(date)

        # ── Calcular equity actual (antes de operar) ──────────────────────────
        pos_value_pre = sum(
            pos["shares"] * price_map.get(t, pos["entry_price"])
            for t, pos in open_positions.items()
        )
        equity_pre = cash + pos_value_pre
        peak_equity = max(peak_equity, equity_pre)

        # ── 1. CIRCUIT BREAKER: caída desde pico (Fase A: sin compounding) ────
        # El pico se mantiene en el MÁXIMO HISTÓRICO REAL (no se resetea hacia
        # abajo). Tras liquidar, se entra en `circuit_active` y solo se re-arma
        # cuando el régimen de mercado vuelve a risk-on (EMA-200), evitando
        # re-entrar 25% más abajo en cada escalón de un bear prolongado.
        drawdown_from_peak = (equity_pre - peak_equity) / peak_equity  # <= 0
        if drawdown_from_peak <= -circuit_pct and open_positions:
            for ticker, pos in list(open_positions.items()):
                price = price_map.get(ticker, pos["entry_price"])
                cash += pos["shares"] * price * (1.0 - cost)
                trade_events.append("CIRCUIT_BREAK")
            open_positions.clear()
            circuit_active = True
            cooldown_remaining = cooldown_total
            logger.warning(
                "Circuit breaker activado: DD=%.1f%% desde pico $%.0f | "
                "esperando re-armado por régimen (%s)",
                drawdown_from_peak * 100, peak_equity, regime_ticker,
            )

        # ── 2. VIX CRISIS EXTREMO: liquidar todo ─────────────────────────────
        elif vix_today > vix_crisis and open_positions:
            for ticker, pos in list(open_positions.items()):
                price = price_map.get(ticker, pos["entry_price"])
                cash += pos["shares"] * price * (1.0 - cost)
                trade_events.append("VIX_LIQUIDATE")
            open_positions.clear()
            logger.warning(
                "VIX crisis extremo (%.1f > %.1f): liquidando todo",
                vix_today, vix_crisis,
            )

        else:
            # ── 3a. TRAILING STOP: el stop sube con el precio, nunca baja ─────
            if trailing_stop:
                for ticker, pos in open_positions.items():
                    price = price_map.get(ticker)
                    if price is None:
                        continue
                    atr = atr_map.get(ticker, 0.0)
                    new_stop = price - atr * stop_loss_atr_mult
                    if new_stop > pos["stop_loss"]:
                        pos["stop_loss"] = new_stop

            # ── 3b. STOP-LOSS DIARIO ─────────────────────────────────────────
            stopped: list[str] = []
            for ticker, pos in open_positions.items():
                price = price_map.get(ticker)
                if price is None:
                    continue
                if price <= pos["stop_loss"]:
                    cash += pos["shares"] * price * (1.0 - cost)
                    stopped.append(ticker)
                    trade_events.append("STOP_LOSS")
                    logger.debug(
                        "Stop-loss: %s @ $%.2f (stop=%.2f entry=%.2f)",
                        ticker, price, pos["stop_loss"], pos["entry_price"],
                    )
            for t in stopped:
                del open_positions[t]

        # ── Gestionar cooldown y re-armado por régimen ────────────────────────
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        in_cooldown = cooldown_remaining > 0

        # Re-armado: salir de circuit_active cuando termina el cooldown Y el
        # mercado vuelve a risk-on. Reseteamos el pico al equity actual: como
        # evitamos la caída quedándonos en cash, empezamos una "campaña nueva"
        # desde un régimen sano (sin esto, el viejo pico re-dispararía al instante).
        if circuit_active and not in_cooldown and (
            not circuit_rearm_regime or regime_risk_on
        ):
            circuit_active = False
            peak_equity = equity_pre
            logger.info(
                "Circuit re-armado (régimen %s risk-on). Pico reiniciado a $%.0f",
                regime_ticker, equity_pre,
            )

        # ── 4. REBALANCEO ────────────────────────────────────────────────────────
        days_since_rebalance += 1
        if rebalance_interval > 0:
            do_rebalance = (days_since_rebalance >= rebalance_interval)
            if do_rebalance:
                days_since_rebalance = 0
        else:
            do_rebalance = (date.isoweekday() == rebalance_dow + 1)
        if do_rebalance:
            # Score cross-sectional según la estrategia de señal seleccionada.
            date_norm = pd.Timestamp(date).normalize()
            scores: dict[str, float] = compute_scores(today, signal_strategy)
            if use_poly:
                for ticker_s in list(scores.keys()):
                    scores[ticker_s] += poly_lookup.get((date_norm, ticker_s), 0.0)

            # Solo los top-N con score MÍNIMO (filtro de condición de mercado)
            qualified = {t: s for t, s in scores.items() if s >= min_entry_score}
            target = set(
                sorted(qualified, key=lambda k: qualified[k], reverse=True)[:max_positions]
            )

            # Vender posiciones fuera del target
            to_sell = [t for t in list(open_positions.keys()) if t not in target]
            for ticker in to_sell:
                pos = open_positions.pop(ticker)
                price = fill_map.get(ticker, price_map.get(ticker, pos["entry_price"]))
                cash += pos["shares"] * price * (1.0 - cost)
                trade_events.append("SELL")

            # Comprar solo si no hay circuit activo, ni cooldown, ni crisis VIX
            if not circuit_active and not in_cooldown and vix_today <= vix_crisis and target:
                pos_value = sum(
                    open_positions[t]["shares"]
                    * price_map.get(t, open_positions[t]["entry_price"])
                    for t in open_positions
                )
                total_value = cash + pos_value
                alloc = total_value / max_positions  # peso igualitario base
                target_atr_pct = float(risk.get("target_atr_pct", 0.02))

                new_buys = [t for t in target if t not in open_positions]
                for ticker in new_buys:
                    price = fill_map.get(ticker, price_map.get(ticker))
                    atr = atr_map.get(ticker, 0.0)
                    if not price or price <= 0.0:
                        continue
                    alloc_i = alloc
                    if vol_sizing and atr > 0:
                        # Escalar inverso a volatilidad: menos tamaño a más ATR%.
                        # Clip [0.3, 1.0] → nunca apalanca por encima del peso igual.
                        atr_pct = atr / price
                        scale = min(1.0, max(0.3, target_atr_pct / atr_pct))
                        alloc_i = alloc * scale
                    budget = min(alloc_i, cash)
                    if budget <= 0.0:
                        continue
                    shares = budget * (1.0 - cost) / price
                    stop_price = price - atr * stop_loss_atr_mult
                    cash -= shares * price * (1.0 + cost)
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


def _wf_metrics(series: pd.Series, label: str) -> dict:
    """Métricas de robustez para un tramo de la curva de equity."""
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


def calcular_walk_forward(
    backtest_portfolio: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Valida robustez con walk-forward MULTI-VENTANA.

    Cambio v2 (de-overfitting): en vez de un único corte in/out-of-sample
    (que es fácil de sobre-ajustar), parte la curva de equity en ``n_folds``
    segmentos consecutivos de igual longitud y reporta Sharpe/MaxDD/CAGR de
    cada uno, más la MEDIANA de Sharpe entre segmentos.

    La mediana de Sharpe entre folds es la métrica de selección recomendada
    para el sweep: premia configuraciones que rinden de forma consistente en
    el tiempo, no las que tuvieron suerte en un único tramo.

    Se conservan además las filas ``In-sample`` / ``Out-of-sample`` / ``Completo``
    para compatibilidad con scripts existentes (run_param_sweep.py lee la fila
    "Out-of-sample").

    Retorna DataFrame con una fila por periodo/segmento + fila resumen.
    """
    bt = parameters["backtesting"]
    split_date_str = str(bt.get("walk_forward_split", "2022-01-01"))
    split_date = pd.Timestamp(split_date_str)
    n_folds = int(bt.get("walk_forward_folds", 5))

    eq = backtest_portfolio["equity"].sort_index()

    rows = []

    # ── Compatibilidad: in-sample / out-of-sample / completo ──────────────────
    in_sample = eq[eq.index < split_date]
    out_sample = eq[eq.index >= split_date]
    rows.append(_wf_metrics(in_sample, f"In-sample (hasta {split_date_str})"))
    rows.append(_wf_metrics(out_sample, f"Out-of-sample (desde {split_date_str})"))
    rows.append(_wf_metrics(eq, "Completo"))

    # ── Multi-ventana: n_folds segmentos consecutivos ────────────────────────
    fold_sharpes: list[float] = []
    fold_cagrs: list[float] = []
    if len(eq) >= n_folds * 5 and n_folds > 1:
        bounds = np.linspace(0, len(eq), n_folds + 1, dtype=int)
        for i in range(n_folds):
            seg = eq.iloc[bounds[i]:bounds[i + 1]]
            lo = seg.index.min().strftime("%Y-%m") if len(seg) else "?"
            hi = seg.index.max().strftime("%Y-%m") if len(seg) else "?"
            m = _wf_metrics(seg, f"Fold {i + 1}/{n_folds} ({lo}->{hi})")
            rows.append(m)
            fold_sharpes.append(m["sharpe"])
            fold_cagrs.append(m["cagr_pct"])

    # ── Resumen de robustez ──────────────────────────────────────────────────
    if fold_sharpes:
        median_sharpe = float(np.median(fold_sharpes))
        min_sharpe = float(np.min(fold_sharpes))
        median_cagr = float(np.median(fold_cagrs))
        rows.append({
            "periodo": "RESUMEN (mediana folds)",
            "sharpe": round(median_sharpe, 2),
            "max_drawdown_pct": round(min_sharpe, 2),  # reutilizado: peor Sharpe de fold
            "cagr_pct": round(median_cagr, 1),
            "n_dias": len(fold_sharpes),
        })
    else:
        median_sharpe = 0.0

    result = pd.DataFrame(rows)
    logger.info(
        "Walk-forward multi-ventana | In-sample Sharpe=%.2f | "
        "Out-of-sample Sharpe=%.2f | MEDIANA folds Sharpe=%.2f",
        rows[0]["sharpe"], rows[1]["sharpe"], median_sharpe,
    )
    return result
