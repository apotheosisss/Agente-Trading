# src/trading_agent/pipelines/backtesting/nodes.py

import logging

import numpy as np
import pandas as pd

from trading_agent.pipelines.shared_utils import score_row

logger = logging.getLogger(__name__)


def ejecutar_backtest(feature_vector: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Portfolio trend-following: rotación semanal + stop-loss ATR diario.

    Cada lunes se re-rankean todos los activos por score (solo los que cierran
    por encima de EMA 200). Se mantienen los top ``max_positions`` en peso igual.
    Cada día se verifican stop-losses individuales por posición.

    Retorna DataFrame con una fila por fecha: equity, cash, trade_type,
    tickers_held, n_positions, buys_today, exits_today.
    """
    bt = parameters["backtesting"]
    initial_capital = float(bt["initial_capital"])
    commission = float(bt["commission"])
    max_positions = int(parameters["risk"]["max_positions"])
    stop_loss_atr_mult = float(parameters["risk"]["stop_loss_atr_mult"])
    rebalance_dow = int(bt.get("rebalance_day", 0))  # 0=lun, isoweekday Mon=1

    cash = initial_capital
    # {ticker: {"shares": float, "entry_price": float, "stop_loss": float}}
    open_positions: dict = {}
    daily_records = []

    dates = sorted(feature_vector.index.unique())

    for date in dates:
        today = feature_vector[feature_vector.index == date]

        # Mapas precio/ATR para este día
        price_map: dict[str, float] = {}
        atr_map: dict[str, float] = {}
        for _, row in today.iterrows():
            t = str(row["ticker"])
            price_map[t] = float(row["close"])
            atr_map[t] = float(row["atr"])

        trade_events: list[str] = []

        # ── 1. STOP-LOSS DIARIO ──────────────────────────────────────────────
        stopped: list[str] = []
        for ticker, pos in open_positions.items():
            price = price_map.get(ticker)
            if price is None:
                continue
            if price <= pos["stop_loss"]:
                cash += pos["shares"] * price * (1.0 - commission)
                stopped.append(ticker)
                trade_events.append("STOP_LOSS")
                logger.debug("Stop-loss: %s @ $%.2f", ticker, price)

        for t in stopped:
            del open_positions[t]

        # ── 2. REBALANCEO SEMANAL ────────────────────────────────────────────
        if date.isoweekday() == rebalance_dow + 1:
            # Score solo activos con datos hoy
            scores: dict[str, float] = {}
            for _, row in today.iterrows():
                s = score_row(row)
                if s > -999.0:
                    scores[str(row["ticker"])] = s

            target = set(
                sorted(scores, key=lambda k: scores[k], reverse=True)[:max_positions]
            )

            # Vender posiciones que salen del target
            to_sell = [t for t in list(open_positions.keys()) if t not in target]
            for ticker in to_sell:
                pos = open_positions.pop(ticker)
                price = price_map.get(ticker, pos["entry_price"])
                cash += pos["shares"] * price * (1.0 - commission)
                trade_events.append("SELL")

            # Valor total del portfolio para asignación equitativa
            pos_value = sum(
                open_positions[t]["shares"]
                * price_map.get(t, open_positions[t]["entry_price"])
                for t in open_positions
            )
            total_value = cash + pos_value
            alloc = total_value / max_positions  # presupuesto por posición

            # Comprar posiciones nuevas del target
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

        # ── 3. SNAPSHOT DIARIO ───────────────────────────────────────────────
        pos_value = sum(
            pos["shares"] * price_map.get(t, pos["entry_price"])
            for t, pos in open_positions.items()
        )
        equity = cash + pos_value

        # Tipo de evento dominante del día
        if "STOP_LOSS" in trade_events:
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
                "exits_today": trade_events.count("SELL")
                + trade_events.count("STOP_LOSS"),
            }
        )

    portfolio = pd.DataFrame(daily_records).set_index("date")

    total_buys = int(portfolio["buys_today"].sum())
    logger.info(
        "Backtest completado: %d periodos | %d compras | %d posiciones abiertas al cierre",
        len(dates),
        total_buys,
        len(open_positions),
    )
    return portfolio


def calcular_benchmark(feature_vector: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Benchmark buy-and-hold sobre SPY (o primer ticker disponible).

    Retorna DataFrame con columnas ``date`` y ``equity`` para PlotlyDataset.
    """
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

    # Una fila por fecha
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

    # Trades = compras (cada compra representa una entrada)
    n_buys = int(portfolio["buys_today"].sum())
    n_exits = int(portfolio["exits_today"].sum())
    n_trades = min(n_buys, n_exits)
    trades_per_year = round(n_trades / n_years, 1) if n_years > 0 else 0.0

    # Win rate y profit factor sobre retornos diarios
    pos_days = int((returns > 0).sum())
    win_rate = pos_days / len(returns) if len(returns) > 0 else 0.0

    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    profit_factor = (
        (gains / losses)
        if losses > 0
        else (float("inf") if gains > 0 else 0.0)
    )

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
            }
        ]
    )

    logger.info(
        "Metricas: Sharpe=%.2f | MaxDD=%.1f%% | CAGR=%.1f%% | "
        "Trades/yr=%.1f | WinRate=%.1f%%",
        sharpe,
        max_drawdown * 100,
        cagr * 100,
        trades_per_year,
        win_rate * 100,
    )

    equity_df = equity.reset_index()
    equity_df.columns = ["date", "equity"]
    return metrics_df, equity_df
