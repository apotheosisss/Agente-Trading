# src/trading_agent/pipelines/backtesting/nodes.py

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ejecutar_backtest(feature_vector: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Simula estrategia RSI+MACD combinada sobre datos históricos.

    Señal de entrada: RSI < 35 Y MACD alcista.
    Señal de salida:  RSI > 65 Y MACD bajista.
    Retorna DataFrame con equidad, cash, valor de posición y tipo de operación por fecha.
    """
    p = parameters["backtesting"]
    initial_capital = float(p["initial_capital"])
    commission = float(p["commission"])

    cash = initial_capital
    position = 0.0
    records = []

    for date, row in feature_vector.iterrows():
        price = float(row["close"])
        rsi = float(row["rsi"])
        macd = float(row["macd"])
        macd_sig = float(row["macd_signal"])

        buy_signal = rsi < 35 and macd > macd_sig
        sell_signal = rsi > 65 and macd < macd_sig

        trade_type = ""
        if buy_signal and position == 0.0:
            shares = (cash * (1.0 - commission)) / price
            position = shares
            cash = 0.0
            trade_type = "BUY"
        elif sell_signal and position > 0.0:
            cash = position * price * (1.0 - commission)
            position = 0.0
            trade_type = "SELL"

        records.append(
            {
                "date": date,
                "equity": cash + position * price,
                "cash": cash,
                "position_value": position * price,
                "trade_type": trade_type,
                "price": price,
            }
        )

    portfolio = pd.DataFrame(records).set_index("date")

    n_buys = (portfolio["trade_type"] == "BUY").sum()
    logger.info(
        f"Backtest completado: {len(feature_vector)} periodos, {n_buys} operaciones de compra"
    )
    return portfolio


def calcular_metricas(portfolio: pd.DataFrame, parameters: dict) -> tuple:
    """Calcula metricas de rendimiento y genera la curva de equity como DataFrame.

    Retorna tupla (metrics_df, equity_df) mapeada a (backtest_metrics, equity_curve).
    """
    initial_capital = float(parameters["backtesting"]["initial_capital"])
    equity = portfolio["equity"]

    # Retornos diarios
    returns = equity.pct_change().dropna()

    # Sharpe Ratio (252 dias, tasa libre de riesgo = 0)
    if returns.std() > 0:
        sharpe = float(returns.mean() * 252 / (returns.std() * np.sqrt(252)))
    else:
        sharpe = 0.0

    # Max Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # CAGR
    n_years = len(equity) / 252
    final_equity = float(equity.iloc[-1])
    cagr = (final_equity / initial_capital) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # Win Rate y Profit Factor (emparejando BUY/SELL consecutivos)
    buy_prices = portfolio.loc[portfolio["trade_type"] == "BUY", "price"].values
    sell_prices = portfolio.loc[portfolio["trade_type"] == "SELL", "price"].values
    n_trades = min(len(buy_prices), len(sell_prices))

    if n_trades > 0:
        pnls = [
            (sell_prices[i] - buy_prices[i]) / buy_prices[i] for i in range(n_trades)
        ]
        win_rate = sum(1 for p in pnls if p > 0) / n_trades
        gains = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0

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
            }
        ]
    )

    logger.info(
        f"Metricas: Sharpe={sharpe:.2f} | MaxDD={max_drawdown:.1%} | "
        f"CAGR={cagr:.1%} | WinRate={win_rate:.1%} | Trades={n_trades}"
    )

    # Curva de equity como DataFrame para PlotlyDataset del catalog
    equity_df = portfolio[["equity"]].reset_index()
    equity_df.columns = ["date", "equity"]

    return metrics_df, equity_df
