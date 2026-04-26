import numpy as np
import pandas as pd
import pytest

from trading_agent.pipelines.backtesting.nodes import calcular_metricas, ejecutar_backtest


def test_backtest_columnas(sample_feature_vector, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    for col in ["equity", "cash", "position_value", "trade_type", "price"]:
        assert col in result.columns


def test_backtest_equity_inicial(sample_feature_vector, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    capital = sample_parameters["backtesting"]["initial_capital"]
    assert abs(result["equity"].iloc[0] - capital) < 1.0


def test_backtest_equity_positivo(sample_feature_vector, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    assert (result["equity"] > 0).all()


def test_backtest_sin_senales(sample_parameters):
    """Cuando RSI siempre es neutral (40-60), no hay trades y equity es constante."""
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    close = pd.Series(np.linspace(20_000, 21_000, 40), index=dates)
    df = pd.DataFrame({
        "open": close, "high": close * 1.001,
        "low": close * 0.999, "close": close, "volume": 5000.0,
        "rsi": 50.0,
        "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
        "bb_upper": close * 1.02, "bb_mid": close, "bb_lower": close * 0.98,
        "ema_20": close, "ema_50": close, "atr": 100.0,
        "sentiment_score": 0.0,
    }, index=dates)
    result = ejecutar_backtest(df, sample_parameters)
    assert (result["trade_type"] == "").all()
    capital = sample_parameters["backtesting"]["initial_capital"]
    assert (result["equity"] == capital).all()


def test_metricas_tipos(sample_feature_vector, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_parameters)
    result = calcular_metricas(portfolio, sample_parameters)
    assert isinstance(result, tuple) and len(result) == 2
    metrics_df, equity_df = result
    assert hasattr(metrics_df, "columns")
    assert hasattr(equity_df, "columns")


def test_metricas_columnas(sample_feature_vector, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_parameters)
    metrics_df, equity_df = calcular_metricas(portfolio, sample_parameters)
    for col in ["sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "cagr_pct", "n_trades"]:
        assert col in metrics_df.columns


def test_metricas_sin_trades(sample_parameters):
    """Sin trades: n_trades=0, win_rate=0, profit_factor=0."""
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    portfolio = pd.DataFrame({
        "equity": capital,
        "cash": capital,
        "position_value": 0.0,
        "trade_type": "",
        "price": 20_000.0,
    }, index=dates)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    assert int(metrics_df["n_trades"].iloc[0]) == 0
    assert float(metrics_df["win_rate_pct"].iloc[0]) == 0.0
    assert float(metrics_df["profit_factor"].iloc[0]) == 0.0


def test_metricas_sharpe_constante(sample_parameters):
    """Equity constante → returns std=0 → Sharpe=0."""
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    portfolio = pd.DataFrame({
        "equity": capital,
        "cash": capital,
        "position_value": 0.0,
        "trade_type": "",
        "price": 20_000.0,
    }, index=dates)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    assert float(metrics_df["sharpe_ratio"].iloc[0]) == 0.0


def test_equity_curve_columnas(sample_feature_vector, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_parameters)
    _, equity_df = calcular_metricas(portfolio, sample_parameters)
    assert "date" in equity_df.columns
    assert "equity" in equity_df.columns
