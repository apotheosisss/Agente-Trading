import numpy as np
import pandas as pd
import pytest

from trading_agent.pipelines.backtesting.nodes import (
    calcular_benchmark,
    calcular_metricas,
    ejecutar_backtest,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_multi_fv(n_tickers: int = 2, n_days: int = 40, close_val: float = 20_000.0):
    """Feature vector sintético multi-ticker para pruebas controladas."""
    dates = pd.date_range("2023-01-02", periods=n_days, freq="D")
    tickers = [f"T{i}" for i in range(n_tickers)]
    frames = []
    for ticker in tickers:
        close = np.full(n_days, close_val)
        df = pd.DataFrame(
            {
                "ticker": ticker,
                "open": close, "high": close * 1.001,
                "low": close * 0.999, "close": close, "volume": 5000.0,
                "rsi": 50.0,
                "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
                "bb_upper": close * 1.02, "bb_mid": close, "bb_lower": close * 0.98,
                "ema_20": close, "ema_50": close,
                "ema_200": close * 0.5,  # muy por debajo → trend filter pasa
                "atr": 100.0,
                "sentiment_score": 0.0,
            },
            index=dates,
        )
        frames.append(df)
    result = pd.concat(frames).sort_index()
    result.index.name = "date"
    return result


def _make_portfolio_df(n: int = 40, capital: float = 10_000.0) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "equity": capital,
            "cash": capital,
            "trade_type": "HOLD",
            "tickers_held": "",
            "n_positions": 0,
            "buys_today": 0,
            "exits_today": 0,
        },
        index=dates,
    )


# ── ejecutar_backtest ─────────────────────────────────────────────────────────

def test_backtest_columnas(sample_feature_vector, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    for col in ["equity", "cash", "trade_type", "tickers_held",
                "n_positions", "buys_today", "exits_today"]:
        assert col in result.columns, f"Falta columna: {col}"


def test_backtest_equity_positivo(sample_feature_vector, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    assert (result["equity"] > 0).all()


def test_backtest_equity_inicial(sample_feature_vector, sample_parameters):
    """La equity inicial nunca supera el capital (sólo pueden haber comisiones)."""
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    assert float(result["equity"].iloc[0]) > 0
    assert float(result["equity"].iloc[0]) <= capital + 0.01


def test_backtest_no_crash_universo_pequenio(sample_parameters):
    fv = _make_multi_fv(n_tickers=2, n_days=30)
    result = ejecutar_backtest(fv, sample_parameters)
    assert len(result) == 30
    assert (result["equity"] > 0).all()


def test_backtest_trade_types_validos(sample_feature_vector, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    tipos_validos = {"BUY", "SELL", "STOP_LOSS", "HOLD"}
    assert set(result["trade_type"].unique()).issubset(tipos_validos)


def test_backtest_n_positions_no_supera_max(sample_feature_vector, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_parameters)
    max_pos = int(sample_parameters["risk"]["max_positions"])
    assert (result["n_positions"] <= max_pos).all()


# ── calcular_metricas ─────────────────────────────────────────────────────────

def test_metricas_tipos(sample_feature_vector, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_parameters)
    result = calcular_metricas(portfolio, sample_parameters)
    assert isinstance(result, tuple) and len(result) == 2
    metrics_df, equity_df = result
    assert hasattr(metrics_df, "columns")
    assert hasattr(equity_df, "columns")


def test_metricas_columnas(sample_feature_vector, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_parameters)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    for col in [
        "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
        "cagr_pct", "n_trades", "trades_per_year",
    ]:
        assert col in metrics_df.columns, f"Falta métrica: {col}"


def test_metricas_sin_trades(sample_parameters):
    """Sin operaciones: n_trades=0."""
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    portfolio = _make_portfolio_df(capital=capital)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    assert int(metrics_df["n_trades"].iloc[0]) == 0


def test_metricas_sharpe_constante(sample_parameters):
    """Equity constante → returns std=0 → Sharpe=0."""
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    portfolio = _make_portfolio_df(capital=capital)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    assert float(metrics_df["sharpe_ratio"].iloc[0]) == 0.0


def test_equity_curve_columnas(sample_feature_vector, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_parameters)
    _, equity_df = calcular_metricas(portfolio, sample_parameters)
    assert "date" in equity_df.columns
    assert "equity" in equity_df.columns


# ── calcular_benchmark ────────────────────────────────────────────────────────

def test_benchmark_columnas(sample_feature_vector, sample_parameters):
    result = calcular_benchmark(sample_feature_vector, sample_parameters)
    assert "date" in result.columns
    assert "equity" in result.columns


def test_benchmark_equity_positiva(sample_feature_vector, sample_parameters):
    result = calcular_benchmark(sample_feature_vector, sample_parameters)
    assert (result["equity"] > 0).all()


def test_benchmark_longitud(sample_feature_vector, sample_parameters):
    result = calcular_benchmark(sample_feature_vector, sample_parameters)
    spy_days = (sample_feature_vector["ticker"] == "SPY").sum()
    assert len(result) == spy_days
