import numpy as np
import pandas as pd
import pytest

from trading_agent.pipelines.feature_engineering.nodes import (
    calcular_indicadores_tecnicos,
    calcular_sentimiento,
    ensamblar_vector_features,
)

TICKERS = ["AAPL", "SPY", "BTC-USD"]


@pytest.fixture
def sample_ohlcv():
    """3 tickers × 60 días = 180 filas con columna 'ticker' e índice DatetimeIndex."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    frames = []
    for i, ticker in enumerate(TICKERS):
        base = 20_000.0 + i * 5_000.0
        close = base + np.cumsum(np.random.randn(60) * 300)
        close = np.abs(close).clip(min=1.0)
        df = pd.DataFrame(
            {
                "ticker": ticker,
                "open": close * (1 + np.random.randn(60) * 0.002),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.random.randint(1_000, 10_000, size=60).astype(float),
            },
            index=dates,
        )
        df.index.name = "date"
        frames.append(df)
    result = pd.concat(frames).sort_index()
    # Asegurar precios positivos
    for col in ["open", "high", "low", "close"]:
        result[col] = result[col].abs().clip(lower=1.0)
    return result


@pytest.fixture
def sample_parameters():
    return {
        "universe": TICKERS,
        "ticker": "SPY",
        "technical": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "ema_200": 200,
        },
        "backtesting": {
            "initial_capital": 10_000,
            "commission": 0.001,
            "rebalance_day": 0,
        },
        "risk": {
            "max_positions": 3,
            "max_position_pct": 0.33,
            "stop_loss_atr_mult": 2.0,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
        },
        "llm": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "confidence_threshold": 0.65,
        },
    }


@pytest.fixture
def sample_feature_vector(sample_ohlcv, sample_parameters):
    tech = calcular_indicadores_tecnicos(sample_ohlcv, sample_parameters["technical"])
    sent = calcular_sentimiento(sample_ohlcv)
    return ensamblar_vector_features(tech, sent)
