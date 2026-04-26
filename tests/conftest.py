import numpy as np
import pandas as pd
import pytest

from trading_agent.pipelines.feature_engineering.nodes import (
    calcular_indicadores_tecnicos,
    calcular_sentimiento,
    ensamblar_vector_features,
)


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    np.random.seed(42)
    close = 20_000 + np.cumsum(np.random.randn(60) * 500)
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + np.random.randn(60) * 0.003)
    volume = np.random.randint(1_000, 10_000, size=60).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def sample_parameters():
    return {
        "technical": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
        },
        "backtesting": {
            "initial_capital": 10_000,
            "commission": 0.001,
        },
        "risk": {
            "max_position_pct": 0.10,
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
