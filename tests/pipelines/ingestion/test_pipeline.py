import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from trading_agent.pipelines.ingestion.nodes import (
    obtener_datos_mercado,
    validar_datos_mercado,
)


@pytest.fixture
def ohlcv_limpio():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    np.random.seed(0)
    close = 20_000 + np.cumsum(np.random.randn(30) * 300)
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.ones(30) * 5000.0,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_validar_datos_ok(ohlcv_limpio):
    result = validar_datos_mercado(ohlcv_limpio)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result) == len(ohlcv_limpio)


def test_validar_datos_columnas_faltantes(ohlcv_limpio):
    df_incompleto = ohlcv_limpio.drop(columns=["close", "volume"])
    with pytest.raises(ValueError, match="Columnas faltantes"):
        validar_datos_mercado(df_incompleto)


def test_validar_datos_precios_negativos(ohlcv_limpio):
    df_malo = ohlcv_limpio.copy()
    df_malo.iloc[0, df_malo.columns.get_loc("close")] = -100.0
    with pytest.raises(ValueError, match="precios negativos"):
        validar_datos_mercado(df_malo)


def test_validar_datos_elimina_nan(ohlcv_limpio):
    df_con_nan = ohlcv_limpio.copy()
    df_con_nan.iloc[5, df_con_nan.columns.get_loc("close")] = np.nan
    result = validar_datos_mercado(df_con_nan)
    assert len(result) == len(ohlcv_limpio) - 1
    assert result.isnull().sum().sum() == 0


def test_validar_datos_reordena_indice(ohlcv_limpio):
    df_desordenado = ohlcv_limpio.iloc[::-1].copy()
    result = validar_datos_mercado(df_desordenado)
    assert result.index.is_monotonic_increasing


def test_obtener_datos_vacio():
    df_vacio = pd.DataFrame()
    with patch("yfinance.download", return_value=df_vacio):
        with pytest.raises(ValueError):
            obtener_datos_mercado("BTC-USD", "2023-01-01", "2023-01-31")
