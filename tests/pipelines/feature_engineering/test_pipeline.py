import numpy as np
import pytest

from trading_agent.pipelines.feature_engineering.nodes import (
    calcular_indicadores_tecnicos,
    calcular_sentimiento,
    ensamblar_vector_features,
)

COLUMNAS_TECNICAS = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",
    "ema_20", "ema_50", "atr",
]


def test_indicadores_columnas(sample_ohlcv, sample_parameters):
    result = calcular_indicadores_tecnicos(sample_ohlcv, sample_parameters["technical"])
    for col in COLUMNAS_TECNICAS:
        assert col in result.columns, f"Falta columna: {col}"


def test_indicadores_sin_nan(sample_ohlcv, sample_parameters):
    result = calcular_indicadores_tecnicos(sample_ohlcv, sample_parameters["technical"])
    assert result.isnull().sum().sum() == 0


def test_rsi_en_rango(sample_ohlcv, sample_parameters):
    result = calcular_indicadores_tecnicos(sample_ohlcv, sample_parameters["technical"])
    assert result["rsi"].between(0, 100).all()


def test_indicadores_reduce_filas(sample_ohlcv, sample_parameters):
    result = calcular_indicadores_tecnicos(sample_ohlcv, sample_parameters["technical"])
    assert len(result) < len(sample_ohlcv)


def test_sentimiento_columna(sample_ohlcv):
    result = calcular_sentimiento(sample_ohlcv)
    assert "sentiment_score" in result.columns


def test_sentimiento_rango(sample_ohlcv):
    result = calcular_sentimiento(sample_ohlcv)
    assert result["sentiment_score"].between(-1.0, 1.0).all()


def test_sentimiento_longitud(sample_ohlcv):
    result = calcular_sentimiento(sample_ohlcv)
    assert len(result) > 0


def test_vector_columnas(sample_feature_vector):
    esperadas = [
        "open", "high", "low", "close", "volume",
        "rsi", "macd", "bb_upper", "ema_20", "ema_50", "atr",
        "sentiment_score",
    ]
    for col in esperadas:
        assert col in sample_feature_vector.columns, f"Falta columna: {col}"


def test_vector_sin_nan_en_sentimiento(sample_feature_vector):
    assert sample_feature_vector["sentiment_score"].isnull().sum() == 0


def test_vector_alinea_fechas(sample_ohlcv, sample_parameters):
    tech = calcular_indicadores_tecnicos(sample_ohlcv, sample_parameters["technical"])
    sent = calcular_sentimiento(sample_ohlcv)
    vector = ensamblar_vector_features(tech, sent)
    assert vector.index.equals(tech.index)
