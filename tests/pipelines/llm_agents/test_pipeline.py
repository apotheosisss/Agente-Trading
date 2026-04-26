import numpy as np
import pandas as pd
import pytest

from trading_agent.pipelines.llm_agents.nodes import (
    agente_decision,
    agente_riesgo,
    agente_sentimiento,
    agente_tecnico,
)


def test_agente_tecnico_tipo(sample_feature_vector):
    result = agente_tecnico(sample_feature_vector)
    assert isinstance(result, str)


def test_agente_tecnico_contiene_precio(sample_feature_vector):
    result = agente_tecnico(sample_feature_vector)
    assert "ANÁLISIS TÉCNICO" in result
    assert "$" in result


def test_agente_tecnico_contiene_senales(sample_feature_vector):
    result = agente_tecnico(sample_feature_vector)
    assert "RSI" in result
    assert "MACD" in result


def test_agente_sentimiento_tipo(sample_feature_vector):
    result = agente_sentimiento(sample_feature_vector)
    assert isinstance(result, str)


def test_agente_sentimiento_clasifica(sample_feature_vector):
    result = agente_sentimiento(sample_feature_vector)
    clasificaciones = ["MUY POSITIVO", "POSITIVO", "NEUTRAL", "NEGATIVO", "MUY NEGATIVO"]
    assert any(c in result for c in clasificaciones)


def test_agente_sentimiento_promedios(sample_feature_vector):
    result = agente_sentimiento(sample_feature_vector)
    assert "7d" in result
    assert "30d" in result


def test_agente_riesgo_tipo(sample_feature_vector):
    result = agente_riesgo(sample_feature_vector)
    assert isinstance(result, str)


def test_agente_riesgo_nivel(sample_feature_vector):
    result = agente_riesgo(sample_feature_vector)
    niveles = ["EXTREMO", "ALTO", "MODERADO", "BAJO"]
    assert any(n in result for n in niveles)


def test_agente_riesgo_contiene_volatilidad(sample_feature_vector):
    result = agente_riesgo(sample_feature_vector)
    assert "Volatilidad" in result
    assert "ATR" in result


def test_decision_columnas(sample_feature_vector, sample_parameters):
    tech = agente_tecnico(sample_feature_vector)
    sent = agente_sentimiento(sample_feature_vector)
    risk = agente_riesgo(sample_feature_vector)
    result = agente_decision(tech, sent, risk, sample_feature_vector, sample_parameters["llm"], "BTC-USD")
    for col in ["signal", "confidence", "reasoning", "ticker", "score", "timestamp"]:
        assert col in result.columns


def test_decision_senial_valida(sample_feature_vector, sample_parameters):
    tech = agente_tecnico(sample_feature_vector)
    sent = agente_sentimiento(sample_feature_vector)
    risk = agente_riesgo(sample_feature_vector)
    result = agente_decision(tech, sent, risk, sample_feature_vector, sample_parameters["llm"], "BTC-USD")
    assert result["signal"].iloc[0] in ("BUY", "SELL", "HOLD")


def test_decision_confianza_rango(sample_feature_vector, sample_parameters):
    tech = agente_tecnico(sample_feature_vector)
    sent = agente_sentimiento(sample_feature_vector)
    risk = agente_riesgo(sample_feature_vector)
    result = agente_decision(tech, sent, risk, sample_feature_vector, sample_parameters["llm"], "BTC-USD")
    conf = float(result["confidence"].iloc[0])
    assert 0.0 <= conf <= 0.95


def _make_vector_con_score(score_params: dict) -> pd.DataFrame:
    """Construye un feature vector mínimo con valores que producen el score deseado."""
    dates = pd.date_range("2023-03-01", periods=35, freq="D")
    close = np.full(35, 20_000.0)
    df = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close, "volume": 5000.0,
        "rsi": score_params.get("rsi", 50.0),
        "macd": score_params.get("macd", 0.1),
        "macd_signal": score_params.get("macd_signal", 0.0),
        "macd_hist": 0.1,
        "bb_upper": close * 1.05,
        "bb_mid": close,
        "bb_lower": close * 0.95,
        "ema_20": score_params.get("ema_20", close),
        "ema_50": score_params.get("ema_50", close),
        "atr": 200.0,
        "sentiment_score": score_params.get("sentiment", 0.0),
    }, index=dates)
    return df


def test_decision_buy():
    """RSI sobreventa + MACD alcista + tendencia alcista → BUY."""
    fv = _make_vector_con_score({"rsi": 20.0, "macd": 1.0, "macd_signal": 0.0,
                                  "ema_20": 20_100.0, "ema_50": 20_050.0, "sentiment": 0.5})
    params = {"model": "gpt-4o-mini", "temperature": 0.1, "confidence_threshold": 0.65}
    tech = agente_tecnico(fv)
    sent = agente_sentimiento(fv)
    risk = agente_riesgo(fv)
    result = agente_decision(tech, sent, risk, fv, params, "BTC-USD")
    assert result["signal"].iloc[0] == "BUY"


def test_decision_sell():
    """RSI sobrecompra + MACD bajista + tendencia bajista → SELL."""
    fv = _make_vector_con_score({"rsi": 80.0, "macd": -1.0, "macd_signal": 0.0,
                                  "ema_20": 19_900.0, "ema_50": 20_000.0, "sentiment": -0.5})
    params = {"model": "gpt-4o-mini", "temperature": 0.1, "confidence_threshold": 0.65}
    tech = agente_tecnico(fv)
    sent = agente_sentimiento(fv)
    risk = agente_riesgo(fv)
    result = agente_decision(tech, sent, risk, fv, params, "BTC-USD")
    assert result["signal"].iloc[0] == "SELL"


def test_decision_hold():
    """RSI neutral + MACD débil → HOLD."""
    fv = _make_vector_con_score({"rsi": 50.0, "macd": 0.01, "macd_signal": 0.0,
                                  "ema_20": 20_000.0, "ema_50": 20_000.0, "sentiment": 0.0})
    params = {"model": "gpt-4o-mini", "temperature": 0.1, "confidence_threshold": 0.65}
    tech = agente_tecnico(fv)
    sent = agente_sentimiento(fv)
    risk = agente_riesgo(fv)
    result = agente_decision(tech, sent, risk, fv, params, "BTC-USD")
    assert result["signal"].iloc[0] == "HOLD"


def test_decision_sin_openai(sample_feature_vector, sample_parameters, monkeypatch):
    """Sin OPENAI_API_KEY el fallback cuantitativo debe funcionar igual."""
    import sys
    monkeypatch.setitem(sys.modules, "openai", None)
    tech = agente_tecnico(sample_feature_vector)
    sent = agente_sentimiento(sample_feature_vector)
    risk = agente_riesgo(sample_feature_vector)
    result = agente_decision(tech, sent, risk, sample_feature_vector, sample_parameters["llm"], "BTC-USD")
    assert result["signal"].iloc[0] in ("BUY", "SELL", "HOLD")
