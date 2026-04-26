import pandas as pd
import pytest

from trading_agent.pipelines.execution.nodes import (
    actualizar_portafolio,
    enviar_orden,
    verificar_riesgo,
)


@pytest.fixture
def signal_buy():
    return pd.DataFrame([{
        "signal": "BUY",
        "confidence": 0.80,
        "reasoning": "Score alto",
        "timestamp": "2023-03-01T00:00:00+00:00",
        "ticker": "BTC-USD",
        "score": 5.0,
    }])


@pytest.fixture
def signal_hold():
    return pd.DataFrame([{
        "signal": "HOLD",
        "confidence": 0.50,
        "reasoning": "Score neutro",
        "timestamp": "2023-03-01T00:00:00+00:00",
        "ticker": "BTC-USD",
        "score": 1.0,
    }])


@pytest.fixture
def signal_baja_confianza():
    return pd.DataFrame([{
        "signal": "BUY",
        "confidence": 0.40,
        "reasoning": "Score bajo",
        "timestamp": "2023-03-01T00:00:00+00:00",
        "ticker": "BTC-USD",
        "score": 2.0,
    }])


# ── verificar_riesgo ──────────────────────────────────────────────────────────

def test_riesgo_hold_rechazado(signal_hold, sample_parameters):
    result = verificar_riesgo(signal_hold, sample_parameters)
    assert result["approved"] is False
    assert result["signal"] == "HOLD"


def test_riesgo_confianza_baja_rechazada(signal_baja_confianza, sample_parameters):
    result = verificar_riesgo(signal_baja_confianza, sample_parameters)
    assert result["approved"] is False
    assert "onfianza" in result["reason"]


def test_riesgo_buy_aprobado(signal_buy, sample_parameters):
    result = verificar_riesgo(signal_buy, sample_parameters)
    assert result["approved"] is True
    assert result["signal"] == "BUY"


def test_riesgo_aprobado_tiene_max_position(signal_buy, sample_parameters):
    result = verificar_riesgo(signal_buy, sample_parameters)
    assert "max_position_pct" in result


# ── enviar_orden ──────────────────────────────────────────────────────────────

def test_orden_rechazada_status(signal_hold, sample_parameters):
    risk = verificar_riesgo(signal_hold, sample_parameters)
    result = enviar_orden(risk, signal_hold, sample_parameters)
    assert result["status"].iloc[0] == "REJECTED"
    assert float(result["order_size_usd"].iloc[0]) == 0.0


def test_orden_aceptada_status(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    result = enviar_orden(risk, signal_buy, sample_parameters)
    assert result["status"].iloc[0] == "FILLED"


def test_orden_size_correcto(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    result = enviar_orden(risk, signal_buy, sample_parameters)
    expected = (
        sample_parameters["backtesting"]["initial_capital"]
        * sample_parameters["risk"]["max_position_pct"]
    )
    assert abs(float(result["order_size_usd"].iloc[0]) - expected) < 0.01


def test_orden_modo_paper(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    result = enviar_orden(risk, signal_buy, sample_parameters)
    assert result["mode"].iloc[0] == "paper"


# ── actualizar_portafolio ─────────────────────────────────────────────────────

def test_portafolio_columnas(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    order = enviar_orden(risk, signal_buy, sample_parameters)
    result = actualizar_portafolio(order, sample_parameters)
    for col in ["cash", "position", "position_value", "total_value", "last_signal"]:
        assert col in result.columns


def test_portafolio_buy(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    order = enviar_orden(risk, signal_buy, sample_parameters)
    result = actualizar_portafolio(order, sample_parameters)
    capital = sample_parameters["backtesting"]["initial_capital"]
    order_size = capital * sample_parameters["risk"]["max_position_pct"]
    assert abs(float(result["cash"].iloc[0]) - (capital - order_size)) < 0.01
    assert float(result["position_value"].iloc[0]) > 0.0


def test_portafolio_rechazado_preserva_capital(signal_hold, sample_parameters):
    risk = verificar_riesgo(signal_hold, sample_parameters)
    order = enviar_orden(risk, signal_hold, sample_parameters)
    result = actualizar_portafolio(order, sample_parameters)
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    assert abs(float(result["cash"].iloc[0]) - capital) < 0.01
    assert float(result["position_value"].iloc[0]) == 0.0
