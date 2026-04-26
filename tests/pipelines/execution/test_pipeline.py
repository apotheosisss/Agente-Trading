import pandas as pd
import pytest

from trading_agent.pipelines.execution.nodes import (
    actualizar_portafolio,
    enviar_orden,
    verificar_riesgo,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def signal_buy():
    return pd.DataFrame([{
        "ticker": "BTC-USD",
        "signal": "BUY",
        "confidence": 0.80,
        "reasoning": "Score alto",
        "score": 5.0,
        "timestamp": "2023-03-01T00:00:00+00:00",
    }])


@pytest.fixture
def signal_hold():
    return pd.DataFrame([{
        "ticker": "BTC-USD",
        "signal": "HOLD",
        "confidence": 0.50,
        "reasoning": "Score neutro",
        "score": 1.0,
        "timestamp": "2023-03-01T00:00:00+00:00",
    }])


@pytest.fixture
def signal_baja_confianza():
    return pd.DataFrame([{
        "ticker": "BTC-USD",
        "signal": "BUY",
        "confidence": 0.40,
        "reasoning": "Score bajo",
        "score": 2.0,
        "timestamp": "2023-03-01T00:00:00+00:00",
    }])


@pytest.fixture
def signal_multi():
    """Ranking multi-ticker: 2 BUY con alta confianza + 1 HOLD."""
    return pd.DataFrame([
        {"ticker": "AAPL", "signal": "BUY", "confidence": 0.85, "score": 6.0,
         "reasoning": "...", "timestamp": "2023-03-01T00:00:00+00:00"},
        {"ticker": "SPY",  "signal": "BUY", "confidence": 0.70, "score": 4.0,
         "reasoning": "...", "timestamp": "2023-03-01T00:00:00+00:00"},
        {"ticker": "GLD",  "signal": "HOLD", "confidence": 0.30, "score": 1.0,
         "reasoning": "...", "timestamp": "2023-03-01T00:00:00+00:00"},
    ])


# ── verificar_riesgo ──────────────────────────────────────────────────────────

def test_riesgo_hold_rechazado(signal_hold, sample_parameters):
    result = verificar_riesgo(signal_hold, sample_parameters)
    assert result["approved"] is False
    assert result["approved_tickers"] == []


def test_riesgo_confianza_baja_rechazada(signal_baja_confianza, sample_parameters):
    result = verificar_riesgo(signal_baja_confianza, sample_parameters)
    assert result["approved"] is False
    assert "onfianza" in result["reason"]


def test_riesgo_buy_aprobado(signal_buy, sample_parameters):
    result = verificar_riesgo(signal_buy, sample_parameters)
    assert result["approved"] is True
    assert "BTC-USD" in result["approved_tickers"]


def test_riesgo_aprobado_tiene_max_position(signal_buy, sample_parameters):
    result = verificar_riesgo(signal_buy, sample_parameters)
    assert "max_position_pct" in result


def test_riesgo_multi_ticker(signal_multi, sample_parameters):
    """Solo los BUY con confianza >= umbral son aprobados."""
    result = verificar_riesgo(signal_multi, sample_parameters)
    assert result["approved"] is True
    assert "AAPL" in result["approved_tickers"]
    assert "SPY" in result["approved_tickers"]
    assert "GLD" not in result["approved_tickers"]


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
    """Orden = capital / max_positions (asignación equitativa)."""
    risk = verificar_riesgo(signal_buy, sample_parameters)
    result = enviar_orden(risk, signal_buy, sample_parameters)
    expected = (
        float(sample_parameters["backtesting"]["initial_capital"])
        / int(sample_parameters["risk"]["max_positions"])
    )
    assert abs(float(result["order_size_usd"].iloc[0]) - expected) < 1.0


def test_orden_modo_paper(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    result = enviar_orden(risk, signal_buy, sample_parameters)
    assert result["mode"].iloc[0] == "paper"


def test_orden_multi_ticker(signal_multi, sample_parameters):
    """Una fila FILLED por cada ticker aprobado."""
    risk = verificar_riesgo(signal_multi, sample_parameters)
    result = enviar_orden(risk, signal_multi, sample_parameters)
    filled = result[result["status"] == "FILLED"]
    assert len(filled) == len(risk["approved_tickers"])


# ── actualizar_portafolio ─────────────────────────────────────────────────────

def test_portafolio_columnas(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    order = enviar_orden(risk, signal_buy, sample_parameters)
    result = actualizar_portafolio(order, sample_parameters)
    for col in ["ticker", "cash", "position_value", "total_value", "last_signal"]:
        assert col in result.columns, f"Falta columna: {col}"


def test_portafolio_buy(signal_buy, sample_parameters):
    risk = verificar_riesgo(signal_buy, sample_parameters)
    order = enviar_orden(risk, signal_buy, sample_parameters)
    result = actualizar_portafolio(order, sample_parameters)
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    max_pos = int(sample_parameters["risk"]["max_positions"])
    order_size = capital / max_pos
    # cash_remaining + position_value = capital
    assert float(result["position_value"].iloc[0]) > 0.0
    assert abs(float(result["total_value"].iloc[0]) - capital) < 1.0


def test_portafolio_rechazado_preserva_capital(signal_hold, sample_parameters):
    risk = verificar_riesgo(signal_hold, sample_parameters)
    order = enviar_orden(risk, signal_hold, sample_parameters)
    result = actualizar_portafolio(order, sample_parameters)
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    assert abs(float(result["cash"].iloc[0]) - capital) < 0.01
    assert float(result["position_value"].iloc[0]) == 0.0
