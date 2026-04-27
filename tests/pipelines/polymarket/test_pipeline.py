# tests/pipelines/polymarket/test_pipeline.py
"""Tests del pipeline de señales Polymarket."""

import pandas as pd
import pytest

from trading_agent.pipelines.polymarket.nodes import (
    _match_market,
    _score_from_probability,
    generar_reporte_polymarket,
    obtener_seniales_polymarket,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_params(universe=None):
    return {
        "universe": universe or ["BTC-USD", "ETH-USD", "NVDA", "SPY", "QQQ"],
        "polymarket": {
            "base_url": "https://gamma-api.polymarket.com",
            "min_volume": 5_000,
            "max_score_contribution": 1.5,
        },
    }


def _make_signals(tickers=None, score=0.0):
    """DataFrame de señales sintético para tests."""
    tickers = tickers or ["BTC-USD", "ETH-USD", "NVDA", "SPY", "QQQ", "_macro"]
    return pd.DataFrame({
        "ticker": tickers,
        "poly_score": [score] * len(tickers),
        "n_markets": [2] * len(tickers),
        "top_market": ["Test market [▲ 70%]"] * len(tickers),
        "timestamp": ["2024-01-01T00:00:00+00:00"] * len(tickers),
    })


# ── _match_market ─────────────────────────────────────────────────────────────

def test_match_market_coincide():
    # "rate cut" sí aparece como substring en "federal reserve rate cut"
    assert _match_market("Will there be a federal reserve rate cut in 2024?", ["rate cut"]) is True


def test_match_market_case_insensitive():
    assert _match_market("Bitcoin price prediction 2024", ["bitcoin"]) is True


def test_match_market_no_coincide():
    assert _match_market("Will it rain tomorrow?", ["rate cut", "bitcoin"]) is False


def test_match_market_lista_vacia():
    assert _match_market("Any question", []) is False


# ── _score_from_probability ───────────────────────────────────────────────────

def test_score_bullish_alto():
    """Probabilidad alta en mercado bullish → score positivo."""
    assert _score_from_probability(0.80, "bullish") == 1.0


def test_score_bullish_medio():
    assert _score_from_probability(0.60, "bullish") == 0.5


def test_score_bullish_bajo():
    """Probabilidad baja en mercado bullish → score negativo."""
    assert _score_from_probability(0.20, "bullish") == -0.5


def test_score_bearish_alto():
    """Probabilidad alta en mercado bearish → score negativo."""
    assert _score_from_probability(0.80, "bearish") == -1.0


def test_score_bearish_bajo():
    """Probabilidad baja en mercado bearish → score positivo."""
    assert _score_from_probability(0.20, "bearish") == 0.5


def test_score_neutral_rango():
    """Probabilidad en zona neutral → 0.0."""
    assert _score_from_probability(0.50, "bullish") == 0.0
    assert _score_from_probability(0.50, "bearish") == 0.0


# ── obtener_seniales_polymarket ────────────────────────────────────────────────

def test_señales_sin_api(monkeypatch):
    """Si la API no está disponible, devuelve señales neutras (no lanza excepción)."""
    # Hacer que requests falle
    import sys
    import types

    fake_requests = types.ModuleType("requests")

    def failing_get(*args, **kwargs):
        raise ConnectionError("API no disponible")

    fake_requests.get = failing_get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    params = _make_params()
    result = obtener_seniales_polymarket(params)

    assert isinstance(result, pd.DataFrame)
    # Debe devolver filas (una por ticker + _macro), todas con score 0.0
    assert "ticker" in result.columns
    assert "poly_score" in result.columns
    assert (result["poly_score"] == 0.0).all()


def test_señales_columnas_requeridas(monkeypatch):
    """El DataFrame de salida siempre tiene las columnas necesarias."""
    import sys
    import types

    fake_requests = types.ModuleType("requests")

    def failing_get(*args, **kwargs):
        raise ConnectionError("sin red")

    fake_requests.get = failing_get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = obtener_seniales_polymarket(_make_params())
    for col in ["ticker", "poly_score", "n_markets", "top_market", "timestamp"]:
        assert col in result.columns, f"Falta columna: {col}"


def test_señales_incluye_macro(monkeypatch):
    """Siempre debe existir una fila con ticker='_macro'."""
    import sys
    import types

    fake_requests = types.ModuleType("requests")

    def failing_get(*args, **kwargs):
        raise ConnectionError("sin red")

    fake_requests.get = failing_get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = obtener_seniales_polymarket(_make_params())
    assert "_macro" in result["ticker"].values


def test_señales_score_en_rango(monkeypatch):
    """poly_score siempre debe estar dentro de [-max_contribution, +max_contribution]."""
    import sys
    import types

    fake_requests = types.ModuleType("requests")

    def failing_get(*args, **kwargs):
        raise ConnectionError("sin red")

    fake_requests.get = failing_get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    params = _make_params()
    max_c = params["polymarket"]["max_score_contribution"]
    result = obtener_seniales_polymarket(params)
    assert (result["poly_score"].abs() <= max_c).all()


# ── generar_reporte_polymarket ────────────────────────────────────────────────

def test_reporte_tipo_string():
    signals = _make_signals()
    result = generar_reporte_polymarket(signals)
    assert isinstance(result, str)


def test_reporte_contiene_header():
    signals = _make_signals()
    result = generar_reporte_polymarket(signals)
    assert "POLYMARKET" in result


def test_reporte_dataframe_vacio():
    """DataFrame vacío no debe lanzar excepción."""
    result = generar_reporte_polymarket(pd.DataFrame())
    assert isinstance(result, str)
    assert "Sin datos" in result


def test_reporte_con_señal_activa():
    """Con poly_score distinto de cero, el reporte debe mencionarlo."""
    signals = _make_signals(["BTC-USD", "_macro"], score=0.8)
    result = generar_reporte_polymarket(signals)
    # Debe contener información sobre señal positiva
    assert "🟢" in result or "BTC-USD" in result
