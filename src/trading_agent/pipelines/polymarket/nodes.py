# src/trading_agent/pipelines/polymarket/nodes.py
"""Pipeline de seniales Polymarket.

Consulta la API publica de Polymarket, identifica mercados relevantes
para cada activo del universo, y calcula un 'poly_score' por ticker
que representa el sentimiento del mercado de predicciones.

Integracion:
    - Solo afecta la decision de HOY (agente_decision).
    - El backtest historico NO usa estos datos.
    - Si la API no esta disponible, devuelve seniales neutras (0.0)
      para que el sistema funcione sin conexion.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

POLYMARKET_API = "https://gamma-api.polymarket.com"

# Mapeo de palabras clave por categoria.
# direction: "bullish" (YES alto = bueno) o "bearish" (YES alto = malo)
# tickers:   lista de activos que afecta. None = todos.

_KEYWORD_CATEGORIES: list[dict[str, Any]] = [
    # ── Cripto ────────────────────────────────────────────────────────────────
    {
        "name": "bitcoin_target",
        # "Will Bitcoin hit $150k / $200k / $300k..."
        "keywords": ["bitcoin hit $", "btc hit $", "bitcoin reach $", "bitcoin above $"],
        "direction": "bullish",
        "tickers": ["BTC-USD", "ETH-USD"],
        "weight": 1.2,
    },
    {
        "name": "bitcoin_crash",
        "keywords": ["bitcoin below $", "bitcoin crash", "bitcoin drop", "btc below $"],
        "direction": "bearish",
        "tickers": ["BTC-USD", "ETH-USD"],
        "weight": 1.0,
    },
    {
        "name": "crypto_etf",
        "keywords": ["bitcoin etf", "ethereum etf", "crypto etf", "spot btc"],
        "direction": "bullish",
        "tickers": ["BTC-USD", "ETH-USD"],
        "weight": 1.0,
    },
    # ── Geopolítica (riesgo macro) ─────────────────────────────────────────────
    {
        "name": "ukraine_ceasefire",
        # Resolución del conflicto → risk-on, bullish para mercados
        "keywords": ["ceasefire", "ukraine peace", "russia ukraine deal"],
        "direction": "bullish",
        "tickers": None,
        "weight": 0.8,
    },
    {
        "name": "taiwan_invasion",
        # Escalada geopolítica → risk-off, muy bearish (tech supply chain)
        "keywords": ["china invade taiwan", "taiwan invasion", "china taiwan war"],
        "direction": "bearish",
        "tickers": ["AAPL", "NVDA", "TSM", "XLK", "QQQ", "SPY"],
        "weight": 1.2,
    },
    {
        "name": "us_war",
        "keywords": ["world war", "us military conflict", "us war"],
        "direction": "bearish",
        "tickers": None,
        "weight": 1.0,
    },
    # ── Política EEUU (incertidumbre = bearish) ────────────────────────────────
    {
        "name": "trump_impeachment",
        "keywords": ["trump impeached", "trump removed", "trump resign"],
        "direction": "bearish",
        "tickers": None,
        "weight": 0.7,
    },
    # ── Macro económico (si reaparecen estos mercados) ─────────────────────────
    {
        "name": "fed_cut",
        "keywords": ["rate cut", "fed cut", "federal reserve cut", "fed pivot"],
        "direction": "bullish",
        "tickers": None,
        "weight": 1.0,
    },
    {
        "name": "recession",
        "keywords": ["recession", "us recession", "gdp contraction"],
        "direction": "bearish",
        "tickers": None,
        "weight": 1.0,
    },
    {
        "name": "trade_war",
        "keywords": ["tariff", "trade war", "trade deal", "china tariff"],
        "direction": "bearish",
        "tickers": ["AAPL", "NVDA", "XLK", "SPY", "QQQ"],
        "weight": 0.9,
    },
]


# Helpers

def _fetch_markets(base_url: str, min_volume: float) -> list[dict]:
    """Descarga los mercados activos desde la API de Polymarket.

    Devuelve lista vacia si la API no esta disponible (fail-safe).
    """
    try:
        import requests

        resp = requests.get(
            f"{base_url}/markets",
            params={"active": "true", "closed": "false", "limit": 1000},
            timeout=15,
        )
        resp.raise_for_status()
        markets = resp.json()

        filtered = [
            m for m in markets
            if float(m.get("volume", 0) or 0) >= min_volume
        ]
        logger.info(
            "Polymarket: %d mercados totales | %d con volumen > $%,.0f",
            len(markets), len(filtered), min_volume,
        )
        return filtered

    except Exception as exc:
        logger.warning(
            "Polymarket API no disponible (%s). "
            "Se usaran seniales neutras (poly_score=0).",
            type(exc).__name__,
        )
        return []


def _parse_yes_price(market: dict) -> float | None:
    """Extrae el precio de YES (probabilidad) de un mercado binario.

    outcomePrices puede llegar como lista o como string JSON serializado.
    """
    import json as _json
    try:
        prices = market.get("outcomePrices", [])
        if isinstance(prices, str):
            prices = _json.loads(prices)
        if prices:
            return float(prices[0])
    except (ValueError, TypeError, _json.JSONDecodeError):
        pass
    return None


def _match_market(question: str, keywords: list[str]) -> bool:
    """Comprueba si la pregunta de un mercado contiene alguna palabra clave."""
    q = question.lower()
    return any(kw.lower() in q for kw in keywords)


def _score_from_probability(yes_price: float, direction: str) -> float:
    """Convierte una probabilidad en un score de senal.

    Escala:
        prob > 0.70 -> +/-1.0
        prob 0.55-0.70 -> +/-0.5
        prob 0.30-0.55 -> 0.0 (neutral)
        prob 0.15-0.30 -> -/+0.5
        prob < 0.15    -> -/+1.0
    """
    if direction == "bullish":
        if yes_price > 0.70:
            return 1.0
        elif yes_price > 0.55:
            return 0.5
        elif yes_price < 0.30:
            return -0.5
        elif yes_price < 0.15:
            return -1.0
        return 0.0
    else:  # bearish
        if yes_price > 0.70:
            return -1.0
        elif yes_price > 0.55:
            return -0.5
        elif yes_price < 0.30:
            return 0.5
        elif yes_price < 0.15:
            return 1.0
        return 0.0


# Nodos publicos

def obtener_seniales_polymarket(parameters: dict) -> pd.DataFrame:
    """Descarga mercados de Polymarket y calcula poly_score por ticker.

    Args:
        parameters: parametros del proyecto (usa seccion 'polymarket').

    Returns:
        DataFrame con columnas:
            ticker, poly_score, n_markets, top_market, timestamp
        Una fila por ticker del universo + fila "_macro" para senal global.
        poly_score en [-1.5, +1.5]
    """
    poly_cfg = parameters.get("polymarket", {})
    base_url = poly_cfg.get("base_url", POLYMARKET_API)
    min_volume = float(poly_cfg.get("min_volume", 5_000))
    max_contribution = float(poly_cfg.get("max_score_contribution", 1.5))
    universe: list[str] = parameters.get("universe", [])

    markets = _fetch_markets(base_url, min_volume)

    accumulator: dict[str, dict] = {
        t: {"sum_score": 0.0, "sum_weight": 0.0, "top_markets": []}
        for t in universe
    }
    accumulator["_macro"] = {"sum_score": 0.0, "sum_weight": 0.0, "top_markets": []}

    for market in markets:
        question = market.get("question", "")
        yes_price = _parse_yes_price(market)
        if yes_price is None:
            continue

        for cat in _KEYWORD_CATEGORIES:
            if not _match_market(question, cat["keywords"]):
                continue

            raw_score = _score_from_probability(yes_price, cat["direction"])
            if raw_score == 0.0:
                continue

            weighted_score = raw_score * cat["weight"]
            affected = cat["tickers"] or list(accumulator.keys())

            for t in affected:
                if t not in accumulator:
                    continue
                acc = accumulator[t]
                acc["sum_score"] += weighted_score
                acc["sum_weight"] += cat["weight"]
                direction_chr = "^" if raw_score > 0 else "v"
                acc["top_markets"].append(
                    f"{question[:60]} [{direction_chr} {yes_price:.0%}]"
                )

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for ticker, acc in accumulator.items():
        if acc["sum_weight"] > 0:
            raw = acc["sum_score"] / acc["sum_weight"]
        else:
            raw = 0.0

        poly_score = max(-max_contribution, min(max_contribution, raw))
        top = " | ".join(acc["top_markets"][:2]) if acc["top_markets"] else "Sin mercados relevantes"

        rows.append({
            "ticker": ticker,
            "poly_score": round(poly_score, 3),
            "n_markets": len(acc["top_markets"]),
            "top_market": top,
            "timestamp": now,
        })

    result = pd.DataFrame(rows)
    n_nonzero = (result["poly_score"] != 0.0).sum()
    logger.info(
        "Polymarket seniales: %d tickers analizados | %d con senal activa",
        len(result), n_nonzero,
    )
    return result


# Alias para compatibilidad con pipeline.py (nombre con tilde en funcion original)
obtener_seniales_polymarket.__name__ = "obtener_seniales_polymarket"


def generar_reporte_polymarket(poly_signals: pd.DataFrame) -> str:
    """Genera un reporte legible de las seniales Polymarket activas.

    Args:
        poly_signals: salida de obtener_seniales_polymarket().

    Returns:
        String ASCII con el reporte formateado para incluir en agente_decision.
    """
    lines = ["=== SENIALES POLYMARKET (Mercados de Prediccion) ==="]

    if poly_signals.empty:
        lines.append("  Sin datos disponibles (API no accesible).")
        return "\n".join(lines)

    macro = poly_signals[poly_signals["ticker"] == "_macro"]
    tickers = poly_signals[poly_signals["ticker"] != "_macro"]

    if not macro.empty:
        ms = float(macro["poly_score"].iloc[0])
        mn = int(macro["n_markets"].iloc[0])
        mt = macro["top_market"].iloc[0]
        trend = "[+] BULLISH" if ms > 0.2 else "[-] BEARISH" if ms < -0.2 else "[=] NEUTRAL"
        lines.append(f"\n  {trend} MACRO GLOBAL (score={ms:+.2f} | {mn} mercados)")
        if mt != "Sin mercados relevantes":
            for m in mt.split(" | "):
                lines.append(f"    * {m}")

    if not tickers.empty:
        lines.append("\n  SENIALES POR ACTIVO:")
        active = tickers[tickers["poly_score"] != 0.0].sort_values(
            "poly_score", ascending=False
        )
        if active.empty:
            lines.append("  Todos los activos en neutral.")
        else:
            for _, row in active.iterrows():
                trend = "[+]" if row["poly_score"] > 0.2 else "[-]"
                lines.append(
                    f"    {trend} [{row['ticker']}] score={row['poly_score']:+.2f} "
                    f"({int(row['n_markets'])} mercados)"
                )
                if row["top_market"] != "Sin mercados relevantes":
                    lines.append(f"       -> {row['top_market'].split(' | ')[0]}")

    return "\n".join(lines)
