# src/trading_agent/pipelines/shared_utils.py
"""Utilidades compartidas entre pipelines de backtesting y llm_agents."""

import pandas as pd


MAX_SCORE = 8.5  # RSI(3) + MACD(1) + EMA(2) + BB(1.5) + sentiment(1)


def score_row(row: pd.Series) -> float:
    """Score de trading para una fila del feature vector.

    Devuelve un valor en [-8.5, +8.5].
    Retorna -999.0 si el activo está por debajo de su EMA 200 (filtro de tendencia).

    Ponderación:
    - RSI (momentum/sobreventa): hasta ±3.0
    - MACD (cruce de señal): ±1.0
    - EMA alineación (close/EMA20/EMA50): ±2.0
    - Bollinger Bands: ±1.5
    - Sentimiento (proxy momentum precio): ±1.0
    """
    close = float(row["close"])
    ema_200 = float(row.get("ema_200", row.get("ema_50", close)))

    # ── Filtro de tendencia largo plazo ─────────────────────────────────────
    if close < ema_200:
        return -999.0

    rsi = float(row["rsi"])
    macd = float(row["macd"])
    macd_sig = float(row["macd_signal"])
    ema_20 = float(row["ema_20"])
    ema_50 = float(row["ema_50"])
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])
    sentiment = float(row.get("sentiment_score", 0.0))

    score = 0.0

    # ── RSI ───────────────────────────────────────────────────────────────────
    if rsi < 30:
        score += 3.0
    elif rsi < 45:
        score += 1.5
    elif rsi > 75:
        score -= 3.0
    elif rsi > 65:
        score -= 1.5

    # ── MACD ─────────────────────────────────────────────────────────────────
    score += 1.0 if macd > macd_sig else -1.0

    # ── EMA alineación ────────────────────────────────────────────────────────
    if close > ema_20 and ema_20 > ema_50:
        score += 2.0
    elif close < ema_20 and ema_20 < ema_50:
        score -= 2.0

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if close < bb_lower:
        score += 1.5
    elif close > bb_upper:
        score -= 1.5

    # ── Sentimiento ───────────────────────────────────────────────────────────
    if sentiment > 0.3:
        score += 1.0
    elif sentiment < -0.3:
        score -= 1.0

    return score
