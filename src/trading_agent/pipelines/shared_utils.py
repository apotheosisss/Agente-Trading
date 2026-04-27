# src/trading_agent/pipelines/shared_utils.py
"""Utilidades compartidas entre pipelines de backtesting y llm_agents."""

import pandas as pd


MAX_SCORE = 9.0  # momentum_90d(3) + EMA(2) + MACD(1) + RSI(1.5) + momentum_252d_bonus(1) + sentiment(0.5)


def score_row(row: pd.Series) -> float:
    """Score de momentum para una fila del feature vector.

    Bifurcación A v3 — Momentum Concentrado refinado.
    Filosofía:
    - momentum_90d es la señal PRIMARIA de ranking (trend following clásico).
    - momentum_252d actúa como filtro bloqueante: si el año es negativo, score baja.
      Pero no se exige momentum_252d positivo para entrar (permite re-entrar en
      recuperaciones antes de que el retorno anual se vuelva positivo).
    - EMA alignment confirma estructura de tendencia.
    - RSI como confirmación de fuerza relativa (no mean-reversion).

    Devuelve un valor en [-9.0, +9.0].
    Retorna -999.0 si el activo está por debajo de su EMA 200.

    Ponderación:
    - Momentum 90d (retorno trimestral): hasta ±3.0  — señal primaria
    - EMA alineación (close/EMA20/EMA50): ±2.0       — estructura de tendencia
    - MACD (cruce de señal): ±1.0                    — momento de corto plazo
    - RSI momentum (fuerza relativa): hasta ±1.5     — RSI alto = fuerza
    - Momentum 252d (modificador anual): ±1.0        — frena si año es muy negativo
    - Sentimiento: ±0.5                              — señal auxiliar
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
    sentiment = float(row.get("sentiment_score", 0.0))
    momentum_90d = float(row.get("momentum_90d", 0.0))
    momentum_252d = float(row.get("momentum_252d", 0.0))

    score = 0.0

    # ── Momentum 90d — señal primaria de ranking ─────────────────────────────
    # Retorno trimestral: identifica activos en aceleración reciente.
    if momentum_90d > 0.30:      # +30%+ trimestral → momentum explosivo
        score += 3.0
    elif momentum_90d > 0.15:    # +15-30% → momentum fuerte
        score += 2.0
    elif momentum_90d > 0.05:    # +5-15% → momentum moderado
        score += 1.0
    elif momentum_90d < -0.10:   # >-10% caída → momentum negativo
        score -= 2.0
    elif momentum_90d < 0.0:     # 0 a -10% → leve debilidad
        score -= 1.0

    # ── EMA alineación ────────────────────────────────────────────────────────
    if close > ema_20 and ema_20 > ema_50:
        score += 2.0
    elif close < ema_20 and ema_20 < ema_50:
        score -= 2.0

    # ── MACD ─────────────────────────────────────────────────────────────────
    score += 1.0 if macd > macd_sig else -1.0

    # ── RSI como confirmación de fuerza (NO mean-reversion) ──────────────────
    if rsi > 65:
        score += 1.5
    elif rsi > 55:
        score += 0.5
    elif rsi < 40:
        score -= 1.5
    elif rsi < 50:
        score -= 0.5

    # ── Momentum 252d — modificador anual (no bloqueante) ────────────────────
    # Bonus si el año es claramente positivo; penalización si el año es muy negativo.
    # No bloquea la entrada — permite capturar recuperaciones antes de que
    # el retorno anual sea positivo (evita el retraso observado en v2).
    if momentum_252d > 0.20:     # año muy positivo → bonus
        score += 1.0
    elif momentum_252d < -0.20:  # año muy negativo → penalización
        score -= 1.0

    # ── Sentimiento (auxiliar con peso reducido) ──────────────────────────────
    if sentiment > 0.3:
        score += 0.5
    elif sentiment < -0.3:
        score -= 0.5

    return score
