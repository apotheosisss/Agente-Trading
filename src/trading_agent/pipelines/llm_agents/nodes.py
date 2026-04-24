# src/trading_agent/pipelines/llm_agents/nodes.py

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def agente_tecnico(feature_vector: pd.DataFrame) -> str:
    """Analiza indicadores técnicos y produce reporte cualitativo."""
    latest = feature_vector.iloc[-1]
    date_str = feature_vector.index[-1].strftime("%Y-%m-%d")

    close = float(latest["close"])
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    ema20 = float(latest["ema_20"])
    ema50 = float(latest["ema_50"])
    bb_upper = float(latest["bb_upper"])
    bb_lower = float(latest["bb_lower"])

    senales = []

    if rsi < 25:
        senales.append(f"RSI en sobreventa extrema ({rsi:.1f}) — señal alcista fuerte")
    elif rsi < 35:
        senales.append(f"RSI en sobreventa ({rsi:.1f}) — posible rebote")
    elif rsi > 75:
        senales.append(f"RSI en sobrecompra extrema ({rsi:.1f}) — señal bajista fuerte")
    elif rsi > 65:
        senales.append(f"RSI en sobrecompra ({rsi:.1f}) — posible corrección")
    else:
        senales.append(f"RSI neutral ({rsi:.1f})")

    if macd > macd_sig:
        senales.append(f"MACD alcista ({macd:.2f} > señal {macd_sig:.2f})")
    else:
        senales.append(f"MACD bajista ({macd:.2f} < señal {macd_sig:.2f})")

    if close > ema20 > ema50:
        senales.append("Tendencia alcista confirmada (precio > EMA20 > EMA50)")
    elif close < ema20 < ema50:
        senales.append("Tendencia bajista confirmada (precio < EMA20 < EMA50)")
    else:
        senales.append("Mercado en consolidación (señales mixtas de EMA)")

    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pct = (close - bb_lower) / bb_range
        if close < bb_lower:
            senales.append("Precio bajo banda de Bollinger inferior — posible rebote")
        elif close > bb_upper:
            senales.append("Precio sobre banda de Bollinger superior — posible corrección")
        else:
            senales.append(f"Precio en banda de Bollinger ({bb_pct:.0%} del rango)")

    report = f"=== ANÁLISIS TÉCNICO — {date_str} ===\n"
    report += f"Precio: ${close:,.2f} | EMA20: ${ema20:,.2f} | EMA50: ${ema50:,.2f}\n\n"
    report += "Señales:\n"
    for s in senales:
        report += f"  • {s}\n"

    logger.info("Reporte técnico generado")
    return report


def agente_sentimiento(feature_vector: pd.DataFrame) -> str:
    """Interpreta el score de sentimiento de mercado."""
    score_actual = float(feature_vector["sentiment_score"].iloc[-1])
    score_7d = float(feature_vector["sentiment_score"].tail(7).mean())
    score_30d = float(feature_vector["sentiment_score"].tail(30).mean())

    def clasificar(s: float) -> str:
        if s > 0.4:
            return "MUY POSITIVO"
        elif s > 0.15:
            return "POSITIVO"
        elif s < -0.4:
            return "MUY NEGATIVO"
        elif s < -0.15:
            return "NEGATIVO"
        return "NEUTRAL"

    tendencia = ""
    if score_actual > score_30d + 0.2:
        tendencia = "→ Sentimiento mejorando significativamente vs. promedio mensual\n"
    elif score_actual < score_30d - 0.2:
        tendencia = "→ Sentimiento deteriorándose vs. promedio mensual\n"

    report = "=== ANÁLISIS DE SENTIMIENTO ===\n"
    report += f"Score actual:    {score_actual:+.3f}  ({clasificar(score_actual)})\n"
    report += f"Promedio 7d:     {score_7d:+.3f}  ({clasificar(score_7d)})\n"
    report += f"Promedio 30d:    {score_30d:+.3f}  ({clasificar(score_30d)})\n"
    if tendencia:
        report += f"\n{tendencia}"

    logger.info("Reporte de sentimiento generado")
    return report


def agente_riesgo(feature_vector: pd.DataFrame) -> str:
    """Evalúa volatilidad y condiciones de riesgo del mercado."""
    returns = feature_vector["close"].pct_change().dropna()
    vol_7d = float(returns.tail(7).std() * (252 ** 0.5))
    vol_30d = float(returns.tail(30).std() * (252 ** 0.5))

    latest = feature_vector.iloc[-1]
    atr = float(latest["atr"])
    close = float(latest["close"])
    atr_pct = atr / close if close > 0 else 0.0

    peak_30d = float(feature_vector["close"].tail(30).max())
    drawdown_actual = (close - peak_30d) / peak_30d if peak_30d > 0 else 0.0

    if vol_30d > 1.0:
        nivel = "EXTREMO"
    elif vol_30d > 0.6:
        nivel = "ALTO"
    elif vol_30d > 0.3:
        nivel = "MODERADO"
    else:
        nivel = "BAJO"

    report = "=== EVALUACIÓN DE RIESGO ===\n"
    report += f"Volatilidad 7d  (anualizada): {vol_7d:.1%}\n"
    report += f"Volatilidad 30d (anualizada): {vol_30d:.1%}\n"
    report += f"ATR actual: ${atr:,.2f} ({atr_pct:.1%} del precio)\n"
    report += f"Drawdown vs max 30d: {drawdown_actual:.1%}\n"
    report += f"\nNivel de riesgo: {nivel}\n"

    if abs(drawdown_actual) > 0.15:
        report += "ADVERTENCIA: Drawdown significativo — considerar reducir posicion\n"

    logger.info(f"Reporte de riesgo generado: nivel {nivel}")
    return report


def agente_decision(
    tech_report: str,
    sent_report: str,
    risk_report: str,
    feature_vector: pd.DataFrame,
    parameters: dict,
    ticker: str,
) -> pd.DataFrame:
    """Sintetiza los reportes especializados y genera la señal de trading final.

    Sistema de scoring cuantitativo sobre indicadores del último período.
    Si OPENAI_API_KEY está disponible, enriquece el razonamiento con LLM.
    """
    latest = feature_vector.iloc[-1]

    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    close = float(latest["close"])
    ema20 = float(latest["ema_20"])
    ema50 = float(latest["ema_50"])
    bb_upper = float(latest["bb_upper"])
    bb_lower = float(latest["bb_lower"])
    sentiment = float(latest["sentiment_score"])

    score = 0.0

    if rsi < 25:
        score += 3.0
    elif rsi < 35:
        score += 1.5
    elif rsi > 75:
        score -= 3.0
    elif rsi > 65:
        score -= 1.5

    score += 1.0 if macd > macd_sig else -1.0

    if close > ema20 and ema20 > ema50:
        score += 2.0
    elif close < ema20 and ema20 < ema50:
        score -= 2.0

    if close < bb_lower:
        score += 1.5
    elif close > bb_upper:
        score -= 1.5

    if sentiment > 0.3:
        score += 1.0
    elif sentiment < -0.3:
        score -= 1.0

    max_score = 8.5  # RSI(3) + MACD(1) + EMA(2) + BB(1.5) + sentiment(1)
    if score > 2.5:
        signal = "BUY"
    elif score < -2.5:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(min(abs(score) / max_score, 0.95), 2)

    reasoning = (
        f"{tech_report}\n{sent_report}\n{risk_report}\n"
        f"=== DECISION FINAL ===\n"
        f"Score: {score:.1f}/{max_score:.1f} -> {signal} (confianza {confidence:.2f})"
    )

    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            f"Eres un analista de trading senior. Analiza la siguiente informacion:\n\n"
            f"{tech_report}\n{sent_report}\n{risk_report}\n\n"
            f"El modelo cuantitativo sugiere {signal} con confianza {confidence:.2f}.\n"
            f"Proporciona en maximo 100 palabras un razonamiento claro para esta decision."
        )
        response = client.chat.completions.create(
            model=parameters.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=parameters.get("temperature", 0.1),
            max_tokens=150,
        )
        reasoning = response.choices[0].message.content
        logger.info("Razonamiento enriquecido con LLM")
    except Exception as exc:
        logger.info(f"LLM no disponible ({type(exc).__name__}) — razonamiento basado en reglas")

    signal_df = pd.DataFrame(
        [
            {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "score": score,
            }
        ]
    )

    logger.info(f"Senal generada: {signal} | confianza={confidence:.2f} | score={score:.1f}")
    return signal_df
