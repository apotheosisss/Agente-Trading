# src/trading_agent/pipelines/llm_agents/nodes.py

import logging
from datetime import datetime, timezone

import pandas as pd

from trading_agent.pipelines.shared_utils import MAX_SCORE, score_row

logger = logging.getLogger(__name__)


# ── Helpers internos ─────────────────────────────────────────────────────────

def _reporte_tecnico_ticker(ticker: str, latest: pd.Series, date_str: str) -> str:
    close = float(latest["close"])
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    ema20 = float(latest["ema_20"])
    ema50 = float(latest["ema_50"])
    ema200 = float(latest.get("ema_200", ema50))
    bb_upper = float(latest["bb_upper"])
    bb_lower = float(latest["bb_lower"])

    senales = []

    if rsi < 25:
        senales.append(f"RSI sobreventa extrema ({rsi:.1f}) — alcista fuerte")
    elif rsi < 35:
        senales.append(f"RSI sobreventa ({rsi:.1f}) — posible rebote")
    elif rsi > 75:
        senales.append(f"RSI sobrecompra extrema ({rsi:.1f}) — bajista fuerte")
    elif rsi > 65:
        senales.append(f"RSI sobrecompra ({rsi:.1f}) — posible corrección")
    else:
        senales.append(f"RSI neutral ({rsi:.1f})")

    senales.append(
        f"MACD {'alcista' if macd > macd_sig else 'bajista'} "
        f"({macd:.4f} vs señal {macd_sig:.4f})"
    )

    if close > ema20 > ema50:
        senales.append("Tendencia alcista confirmada (precio > EMA20 > EMA50)")
    elif close < ema20 < ema50:
        senales.append("Tendencia bajista confirmada (precio < EMA20 < EMA50)")
    else:
        senales.append("Mercado en consolidación (EMAs mixtas)")

    trend_filter = "✓ Por encima EMA200" if close > ema200 else "✗ Por debajo EMA200"
    senales.append(trend_filter)

    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        if close < bb_lower:
            senales.append("Precio bajo banda Bollinger inferior")
        elif close > bb_upper:
            senales.append("Precio sobre banda Bollinger superior")
        else:
            pct = (close - bb_lower) / bb_range
            senales.append(f"Precio en banda Bollinger ({pct:.0%} del rango)")

    lines = [f"  [{ticker}] ${close:,.2f} | EMA20 ${ema20:,.2f} | EMA50 ${ema50:,.2f}"]
    for s in senales:
        lines.append(f"    • {s}")
    return "\n".join(lines)


def _reporte_sentimiento_ticker(ticker: str, grupo: pd.DataFrame) -> str:
    score_act = float(grupo["sentiment_score"].iloc[-1])
    score_7d = float(grupo["sentiment_score"].tail(7).mean())
    score_30d = float(grupo["sentiment_score"].tail(30).mean())

    def cls(s: float) -> str:
        if s > 0.4:
            return "MUY POS"
        elif s > 0.15:
            return "POS"
        elif s < -0.4:
            return "MUY NEG"
        elif s < -0.15:
            return "NEG"
        return "NEUTRAL"

    return (
        f"  [{ticker}] actual={score_act:+.3f}({cls(score_act)}) "
        f"7d={score_7d:+.3f} 30d={score_30d:+.3f}"
    )


def _reporte_riesgo_ticker(ticker: str, grupo: pd.DataFrame) -> str:
    returns = grupo["close"].pct_change().dropna()
    vol_30d = float(returns.tail(30).std() * (252**0.5))
    latest = grupo.iloc[-1]
    atr = float(latest["atr"])
    close = float(latest["close"])
    atr_pct = atr / close if close > 0 else 0.0
    peak = float(grupo["close"].tail(30).max())
    dd = (close - peak) / peak if peak > 0 else 0.0

    nivel = "EXTREMO" if vol_30d > 1.0 else "ALTO" if vol_30d > 0.6 else "MODERADO" if vol_30d > 0.3 else "BAJO"
    warn = " ⚠ DD > 15%" if abs(dd) > 0.15 else ""
    return (
        f"  [{ticker}] vol30d={vol_30d:.1%} | ATR={atr_pct:.1%} precio | "
        f"DD30d={dd:.1%} | riesgo={nivel}{warn}"
    )


# ── Nodos públicos ────────────────────────────────────────────────────────────

def agente_tecnico(feature_vector: pd.DataFrame) -> str:
    """Reporte técnico multi-ticker para el último día disponible por ticker."""
    # Cada ticker puede tener distinta última fecha (crypto cotiza fines de semana)
    last_day = (
        feature_vector.reset_index()
        .sort_values("date")
        .groupby("ticker", sort=False)
        .last()
        .reset_index()
    )
    date_str = last_day["date"].max().strftime("%Y-%m-%d")

    lines = [f"=== ANÁLISIS TÉCNICO — {date_str} ==="]
    for _, row in last_day.iterrows():
        ticker = str(row["ticker"])
        lines.append(_reporte_tecnico_ticker(ticker, row, date_str))

    report = "\n".join(lines)
    logger.info("Reporte técnico generado (%d tickers)", len(last_day))
    return report


def agente_sentimiento(feature_vector: pd.DataFrame) -> str:
    """Reporte de sentimiento multi-ticker."""
    lines = ["=== ANÁLISIS DE SENTIMIENTO ==="]
    for ticker, grupo in feature_vector.groupby("ticker"):
        lines.append(_reporte_sentimiento_ticker(str(ticker), grupo))

    report = "\n".join(lines)
    logger.info("Reporte de sentimiento generado")
    return report


def agente_riesgo(feature_vector: pd.DataFrame) -> str:
    """Reporte de riesgo multi-ticker."""
    lines = ["=== EVALUACIÓN DE RIESGO ==="]
    for ticker, grupo in feature_vector.groupby("ticker"):
        lines.append(_reporte_riesgo_ticker(str(ticker), grupo.sort_index()))

    report = "\n".join(lines)
    logger.info("Reporte de riesgo generado")
    return report


def _get_poly_boost(ticker: str, poly_signals: pd.DataFrame) -> float:
    """Obtiene el poly_score para un ticker específico.

    Combina la señal macro global (_macro) con la señal específica del ticker.
    Si no hay datos, devuelve 0.0.
    """
    if poly_signals is None or poly_signals.empty:
        return 0.0

    macro_rows = poly_signals[poly_signals["ticker"] == "_macro"]
    ticker_rows = poly_signals[poly_signals["ticker"] == ticker]

    macro_score = float(macro_rows["poly_score"].iloc[0]) if not macro_rows.empty else 0.0
    ticker_score = float(ticker_rows["poly_score"].iloc[0]) if not ticker_rows.empty else 0.0

    # Combinar: macro como base + señal específica del ticker
    # Se usa promedio ponderado: 40% macro, 60% ticker-específico
    combined = macro_score * 0.4 + ticker_score * 0.6
    return round(combined, 3)


def agente_decision(
    tech_report: str,
    sent_report: str,
    risk_report: str,
    feature_vector: pd.DataFrame,
    parameters: dict,
    universe: list,
    poly_report: str = "",
    poly_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ranking multi-activo: score + señal (BUY/HOLD/SELL) por ticker.

    Usa la misma lógica de scoring que el backtest (shared_utils.score_row).
    Si poly_signals está disponible, el score se ajusta con las probabilidades
    de Polymarket antes de determinar la señal final.
    Si OPENAI_API_KEY está disponible, enriquece el razonamiento con LLM.

    Args:
        poly_report:  Reporte texto de señales Polymarket (opcional).
        poly_signals: DataFrame con poly_score por ticker (opcional).
                      Si no se provee, funciona igual que antes.
    """
    # Tomar el ultimo dato disponible por ticker (crypto cotiza fines de semana)
    last_day = (
        feature_vector.reset_index()
        .sort_values("date")
        .groupby("ticker", sort=False)
        .last()
        .reset_index()
    )

    rows = []
    for _, row in last_day.iterrows():
        ticker = str(row["ticker"])
        base_score = score_row(row)

        if base_score == -999.0:
            signal = "HOLD"
            confidence = 0.0
            reasoning = "Excluido: precio por debajo de EMA 200 (filtro de tendencia)"
            final_score = -999.0
        else:
            # Aplicar boost de Polymarket al score cuantitativo
            poly_boost = _get_poly_boost(ticker, poly_signals)
            final_score = base_score + poly_boost

            if final_score > 2.5:
                signal = "BUY"
            elif final_score < -2.5:
                signal = "SELL"
            else:
                signal = "HOLD"

            confidence = round(min(abs(final_score) / MAX_SCORE, 0.95), 2)

            poly_note = ""
            if poly_boost != 0.0:
                direction = "↑ Poly+" if poly_boost > 0 else "↓ Poly"
                poly_note = f" | {direction}{poly_boost:+.2f}"

            reasoning = (
                f"Score técnico: {base_score:.1f}{poly_note} "
                f"→ final: {final_score:.1f}/{MAX_SCORE:.1f} → {signal}"
            )

        rows.append(
            {
                "ticker": ticker,
                "signal": signal,
                "score": round(final_score, 2) if final_score != -999.0 else -999.0,
                "score_base": round(base_score, 2) if base_score != -999.0 else -999.0,
                "poly_boost": _get_poly_boost(ticker, poly_signals) if base_score != -999.0 else 0.0,
                "confidence": confidence,
                "reasoning": reasoning,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    signal_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    # Enriquecimiento LLM opcional
    try:
        from openai import OpenAI
        from pathlib import Path
        import yaml as _yaml

        # Cargar credenciales para OpenRouter/OpenAI
        _cred_path = Path("conf/local/credentials.yml")
        _openai_cfg = {}
        if _cred_path.exists():
            with open(_cred_path, encoding="utf-8") as _f:
                _openai_cfg = (_yaml.safe_load(_f) or {}).get("openai", {})

        _api_key = _openai_cfg.get("api_key") or None
        _base_url = "https://openrouter.ai/api/v1" if (
            _api_key and _api_key.startswith("sk-or-")
        ) else None

        client = OpenAI(api_key=_api_key, base_url=_base_url) if _api_key else OpenAI()

        top_tickers = signal_df[signal_df["signal"] == "BUY"]["ticker"].tolist()[:3]
        if top_tickers:
            poly_context = f"\n{poly_report}" if poly_report else ""
            prompt = (
                f"Eres un analista de trading senior. Revisa esta información:\n\n"
                f"{tech_report}\n{sent_report}\n{risk_report}{poly_context}\n\n"
                f"El modelo cuantitativo sugiere BUY en: {', '.join(top_tickers)}.\n"
                f"En máximo 80 palabras, justifica brevemente estas selecciones."
            )
            response = client.chat.completions.create(
                model=parameters.get("model", "openai/gpt-5-nano"),
                messages=[{"role": "user", "content": prompt}],
                temperature=parameters.get("temperature", 0.1),
                max_tokens=120,
            )
            llm_reasoning = response.choices[0].message.content
            for ticker in top_tickers:
                mask = signal_df["ticker"] == ticker
                signal_df.loc[mask, "reasoning"] = llm_reasoning
            logger.info("Razonamiento LLM aplicado a %d tickers", len(top_tickers))
    except Exception as exc:
        logger.info("LLM no disponible (%s) — razonamiento basado en reglas", type(exc).__name__)

    n_buy = (signal_df["signal"] == "BUY").sum()
    has_poly = poly_signals is not None and not poly_signals.empty
    logger.info(
        "Señales generadas: %d tickers | %d BUY | %d HOLD | %d SELL | Polymarket=%s",
        len(signal_df), n_buy,
        (signal_df["signal"] == "HOLD").sum(),
        (signal_df["signal"] == "SELL").sum(),
        "activo" if has_poly else "inactivo",
    )
    return signal_df


def filtrar_signals_tradingagents(
    signal_df: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Filtra señales BUY de alto score a través de TradingAgents para confirmación.

    Flujo:
    1. Toma señales BUY con confidence >= confidence_threshold
    2. Las corre a través de TradingAgents graph (Anthropic + multi-agent)
    3. Solo mantiene señales donde AMBOS sistemas coinciden (BUY + Buy/Overweight)
    4. Retorna DataFrame con las señales confirmadas

    Args:
        signal_df: Señales generadas por agente_decision
        parameters: Parametros del proyecto (usa seccion 'llm.tradingagents')

    Returns:
        DataFrame filtrado con solo las señales confirmadas por TradingAgents
    """
    try:
        from tradingagents.graph import TradingAgentsGraph
        from pathlib import Path
        import yaml
    except ImportError as e:
        logger.warning("TradingAgents no disponible (%s) — saltando filtro", type(e).__name__)
        return signal_df

    # Extraer parámetros
    ta_cfg = parameters.get("llm", {}).get("tradingagents", {})
    confidence_threshold = float(ta_cfg.get("confidence_threshold", 0.60))
    top_n = int(ta_cfg.get("top_n_signals", 3))
    enabled = ta_cfg.get("enabled", False)

    if not enabled:
        logger.info("TradingAgents filtro deshabilitado (enabled=false)")
        return signal_df

    # Filtrar top N BUY signals de alto score
    buy_signals = signal_df[
        (signal_df["signal"] == "BUY") &
        (signal_df["confidence"] >= confidence_threshold)
    ].head(top_n).copy()

    if buy_signals.empty:
        logger.info("Sin señales BUY para confirmar con TradingAgents")
        return signal_df

    # Cargar API key de OpenAI desde credenciales
    try:
        cred_path = Path("conf/local/credentials.yml")
        if cred_path.exists():
            with open(cred_path, encoding="utf-8") as f:
                creds = yaml.safe_load(f) or {}
                openai_key = creds.get("openai", {}).get("api_key")
                if openai_key:
                    import os
                    os.environ["OPENAI_API_KEY"] = openai_key
    except Exception as exc:
        logger.warning("Error cargando credenciales OpenAI: %s", exc)
        return signal_df

    # Crear instancia de TradingAgents con config explícita (OpenRouter + gpt-5-nano)
    try:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        import os as _os
        _home = _os.path.join(_os.path.expanduser("~"), ".tradingagents")
        ta_config = {
            "project_dir": _os.path.abspath("."),
            "results_dir": _os.path.join(_home, "logs"),
            "data_cache_dir": _os.path.join(_home, "cache"),
            "memory_log_path": _os.path.join(_home, "memory", "trading_memory.md"),
            "memory_log_max_entries": None,
            "llm_provider": "openai",
            "deep_think_llm": "openai/gpt-5-nano",
            "quick_think_llm": "openai/gpt-5-nano",
            "backend_url": "https://openrouter.ai/api/v1",
            "anthropic_effort": None,
            "openai_reasoning_effort": None,
            "google_thinking_level": None,
            "checkpoint_enabled": False,
            "output_language": "English",
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_recur_limit": 100,
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "technical_indicators": "yfinance",
                "fundamental_data": "yfinance",
                "news_data": "yfinance",
            },
            "tool_vendors": {},
        }

        graph = TradingAgentsGraph(
            selected_analysts=["market", "news", "fundamentals"],
            debug=False,
            config=ta_config,
        )

        confirmed = []
        for _, row in buy_signals.iterrows():
            ticker = str(row["ticker"])
            try:
                logger.info("TradingAgents: analizando %s", ticker)
                final_state, ta_signal = graph.propagate(ticker, today)

                # ta_signal es uno de: "Buy", "Overweight", "Hold", "Underweight", "Sell"
                is_bullish = ta_signal in ["Buy", "Overweight"]

                if is_bullish:
                    row_dict = row.to_dict()
                    row_dict["tradingagents_signal"] = ta_signal
                    row_dict["confirmed"] = True
                    confirmed.append(row_dict)
                    logger.info(
                        "✓ %s confirmado por TradingAgents (%s)",
                        ticker, ta_signal,
                    )
                else:
                    logger.info(
                        "✗ %s rechazado por TradingAgents (%s vs BUY)",
                        ticker, ta_signal,
                    )

            except Exception as exc:
                logger.error("Error ejecutando TradingAgents para %s: %s", ticker, exc)
                # En caso de error, mantener la señal original (fail-safe)
                row_dict = row.to_dict()
                row_dict["tradingagents_signal"] = "ERROR"
                row_dict["confirmed"] = False
                confirmed.append(row_dict)

        # Retornar original si ninguna se confirmó, o retornar solo confirmadas
        if confirmed:
            confirmed_df = pd.DataFrame(confirmed)
            logger.info(
                "TradingAgents: %d/%d señales confirmadas",
                len(confirmed_df),
                len(buy_signals),
            )
            # Reordenar columnas para mantener compatibilidad
            cols_orig = signal_df.columns.tolist()
            for col in ["tradingagents_signal", "confirmed"]:
                if col not in cols_orig:
                    cols_orig.append(col)
            return confirmed_df[cols_orig]
        else:
            logger.info("TradingAgents: ninguna señal fue confirmada")
            return signal_df

    except Exception as exc:
        logger.error("Error inicializando TradingAgents: %s", exc)
        return signal_df
