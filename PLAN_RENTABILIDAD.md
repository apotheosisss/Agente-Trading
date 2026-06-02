# Plan de mejora — Enfoque rentabilidad pura

Objetivo del proyecto: **inversión autónoma rentable.** No es un caso de estudio.

## Métrica única de éxito

¿Bate a **SPY buy & hold out-of-sample**, neto de costes, con **MaxDD ≤ -25%**, de forma **consistente entre folds** (≥4 de 5)?

Si una mejora no mueve esa aguja, no entra. **LLM, multi-agente (TradingAgents) y Polymarket quedan CONGELADOS (flag off)** hasta que el motor cuantitativo pase ese listón solo. No son fuentes de alfa; son coste y fragilidad.

**Baseline a batir (costes realistas):**
- SPY OOS (2022-2026): ~11% CAGR.
- Sistema actual OOS: ~7% CAGR, Sharpe 0.38, MaxDD -35%. → Pierde contra el índice.

---

## FASE A — Arreglar el riesgo (2-3 días)
Mayor retorno/esfuerzo. No genera alfa pero deja de regalar capital en drawdowns.
1. Circuit breaker que NO componga caídas (hoy resetea el pico → encadena -25% hasta -70%). Medir DD desde máximo histórico real; no re-armar hasta recuperar % de la caída.
2. Trailing stop a nivel portfolio (en vez de stop fijo por posición).
3. Sizing por volatilidad (riesgo igualado, no peso igualado).

**Gate A:** MaxDD baja de -35% a ≤ -25% sin destruir CAGR.

## FASE B — Buscar señal con edge real (2-4 semanas) — SIN LLM
1. Limpiar `score_row`: sustituir umbrales hardcodeados por ranking cross-sectional continuo normalizado por volatilidad.
2. Probar 3-4 hipótesis, una a una, en walk-forward:
   - Momentum cross-sectional (top-N fuerza relativa, rotación).
   - Trend-following con filtro de régimen (operar solo si SPY > EMA200).
   - Dual-momentum (a cash/bonos si nada supera el umbral absoluto).
   - Mean-reversion corto plazo.
3. Repensar universo (hoy 18 tech+energía, muy correlacionado). Añadir diversificadores (bonos, oro, sectores) para refugio en bears.
4. Sweep validado por **mediana de Sharpe entre folds** + filtro MaxDD + penalización por nº de parámetros.

**Gate B (decisivo):** ¿Alguna hipótesis da Sharpe OOS ≥ 0.7 estable en ≥4/5 folds Y bate a SPY OOS neto de costes?
- Sí → Fase C. No → **STOP** (ver Gate de realidad).

## FASE C — Robustez antes de dinero (1-2 semanas)
1. Purged/embargoed walk-forward, universo perturbado, Monte Carlo sobre orden de trades.
2. Costes pesimistas (slippage 10-15 bps, fills parciales, comisiones reales).
3. Análisis de régimen y correlación.

**Gate C:** el edge sobrevive a costes pesimistas y perturbaciones.

## FASE D — Ejecución real fiable (1 semana)
1. Reconciliar backtest ↔ Alpaca real (rechazos, parciales, fills a apertura).
2. Fail-safes explícitos (abortar vs continuar) + notificación.
3. Tests de invariantes (no look-ahead, caja ≥ 0, exposición ≤ capital).

## FASE E — Paper trading prueba final (30-60 días)
Medir tracking error live vs backtest. Solo tras tracking aceptable → capital real pequeño.

---

## Gate de realidad
Si tras Fase B **ninguna señal bate a SPY OOS de forma robusta** (resultado más probable):
- NO operar con dinero real.
- Capital en índice de bajo coste; seguir investigando sin arriesgar.
- Perder lento contra un índice > perder rápido sin edge.

## Qué NO hacer
- ❌ Reintroducir LLM/TradingAgents/Polymarket antes de tener señal.
- ❌ Optimizar mirando métricas full/in-sample.
- ❌ Añadir features sobre una señal base que no funciona sola.
- ❌ Pasar a real sin 30+ días de paper validado.
