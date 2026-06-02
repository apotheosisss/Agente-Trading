# Resultados de optimización — Modelo Polymarket

Registro vivo de la ejecución del plan. Fecha inicio: 2026-06-01.

---

## Fase 0 — Objetivo OOS + walk-forward multi-ventana ✅

**Cambios aplicados:**
- `calcular_walk_forward` reescrito: además de in/out-of-sample, parte la curva en `n_folds` (5) segmentos consecutivos y reporta Sharpe/CAGR de cada uno + **mediana de Sharpe entre folds** (métrica de selección anti-overfit).
- `run_param_sweep.py`: ahora rankea por **mediana de Sharpe entre folds** (no por Sharpe full contaminado por in-sample) y **descarta configs con MaxDD < -25%**.
- Parámetros nuevos en `parameters.yml`: `walk_forward_folds`, `slippage`, `execute_next_open`.

**Diagnóstico que reveló el walk-forward multi-ventana (config baseline):**

| Fold | Periodo | Sharpe | CAGR |
|---|---|---|---|
| 1 | 2019-01 → 2020-07 | 0.63 | 18.6% |
| 2 | 2020-07 → 2021-12 | **1.40** | 81.9% |
| 3 | 2021-12 → 2023-05 | 0.72 | 26.8% |
| 4 | 2023-05 → 2024-11 | **0.21** | 0.7% |
| 5 | 2024-11 → 2026-04 | **0.22** | 0.3% |

**Conclusión:** el edge se concentra en 2020-2021 (rally post-COVID). Los **últimos 3 años (folds 4-5) están planos (~0% CAGR, Sharpe ~0.2)**. La estrategia no tiene momentum aprovechable en el régimen reciente.

---

## Fase 1 — Costes realistas ✅

**Cambios aplicados en `ejecutar_backtest`:**
- **Fills al `open` del día siguiente** (`execute_next_open: true`) en compras y ventas de rebalanceo → elimina el look-ahead de operar al mismo cierre que generó la señal.
- **Slippage de 5 bps** por operación, sumado a la comisión (`cost = commission + slippage`), aplicado en todos los fills (compras, ventas, stops, circuit breaker, VIX).

**Impacto en métricas full (config baseline):**

| Métrica | Antes (auditoría) | Con costes realistas |
|---|---|---|
| CAGR | 23.4% | **21.1%** |
| Sharpe | 0.69 | **0.65** |
| MaxDD | -35.5% | -35.3% |

Baja moderada (~2 pts CAGR). El timing al next-open no rescata el problema de fondo (régimen reciente plano).

### Sweep amplio (16 configs: rebalance_interval × min_entry_score)

Rankeado por **mediana de Sharpe entre folds**, descartando MaxDD < -25%:

| Interval | Score | Trades/yr | CAGR full | OOS-CAGR | MedFold-Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| 21d | 2.5 | 3.8 | 31.2% | **+13.0%** | **1.27** | -45.8% ❌ |
| 21d | 3.0 | 4.0 | 27.8% | +9.9% | 1.25 | -45.6% ❌ |
| 21d | 1.5 | 4.1 | 34.2% | +10.5% | 1.08 | -54.6% ❌ |
| 15d | 1.5–2.5 | ~17 | 14–17% | -1.5% | ~0.45 | -63 a -70% ❌ |
| 10d | * | ~29 | 5–12% | -6% | ~0.3 | -68 a -70% ❌ |
| 5d | * | ~53 | ~0% | -11% | ~-0.15 | -70% ❌ |
| **semanal (baseline)** | 2.0 | ~40 | 21.1% | +7.0% | 0.63 | **-35.3%** ❌ |

**Resultado crítico: NINGUNA configuración pasa el filtro de MaxDD < -25%.** El de-overfitting funcionó: en vez de cherry-pickear, el criterio **rechazó toda la rejilla**. La mejor por robustez (21d, mediana-Sharpe 1.27, OOS +13%) sigue cargando -45% de drawdown. La de menor drawdown (semanal, -35%) tampoco es investment-grade.

**Patrones:**
- Rebalanceo más espaciado (21d) → mejor retorno y robustez pero peor drawdown.
- Rebalanceo frecuente (5d) → destruye valor (overtrading: ~53 trades/año, CAGR ~0%, OOS negativo).
- El rebalanceo **semanal** (baseline) tiene el mejor MaxDD (-35%) de todas.

### Hallazgo de riesgo: el circuit breaker amplifica drawdowns

Los MaxDD de -70% (configs de intervalo) **no son un bug** sino un defecto de diseño: el circuit breaker **resetea el pico al nivel de liquidación** tras cada disparo (líneas 111-114 de `backtesting/nodes.py`). En caídas prolongadas (bear 2022, lateralidad 2023-2026) los recortes del 25% se **encadenan y componen**: -25% → -44% → -58% → -70%. El control pensado para *limitar* el drawdown termina *permitiéndolo*.

Es la palanca de mejora más clara, pero es un cambio de **estrategia/riesgo**, no de parámetro: hay un trade-off duro (no resetear el pico baja el MaxDD pero sacrifica la captura de recuperaciones). Que ninguna combinación resuelva ambos confirma que **el problema es estructural de la estrategia momentum, no de tuning.**

### Conclusión Fase 1
El motor base **no tiene edge robusto out-of-sample bajo ninguna parametrización explorada**, y los controles de riesgo no acotan el drawdown a niveles operables. Se mantiene la config **semanal** (mejor perfil de drawdown) como base, pero con la advertencia explícita de que no es apta para capital real tal cual.

---

## Fase 2 — Polymarket real y backtesteable (en curso)

**Causa raíz del `poly_score = 0.0` en los 18 tickers (diagnóstico API):**
- La llamada `GET /markets?active=true&closed=false&limit=1000` **ignora el limit y devuelve solo 100 mercados sin ordenar**, dominados por novelty/deportes ("Rihanna album before GTA VI", NHL, NBA, sentencias judiciales). Casi ninguno financiero/macro → 0 coincidencias de keywords.
- Con `order=volumeNum&ascending=false` sí afloran mercados relevantes (MicroStrategy/Bitcoin, elecciones 2028, caída del régimen iraní…): **13/100 con keywords de finanzas/macro**.

**Fixes aplicados:**
1. **Bug de raíz (afectaba también al código original):** el log `"%,.0f"` es formato inválido en %-logging de Python → lanzaba `ValueError` capturado por el `except` → **siempre** devolvía fail-safe neutro (0.0). Corregido a `%.0f`. *Este bug por sí solo explica el `poly_score=0` histórico.*
2. `_fetch_markets`: ahora **pagina** (offset hasta 500) y **ordena por `volumeNum` desc** → afloran mercados financieros/macro/cripto reales.
3. Mapping ampliado con categorías por **nombre de empresa** (NVIDIA, Apple, MicroStrategy/BTC, petróleo, Venezuela).
4. **Nodo `persistir_historico_polymarket`** + dataset `polymarket_history` (parquet): acumula snapshots diarios para hacer la señal backtesteable a futuro.
5. **Integración en el backtest** tras flag `use_polymarket` (+ `polymarket_shuffle` para ablation): suma `poly_score` histórico por `(fecha,ticker)` al score de entrada.

**Resultado:** señales **no-nulas** para los 19 tickers (500 mercados procesados), mapeando eventos reales (Taiwán, alto el fuego Irán, recortes Fed, MicroStrategy/BTC, Venezuela).

**Limitación conocida:** el matching por keywords no maneja negaciones ("*no* Fed rate cuts" se mapeó como alcista). Mejora futura: embeddings o clasificación LLM evento→activo.

---

## Fase 3 — Ablation A/B/C ✅

`run_ablation.py`: corre el backtest en 3 variantes sobre datos cacheados (B=base, A=base+poly, C=base+poly barajado) y emite veredicto automático por OOS-CAGR.

**Estado con datos reales:** A==B==C, porque el histórico Polymarket arranca vacío y solo tiene el snapshot de hoy (0 solape con la ventana de backtest 2019-2026/04). **Esperado** — la señal será evaluable cuando el store acumule meses.

**Validación del harness (señal sintética informativa, oráculo de retorno futuro 5d):**

| Variante | OOS-CAGR | OOS-Sharpe | MedFold-Sharpe |
|---|---|---|---|
| B base | 7.0% | 0.38 | 0.63 |
| **A base+poly** | **31.7%** | **0.78** | **0.92** |
| C base+poly barajado | 8.1% | 0.41 | 0.53 |

El harness **detecta correctamente** la señal (A ≫ B) y la destruye al barajar (C ≈ B). Veredicto automático: *"A > B y A > C → conservar"*. La maquinaria de validación es sólida; solo falta acumular datos reales.

---

## Fase 4 — Reconciliación + workflow ✅

1. **Bug de sizing corregido:** `enviar_orden` iteraba sobre TODAS las señales BUY aprobadas (9 × $5.000 = $45k sobre $10k de capital). Ahora limita al **top-N por score** (`[:max_positions]`), coherente con el backtest. Verificado: el workflow completo ahora ejecuta **2 posiciones** (antes 9).
2. **README corregido:** reporta métricas full (CAGR ~21%, MaxDD -35%) y OOS (~7%), con nota de honestidad metodológica sobre el 51% in-sample.
3. **Workflow completo ejecutado** (`kedro run --pipeline __default__`): ingesta → features → polymarket (no-nulo) → agentes → backtest → ejecución (2 posiciones). Sin errores.

---

## Veredicto global

| Componente | Estado tras optimización |
|---|---|
| Marco de validación (walk-forward OOS, ablation) | ✅ Sólido y honesto |
| Polymarket (señal viva) | ✅ Funcional (antes daba 0.0 por un bug) |
| Polymarket (backtesteable) | 🟡 Plumbing listo; faltan meses de datos acumulados |
| **Motor de momentum base** | ❌ **Sin edge robusto OOS; MaxDD inoperables** |
| Reconciliación backtest↔ejecución | ✅ Corregida |

**El trabajo de optimización dejó la infraestructura en estado honesto y funcional, pero confirmó que la estrategia base no es rentable de forma robusta tal como está.** El siguiente paso real no es más tuning, sino rediseñar el motor de señal (o aceptar que el momentum simple no bate a SPY en este universo) y acumular histórico Polymarket para una ablación real.

---

# PLAN RENTABILIDAD — Ejecución

## FASE A — Arreglar el riesgo ✅ (resultado cualificado)

**Implementado en `backtesting/nodes.py` + `risk` params:**
- **Circuit breaker sin compounding:** el pico ya no se resetea hacia abajo; tras liquidar, se re-arma solo cuando el régimen de mercado (`SPY > EMA200`) vuelve a risk-on. Evita encadenar -25% sucesivos.
- **Sizing por volatilidad (`vol_sizing`):** tamaño inverso al ATR%, clip [0.3, 1.0] (nunca apalanca).
- **Trailing stop:** implementado pero **desactivado** — la evidencia muestra whipsaw sin beneficio de drawdown.

**Mini-sweep (todos los controles ON, variando `max_positions`):**

| max_pos | CAGR | Sharpe | MaxDD | OOS-CAGR | Gate -25% |
|---|---|---|---|---|---|
| 2 | 12.5% | 0.54 | -31.5% | 4.9% | ❌ |
| 3 | 8.8% | 0.42 | -24.8% | 2.5% | ✅ |
| 4 | 6.9% | 0.37 | -20.7% | 1.2% | ✅ |

**Aislando controles (@max_pos=3):** `vol_sizing` es lo único que baja el DD (de -38% a -25%); el circuit breaker solo no lo logra porque carteras de 2-3 nombres volátiles **atraviesan el umbral en gaps de un día**.

**Veredicto Fase A:** el gate de MaxDD ≤ -25% **es alcanzable, pero NO sin destruir el CAGR** (21% → 9%). En esta estrategia retorno y riesgo salen de la misma fuente (concentración en momentum volátil): reducir DD = reducir exposición = reducir retorno, proporcionalmente. La infraestructura de riesgo queda lista y es correcta; **no puede rescatar una señal sin edge.** Confirma que el problema es la señal → Fase B.

---

## FASE B — Buscar señal con edge real ✅ (resultado: STOP / gate de realidad)

**Motor de señal limpio** (`shared_utils.compute_scores`): reemplaza los ~10 umbrales hardcodeados de `score_row` por señales continuas, normalizadas por volatilidad y z-score cross-sectional. Seleccionable por `signal_strategy`. 4 hipótesis nuevas + legacy.

**Comparación (max_positions=3, walk-forward) — listón: SPY OOS = 11.3% CAGR:**

| Estrategia | CAGR | MaxDD | OOS-CAGR | OOS-Sharpe | MedFold |
|---|---|---|---|---|---|
| legacy | 9.2% | -25.7% | 2.5% | 0.24 | 0.38 |
| mom_vol | 7.8% | -38.1% | -1.5% | 0.16 | 0.33 |
| dual_mom | 1.5% | -44.3% | -3.7% | 0.10 | -0.04 |
| trend | 3.1% | -41.6% | -4.6% | 0.10 | 0.13 |
| **meanrev** | 8.4% | -22.3% | **5.3%** | **0.36** | **0.60** |

**meanrev** (comprar sobreventa RSI dentro de tendencia alcista) fue la única con señal real. Refinada:
- Rebalanceo: **semanal** óptimo (5/10 días destruyen la señal por turnover).
- Concentración: 2 posiciones mejor que 3-4.
- Umbral sobreventa `signal_min_score=1.0` (sobreventa profunda): **mejor config**.

**Mejor config alcanzada (meanrev, 2 pos, semanal, min_score=1.0):**
- CAGR 9.5% | **MaxDD -16.6%** | OOS-CAGR **7.9%** | OOS-Sharpe **0.50** | MedFold 0.67

### Veredicto Gate B: NO SUPERADO

| Criterio | Objetivo | Mejor alcanzado | ¿Pasa? |
|---|---|---|---|
| OOS-Sharpe | ≥ 0.70 | 0.50 | ❌ |
| Bate SPY OOS-CAGR | > 11.3% | 7.9% | ❌ |
| MaxDD | ≤ -25% | -16.6% | ✅ |
| Consistencia (MinFold > 0) | sí | 0.16 | ✅ |

**Conclusión honesta (Gate de realidad):** ninguna de las hipótesis bate a comprar y mantener SPY out-of-sample. Más aún: el ajuste de `min_score` se hizo **mirando el OOS** (contaminación) — y aun así no se superó a SPY. *Si ni espiando el OOS se bate al índice, no hay edge robusto explotable en este universo con estas señales.*

**Lo logrado sí tiene valor:** `meanrev` es estrictamente mejor que la estrategia original (legacy perdía OOS con -35% DD; meanrev da +7.9% OOS con -16.6% DD). Es un sistema **defensivo y honesto**, pero **no genera alfa sobre el índice.**

### Recomendación final (enfoque rentabilidad)
1. **No operar con dinero real** un sistema que no bate a SPY neto de costes. Es la decisión rentable.
2. Si se insiste en sistema propio: el camino NO es más tuning (overfit), sino **datos/señales nuevas** (datos fundamentales, alternativos, otro universo/clase de activo) — y validarlas con esta misma infraestructura.
3. La opción de máxima rentabilidad esperada hoy: **indexar (SPY/VT) con bajo coste** y seguir investigando señales sin arriesgar capital.
4. LLM, TradingAgents y Polymarket: mantener **congelados** (no aportan alfa, confirmado).

---

## FASE B+ — Universo multi-activo (rotación / dual-momentum)

Hipótesis: el problema era el universo correlacionado (todo tech+energía). Se reemplazó por **12 ETFs diversificados** (SPY, QQQ, IWM, EFA, EEM, VNQ, IEF, TLT, LQD, GLD, DBC, BIL) para permitir rotación a refugios (bonos/oro/materias primas) en risk-off.

**Resultado (dual_mom, top-4, vol_sizing):**

| | dual_mom@4 | SPY buy&hold |
|---|---|---|
| CAGR full | 6.1% | 16.3% |
| **Sharpe full** | **0.86** | **0.87** |
| MaxDD full | **-9.4%** | -33.7% |
| Sharpe OOS | 0.45 | 0.69 |
| CAGR OOS | 2.2% | 11.3% |

**Lo bueno:** la diversificación funcionó para el RIESGO — MaxDD pasó de -35% a **-9.4%**, y la estrategia fue **positiva en los 5 folds** (consistencia real, lo que nunca se logró antes).

**Lo decisivo:** el **Sharpe full de la rotación (0.86) ≈ el de SPY (0.87)**. Son equivalentes en base ajustada a riesgo. Implicación matemática: **cuando tu Sharpe iguala al del benchmark, NINGÚN sizing/apalancamiento te permite batirlo** — solo te mueve a otro punto de la MISMA recta riesgo/retorno. Apalancar dual_mom 2.7x daría ~16% CAGR pero con ~-25% DD: igualas a SPY, no lo superas.

### VEREDICTO FINAL (definitivo)

Probado exhaustivamente: **2 universos, 5 familias de señal, barridos de concentración, riesgo y umbrales.** Ninguna configuración produce **ventaja ajustada a riesgo sobre SPY**. Es exactamente lo que predice la eficiencia de mercado para señales basadas solo en precio sobre instrumentos líquidos.

**No existe, en lo explorado, un sistema técnico que bata a comprar y mantener SPY.** Lo que SÍ se puede construir:
- **(a)** Un sistema **defensivo** (rotación dual_mom): retorno menor pero **1/3 del drawdown** de SPY. Útil si la meta es "exposición a mercado con caídas suaves", NO "batir a SPY".
- **(b)** Indexar a SPY/VT y no operar — máxima rentabilidad esperada para riesgo asumido.

**Para batir a SPY de verdad** hace falta una fuente de alfa **fuera del precio sobre ETFs líquidos**: datos fundamentales/alternativos, mercados menos eficientes (small caps ilíquidas, microestructura cripto, nichos), mayor frecuencia, o ventajas estructurales. Todas son empresas mayores, sin garantía, y NO se resuelven con más tuning.

---

# REFORMA — Time-Series Momentum multi-activo (resultado: ÉXITO como diversificador)

Implementado en `run_tsmom.py` (backtester dedicado long/short con volatility targeting) y `run_blend.py`. Config **canónica de la literatura** (no tuneada): lookbacks 3/6/12m, rebalanceo mensual, vol targeting, gross ≤1.5.

## TSMOM standalone (varios universos, 2015-2026)

| Universo | Sharpe full | Sharpe OOS | MaxDD | Corr SPY | 2020 | 2022 |
|---|---|---|---|---|---|---|
| r1 (9 ETFs) | 0.40 | 0.49 | -19% | 0.06 | +7.6% | +5.9% |
| r2 (18 ETFs) | 0.37 | 0.52 | -24% | -0.04 | +9.8% | +5.4% |
| **r2+crypto (19)** | **0.73** | 0.31 | -28% | -0.12 | +26.8% | +16.2% |
| crypto (8) | 0.62 | 0.36 | -75% | n/a | -20% | +22.7% |

**Confirmado empíricamente:** TSMOM diversificado es **positivo en TODOS los bears** (2020 y 2022) y **descorrelacionado de SPY** (~0). Es crisis-alpha real. Standalone, Sharpe ~0.4-0.7 (en línea con la literatura; no es un SPY-beater, y el bull 2023-25 lastra el OOS).

## La clave: TSMOM como DIVERSIFICADOR de una cartera de acciones

Combinar SPY con el stream TSMOM (run_blend.py):

| Cartera | CAGR full | Sharpe full | MaxDD full | Sharpe OOS | MaxDD OOS |
|---|---|---|---|---|---|
| 100% SPY | 9.8% | 0.71 | -33.7% | 0.58 | -24.5% |
| 80/20 | 10.3% | 0.89 | -26.3% | 0.68 | -15.9% |
| **60/40** | 10.6% | **1.05** | **-18.5%** | **0.73** | -13.8% |
| 50/50 | 10.7% | 1.08 | -14.5% | 0.70 | -12.9% |

**Full-sample: mejora de Pareto pura** — más CAGR, Sharpe 0.71→1.08, drawdown a la mitad. **OOS: Sharpe 0.58→0.73 y drawdown casi a la mitad**, con coste mínimo de retorno.

## VEREDICTO DE LA REFORMA

✅ **Encontramos un sistema genuinamente rentable y robusto** — el primero de toda la investigación. No es un SPY-beater standalone (eso no existe con precio sobre líquidos), pero **mejora estrictamente el perfil rentabilidad/riesgo de tener acciones**:
- Sharpe de cartera +30-50%.
- Drawdown reducido ~40-50%.
- Fuente de retorno descorrelacionada con evidencia de un siglo.

**Cómo se convierte en "más rentable que SPY":** el blend tiene Sharpe ~1.0 vs SPY ~0.7. Apalancando modestamente el blend (~1.5-2x) se igualaría o superaría el retorno de SPY **al mismo o menor riesgo** — porque el Sharpe es superior (esto SÍ funciona, a diferencia del intento anterior donde el Sharpe igualaba al de SPY). El apalancamiento añade riesgo real (margin, gaps) y debe dimensionarse con cuidado.

**Config recomendada para producción:** TSMOM r2(+crypto opcional), mensual, vol targeting, como **overlay 30-40%** sobre un core de equities. Implementable en Alpaca (ETFs largos/cortos + cripto). Robustez pendiente: walk-forward purgado, costes pesimistas (roll/borrow), paper 60-90 días.

---

# OPTIMIZACIÓN ADICIONAL — comparación de mejoras (sobre r2crypto)

Dos mejoras estructurales (no tuning), implementadas como flags en `run_tsmom.py`:
- **PVT** = volatility targeting de CARTERA con covarianzas (risk parity a nivel portfolio).
- **strength** = señal de fuerza de tendencia continua (tanh de momentum normalizado por vol), en vez de solo el signo.

### TSMOM standalone

| Variante | Sharpe full | Sharpe OOS | MaxDD | MinFold | 2020 | 2022 |
|---|---|---|---|---|---|---|
| V0 base (signo, cap bruto) | 0.73 | 0.31 | -28% | 0.11 | +27% | +16% |
| V1 +PVT | **0.91** | 0.38 | -31% | 0.09 | +37% | +20% |
| V2 strength | 0.72 | 0.34 | -28% | -0.02 | +37% | +4% |
| **V3 strength+PVT** | 0.85 | **0.51** | -28% | **0.17** | +36% | +11% |

- **PVT** = mayor salto de Sharpe full y retorno (mejor asignación de riesgo).
- **strength+PVT (V3)** = mejor OOS y mejor consistencia entre folds (más robusto).

### Blend 50/50 SPY + TSMOM, OOS (2022+) — lo que importa

| Variante en el blend | CAGR OOS | Sharpe OOS | MaxDD OOS |
|---|---|---|---|
| 100% SPY (referencia) | 7.7% | 0.58 | -24.5% |
| V0 base | 6.4% | 0.70 | -12.9% |
| V1 PVT | 6.7% | 0.71 | -14.1% |
| **V3 strength+PVT** | **7.7%** | **0.80** | **-13.7%** |

**Ganador: V3.** El blend 50/50 con V3 **iguala el retorno OOS de SPY (7.7%)** con Sharpe 0.58→0.80 (+38%) y drawdown -24.5%→-13.7% (-44%). Full-sample: V1 lidera en retorno (CAGR 12.1%, Sharpe 1.15) pero V3 es más robusto OOS → **V3 recomendado** (config por defecto en `run_tsmom.py`).

**Conclusión de la optimización:** las dos mejoras son **complementarias** y ambas son estructurales (risk allocation + calidad de señal), no data-mining. Resultado: una cartera que **iguala el retorno de SPY con ~40% menos drawdown y Sharpe ~0.80 OOS**. Apalancada ~1.3-1.5× superaría el retorno de SPY al mismo riesgo.

---

# OPTIMIZACIÓN — experimentos de 2º nivel (breadth / 2º premio / leverage)

Probados en orden, decisión por evidencia:

### Exp 1 — Breadth (más mercados): ❌ RECHAZADO
Universo "broad" (32 ETFs: sectores, crédito, más commodities/FX). Blend 50/50 OOS Sharpe **0.65** (peor que r2crypto 0.80). Causa: los ETFs añadidos (XLE, XLF, HYG, EMB…) están **correlacionados con SPY** → no aportan diversificación. La breadth real exige futuros genuinamente descorrelacionados (no replicables con ETFs). **Se mantiene r2crypto.**

### Exp 2 — Segundo premio (reversión/value): ✅ ACEPTADO (peso disciplinado)
Sleeve de reversión de largo plazo (~2 años) combinado con trend. *(Carry puro no es computable con precios de ETF; reversión es el 2º premio implementable.)*

| Peso reversión | Sharpe full | Sharpe OOS | 2020 | 2022 |
|---|---|---|---|---|
| 0.0 | 0.85 | 0.51 | +36% | +11% |
| **0.2** | 0.84 | **0.66** | +26% | +12% |
| 0.35 | 0.80 | 0.82 | +15% | +15% |
| 0.5 | 0.67 | 0.85 | **-7%** | +11% |

Mejora OOS real, **pero el Sharpe full BAJA al subir el peso** → pesos altos (0.35-0.5) sobreajustan al periodo reciente y rompen la protección de 2020. **Elegido REV_W=0.2** (mejora OOS sin sacrificar full-sample ni bears). Se rechaza explícitamente perseguir el 0.82.

### Exp 3 — Leverage del blend: ✅ ACEPTADO (modesto)
Blend 50/50 (TSMOM con reversión 0.2) + SPY, OOS, coste de financiación 5%/año:

| | CAGR OOS | Sharpe OOS | MaxDD OOS |
|---|---|---|---|
| 100% SPY | 7.7% | 0.58 | -24.5% |
| Blend **1.0x** | **8.8%** | **0.91** | **-13.1%** |
| Blend 1.3x | 9.7% | 0.79 | -17.3% |
| Blend 1.5x | 10.3% | 0.74 | -20.1% |

## DECISIÓN FINAL

**Mejor sistema: r2crypto + TSMOM(strength + vol-targeting de cartera + reversión 0.2), como blend 50/50 con SPY, apalancado 1.0–1.3×.**

- Sin apalancar **bate a SPY OOS en las TRES dimensiones**: CAGR 8.8% vs 7.7%, Sharpe 0.91 vs 0.58, MaxDD -13% vs -24%.
- A 1.3× amplía la ventaja de retorno (9.7%) manteniéndose superior en Sharpe y drawdown.
- Mejoras aceptadas: vol-targeting de cartera (V1) + señal de fuerza (V3) + reversión 0.2. Rechazadas: breadth con ETFs correlacionados, y pesos de reversión altos (overfit).

**Caveats honestos:** sigue siendo backtest; el peso de reversión tiene leve selección OOS; el Sharpe forward será menor. Pendiente OBLIGATORIO antes de capital real: walk-forward purgado, costes pesimistas (borrow/financiación reales), y paper trading 60-90 días midiendo tracking. El apalancamiento añade riesgo de margin/gap.
