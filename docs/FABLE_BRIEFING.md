# Briefing para Fable — Agente-Trading

> Documento vivo. Pégalo completo al inicio de una sesión nueva de Claude para
> que actúe como "cerebro" del proyecto sin tener que redescubrir el historial
> (lo que costó varias horas de investigación la última vez que no existía).
> Actualízalo cuando cambie algo estructural: nueva estrategia, cambio de
> apalancamiento, hallazgo de auditoría, pivote de universo.

Última actualización: 2026-08-27.

---

## 1. Qué es este proyecto

Investigación personal de trading algorítmico (Claudio, Ingeniería Informática
mención Ciencia de Datos, DuocUC). **No es un caso de estudio — el objetivo es
un sistema de inversión autónoma genuinamente rentable**, evaluado en paper
trading (Alpaca) antes de considerar capital real. Kedro + Python + UV,
desplegado via GitHub Actions.

Dos cuentas Alpaca paper corriendo en paralelo:

| Cuenta | Secrets | Cron |
|---|---|---|
| **Polymarket** | `ALPACA_API_KEY_POLY` / `ALPACA_SECRET_KEY_POLY` | Lun-Vie 21:20 UTC |
| **Crypto** | `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Lun-Vie 13:15 UTC |

Ambas corren **la misma estrategia y el mismo código** (rama `feature/polymarket`).
El nombre "Crypto" es historia — ya no es un modelo cripto, ver §3.

Rama por defecto en GitHub: **`develop`** (ahí vive el CI/CD real). `main` se
mantiene sincronizada por higiene pero no ejecuta nada. El código de estrategia
vive en `feature/polymarket`. La antigua rama `feature/crypto` (diseño anterior,
ver §2) fue **archivada el 2026-08-27** como tag `archive/feature-crypto` y
borrada del remoto; recuperable con
`git checkout -b feature/crypto archive/feature-crypto`. Las ramas de trabajo
ya mergeadas (feature/tsmom-strategy, feature/alpaca-tsmom-exec, chore/*)
también se podaron — el repo queda en develop + feature/polymarket + main.

---

## 2. Historia — por qué se abandonó el enfoque anterior (importante, no repetir)

**Diseño original (hasta ~mayo 2026):** señales generadas por agentes LLM
(técnico/sentimiento/riesgo/decisión) + confirmación multi-agente
(TradingAgents vía OpenRouter) + sentimiento de Polymarket como boost de
score. Sesgado además por un evento puntual: un brote de Hantavirus en un
crucero (13 casos, 3 muertes, contenido y declarado terminado por la OMS el
2026-07-02 — **nunca fue una amenaza sistémica**), que llevó a diseñar
controles de "crisis pandémica" (VIX>50 como gatillo de liquidación) mal
calibrados para una cuenta cripto (VIX mide miedo de renta variable, no de
cripto — un crash cripto puro puede ocurrir con VIX en calma total).

**Auditoría (2026-06-01, ver `AUDITORIA_HALLAZGOS.md` en `feature/polymarket`):**
hallazgos duros, no de tuning sino estructurales:
- El CAGR 51% que decía el README era **la métrica in-sample**. El backtest
  completo real daba 23.4% CAGR / Sharpe 0.69 / MaxDD -35.5%.
- **Out-of-sample (2022+): 7.2% CAGR, Sharpe 0.38 — pierde contra SPY buy&hold
  (11.3%)**. Con LLM + TradingAgents + Polymarket + 40 trades/año y drawdowns
  de 35%, rendía menos que comprar SPY y no tocar nada.
- **Polymarket aportaba 0.0 literal**: 0 mercados relevantes matcheados para
  los 18 tickers del universo, y por diseño nunca entraba en el backtest (solo
  modificaba la decisión del día, sin atribución medible).
- **Bug crítico backtest-vs-vivo**: el backtest validaba `max_positions=2`
  pero la ejecución en vivo compraba 9 posiciones de $5.000 c/u — lo que se
  validó nunca fue lo que se operó.

**Plan de reforma ejecutado** (`PLAN_RENTABILIDAD.md`, `PLAN_REFORMA.md`):
LLM/TradingAgents/Polymarket quedaron **congelados** (no se borraron del
código/config por servir de referencia, pero fuera de la ruta de producción).
Se buscó edge con evidencia académica real: **Time-Series Momentum (TSMOM)**,
managed-futures style, con un siglo de evidencia OOS, funciona largo Y corto,
y su Sharpe viene de **diversificar entre clases de activo descorrelacionadas**
— no de predecir precios. Se validó con walk-forward purgado antes de llevarlo
a producción (2026-06-02). Ver §3.

**Lección explícita para el futuro (repetida en `PLAN_RENTABILIDAD.md`):**
> ❌ Reintroducir LLM/TradingAgents/Polymarket antes de tener señal.
> ❌ Optimizar mirando métricas full/in-sample.
> ❌ Pasar a real sin 30+ días de paper validado.

---

## 3. Estrategia actual en producción: TSMOM multi-activo largo/corto

Código: `src/trading_agent/pipelines/tsmom/` (motor genérico, `strategy.py` es
la fuente única) + `src/trading_agent/pipelines/alpaca_tsmom/` (ejecución).
Config: sección `tsmom:` en `conf/base/parameters.yml` (rama `feature/polymarket`).

**Universo (20 activos, multi-clase, descorrelacionados):**
Renta variable (SPY, QQQ, IWM, EFA, EEM, VNQ) · Renta fija (IEF, TLT, LQD, SHY)
· Reales (GLD, DBC, USO, DBA, SLV) · Divisas (UUP, FXE, FXY) · Cripto
(BTC-USD, ETH-USD — solo 2 de 20, y solo largo, cripto no se puede shortear
en Alpaca).

**Mecánica:** cada activo entra largo si su tendencia propia (lookbacks 3/6/12
meses) es alcista, corto si es bajista — sin ranking cross-sectional entre
activos. Vol-targeting: cada posición y la cartera completa se dimensionan
para un riesgo objetivo (`pvt: 0.12` = 12% vol anual), no por peso igualitario.
Sleeve de reversión (20% peso, lookback ~2 años) como diversificador adicional.
Rebalanceo mensual (`rebal: 21` días hábiles).

**Validación (`tsmom_validation.csv`, reproducible via `kedro run --pipeline
tsmom_live`):** FULL Sharpe **0.84**, CAGR 11.6%, MaxDD -25% | **OOS Sharpe
0.68**. Objetivo de diseño explícito: Sharpe OOS ≥0.7 en ≥4/5 folds,
correlación con SPY ≤0.3 ("la diversificación ES el producto" — gana cuando
las acciones se hunden, sin depender de un gatillo frágil tipo VIX).

**Ejecución en vivo:** `kedro run --pipeline=tsmom_trade` (ingestion + tsmom +
alpaca_tsmom, todo en un solo comando — reemplaza el viejo baile de dos
pipelines `signals`+`alpaca`). Adaptador de órdenes delta (lleva la cuenta a
la posición objetivo), soporta largo y corto. Seguridad: paper salvo
`paper_trading:false` explícito; cortos en ETFs no-shorteables se omiten con
error aislado (no rompe el run); órdenes menores a `min_order_usd` ($25) se
omiten.

**Config de riesgo en vivo (`tsmom:` en parameters.yml — ambas cuentas
comparten esta config desde 2026-08-27):**
```yaml
live_gross: 1.0             # exposición bruta objetivo (1.0 = sin apalancamiento)
live_max_position_pct: 0.30 # cap por posición
min_order_usd: 25.0
```

---

## 4. Incidente de apalancamiento (corregido 2026-08-27)

El 2026-06-02, el mismo día del despliegue de TSMOM, la cuenta **Crypto**
recibió un perfil "AGRESIVO" vía override de CLI en el workflow: `live_gross=2.0,
live_max_position_pct=0.5, pvt=0.24, max_lev=4.0` — es decir, **2x
apalancamiento y doble volatilidad objetivo, sin ningún día de validación en
paper**. Esto contradice directamente el propio `PLAN_RENTABILIDAD.md`
("Apalancamiento: subir live_gross solo tras validar"; sugiere 1.3-1.5x, y
solo tras 60-90 días de paper limpio).

A fecha 2026-08-27 (~86 días / 2-3 rebalanceos mensuales desde el despliegue),
la cuenta Crypto reportaba pérdida y la cuenta Polymarket estancamiento. Esto
es **consistente con**: (a) muestra estadística demasiado pequeña para juzgar
una estrategia de rebalanceo mensual (2-3 puntos de datos no dicen nada), y
(b) el apalancamiento no validado amplificando esa varianza en la cuenta
Crypto. **Se corrigió** revirtiendo la cuenta Crypto a la misma config base
sin apalancar que Polymarket (commit `d334d13` en `develop`). Ambas cuentas
corren ahora idéntica config.

**Regla para el futuro:** no volver a subir `live_gross`/`pvt`/`max_lev` sin
que hayan pasado ≥60-90 días de paper trading limpio (sin cambios de config
en medio) Y sin medir explícitamente que el tracking real vs. backtest es
aceptable. Subir gradual (1.0 → 1.3 → 1.5), nunca de un salto a 2x.

**Nota de timing:** el cron del 2026-08-27 (retrasado por GitHub hasta las
22:58 UTC) disparó 23 minutos ANTES del push del fix, así que ese día corrió
con 2x por última vez. **El primer run a 1x de la cuenta Crypto es el
2026-08-28** — esa es la fecha baseline para el período de validación limpio.

## 4b. Bug de churn cripto (encontrado y corregido 2026-08-27)

Revisando los logs de Actions (item de §7): el adaptador delta TSMOM solo
normalizaba `ETH/USD → ETH-USD`, pero Alpaca devuelve posiciones cripto SIN
separador (`ETHUSD`). La posición quedaba como símbolo huérfano (objetivo 0 →
CLOSE) y el objetivo `ETH-USD` como posición inexistente (→ BUY completo):
**cada run diario cerraba y recompraba la posición cripto entera**, pagando
spread ×2/día (verificado: `CLOSE ETHUSD $164` + `BUY ETH-USD $169` en los
runs del 26 y 27 ago). Con el peso cripto actual (~$170, la vol-targeting
asigna poco a activos de alta vol) el coste eran centavos, pero escalaba
linealmente con el peso. Mismo bug de normalización ya corregido en mayo en
el adaptador legacy, reintroducido en el nuevo. Fix: commit `ab2708c` en
`feature/polymarket` (normalización de 3 variantes + test). Polymarket no lo
sufría: a $10k de equity su objetivo cripto queda bajo `min_order_usd`.

## 4c. Primer chequeo de tracking real vs. backtest (2026-08-27)

Del log de Actions (~86 días desde el despliegue TSMOM del 2026-06-02):

| Cuenta | Config del período | Equity | Retorno | Esperado backtest |
|---|---|---|---|---|
| Polymarket | 1x limpio | $10,226 | **+2.3%** | ~+2.9% ± 6% (1σ) |
| Crypto | 2x no validado | $95,940 | **-4.1%** | (sin backtest a 2x) |

**Polymarket trackea el backtest casi perfecto** — el Gate de Fase E va bien
encaminado en la cuenta limpia. La divergencia de Crypto es coherente con el
apalancamiento no validado (doble varianza + vol drag), no con un fallo del
motor: misma estrategia, mismo período, resultados opuestos según leverage.
Es la evidencia empírica más clara a favor de la regla de §4.

---

## 5. Contexto de mercado 2026 (para no repetir el sesgo del Hantavirus)

Investigado el 2026-08-27 para evaluar si el diseño pandémico-defensivo tenía
sentido dado el régimen real de mercado. Conclusión: **el mercado de 2026 fue
genuinamente alcista/resiliente, con un shock geopolítico agudo pero breve —
no un escenario de crisis lenta tipo pandemia.**

- **S&P 500:** cerró H1 2026 con +9.6% (+10.2% total return). ~85% de las
  empresas del S&P 500 batieron estimados de earnings (vs. 78% promedio
  5 años).
- **Shock geopolítico real del año — guerra Irán:** EEUU/Israel atacaron
  Irán el 28 de febrero 2026. Corrección rápida y en forma de V: S&P -0.8%
  en febrero, Stoxx Europe 600 -8% en marzo (peor mes desde 2022), estimado
  de PIB de la Fed de Atlanta cayó de 3.6% a 1.9%. **Recuperación completa
  en Q2**: S&P +15% y Nasdaq +21% desde fin de marzo, mejor trimestre en 6
  años. Patrón: caída ~6-8% en 3 semanas, recuperación total en las 3
  siguientes — nada parecido a un colapso pandémico prolongado.
- **Fed:** sin recortes en todo 2026 (tasa en 3.5-3.75% desde dic. 2025).
  Inflación por el shock petrolero de la guerra empujó las expectativas de
  recorte a 2027. "Higher for longer", no favorece apuestas puramente
  especulativas sin narrativa propia.
- **Cripto:** BTC cayó a mínimo de 21 meses (~$59.300) en junio — una caída
  específica de cripto, **sin que el VIX de renta variable se moviera** (la
  prueba definitiva de que VIX era el proxy equivocado para riesgo cripto).
  En julio, rotación violenta: acciones de semiconductores/IA cayeron -22-25%
  (el índice SOX perdió $2.2 billones de capitalización) por dudas sobre la
  sostenibilidad del capex de IA (Meta Compute, modelo chino Moonshot,
  retraso de SK Hynix en HBM4) — y ese capital rotó hacia cripto (ETH +20%,
  BTC +9% en julio). En agosto BTC siguió subiendo hasta ~$80k.
- **Hantavirus:** contenido en un solo crucero (MV Hondius), 13 casos, 3
  muertes, declarado terminado por la OMS el 2 de julio. **Cero impacto de
  mercado.** Diseñar en torno a esto como "amenaza pandémica inminente" fue
  el sesgo correcto de identificar y corregir.

**Implicación para el diseño:** TSMOM ya es la respuesta estructuralmente
correcta a este contexto — no necesita "saber" que el mercado es alcista
(lo detecta solo, por activo, vía su propia tendencia de precio) y su
protección ante shocks agudos viene de poder ir corto y de diversificar entre
clases de activo, no de un gatillo binario tipo VIX. **No hace falta
recalibrar el motor para "mercado creciente"** — eso ya es lo que hace por
diseño. Lo que sí se corrigió fue el apalancamiento prematuro (§4).

---

## 6. Qué NO hacer (reglas duras, con evidencia detrás)

1. **No reintroducir LLM, TradingAgents ni sentimiento de Polymarket** en la
   ruta de producción sin datos históricos reales que permitan backtestear
   la señal (Polymarket Gamma API solo expone mercados activos, no serie
   histórica — no es backtesteable tal como está diseñada la API).
2. **No subir apalancamiento sin validar** (ver §4). 60-90 días mínimo,
   subida gradual.
3. **No juzgar la estrategia con <6 meses de datos live** (rebalanceo
   mensual → necesitas muchos ciclos para separar señal de ruido).
4. **No optimizar mirando métricas full/in-sample.** Siempre walk-forward
   OOS, con costes realistas (slippage, comisiones, borrow en cortos).
5. **No diseñar controles de riesgo alrededor de un evento puntual/noticia**
   (lección del Hantavirus) — calibrar con datos de mercado, no con miedo.
6. **No rehacer el motor de estrategia sin evidencia nueva que lo justifique.**
   Ya pasó por un proceso riguroso (auditoría → hipótesis → walk-forward →
   robustez → paper). Rehacerlo sin evidencia repetiría exactamente el error
   que se está corrigiendo ahora (cambios impulsivos, como el salto a 2x
   leverage el mismo día del despliegue).

## 7. Qué SÍ vigilar / próximos pasos razonables

- Dejar correr ambas cuentas sin tocar config hasta completar 60-90 días
  desde el **2026-08-28** (primer run limpio a 1x en ambas; idealmente hasta
  fin de octubre / noviembre 2026).
- ✅ ~~Revisar `tsmom_alpaca_log.csv` por errores de ejecución~~ — hecho
  2026-08-27: sin errores de shorts en runs recientes; se encontró y corrigió
  el bug de churn cripto (§4b). Repetir el chequeo de logs ~mensualmente.
- ✅ ~~Primer chequeo de tracking real vs. backtest~~ — hecho 2026-08-27,
  resultados en §4c (Polymarket on-track). Repetir al cierre del período de
  validación con la serie completa de equity.
- ✅ ~~Rama huérfana `feature/crypto`~~ — archivada como tag y borrada
  (2026-08-27, ver §1).
- Limpieza opcional de bajo riesgo: si en algún momento se quiere borrar de
  verdad el código legacy (LLM/TradingAgents/Polymarket-sentiment, pipeline
  `llm_agents`, `alpaca` long-only), hacerlo en una rama aparte con tests,
  ya que `run_ablation.py` / `run_param_sweep.py` / `run_blend.py` (scripts
  de investigación) pueden depender de esa config para comparaciones.

## 8. Mapa de archivos clave

| Qué | Dónde |
|---|---|
| Estrategia TSMOM (motor) | `feature/polymarket:src/trading_agent/pipelines/tsmom/strategy.py` |
| Config viva | `feature/polymarket:conf/base/parameters.yml` (sección `tsmom:`) |
| Ejecución Alpaca long/short | `feature/polymarket:src/trading_agent/pipelines/alpaca_tsmom/` |
| Auditoría que mató el modelo LLM | `feature/polymarket:AUDITORIA_HALLAZGOS.md` |
| Plan de reforma (filosofía, gates) | `feature/polymarket:PLAN_RENTABILIDAD.md`, `PLAN_REFORMA.md` |
| Resultados de optimización | `feature/polymarket:OPTIMIZACION_RESULTADOS.md` |
| Workflow crypto (cuenta apalancada→des-apalancada) | `develop:.github/workflows/crypto-signals.yml` |
| Workflow polymarket | `develop:.github/workflows/polymarket-signals.yml` |
| Este briefing | `develop:docs/FABLE_BRIEFING.md` |

## 9. Cómo colaborar con Claudio (normas observadas)

- Prefiere que se actúe con criterio propio cuando la evidencia es clara, en
  vez de preguntar en cada paso — pero valora que se le explique el
  razonamiento y los trade-offs en el texto de respuesta.
  Reservar preguntas (`AskUserQuestion`) para decisiones genuinamente suyas
  (apetito de riesgo, alcance de producto), no para cosas inferibles de la
  evidencia.
- Prefiere respuestas en español, directas, sin relleno.
- Ya se equivocó una vez sesgando el diseño por una noticia puntual
  (Hantavirus) y por impaciencia (apalancar sin validar) — pedirá
  explícitamente evidencia/datos de mercado antes de cambios de riesgo
  grandes; seguir ese hábito de verificar contra datos reales antes de tocar
  parámetros de riesgo.
- Le interesa el resultado económico real, no la elegancia técnica del
  sistema — cualquier propuesta de cambio debe justificarse en términos de
  Sharpe/CAGR/MaxDD con evidencia OOS, no en teoría.
