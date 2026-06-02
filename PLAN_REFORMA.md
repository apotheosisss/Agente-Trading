# Plan de Reforma — Hacia un sistema genuinamente rentable

> Resultado de investigación (evidencia académica + diagnóstico del código). Sustituye la meta
> "batir a SPY con momentum técnico" — demostrada inviable — por un objetivo alcanzable y honesto.

---

## 0. Diagnóstico de raíz: por qué el sistema actual NO puede ser rentable

Tras probar 2 universos × 5 familias de señal, el techo no es de tuning sino **estructural**. Cuatro límites, cada uno respaldado por evidencia:

| # | Límite estructural | Por qué mata el edge | Evidencia |
|---|---|---|---|
| 1 | **Solo precio** (RSI/EMA/momentum) | No hay ventaja informativa; todo el mundo ve lo mismo | Decay ~5pp Sharpe/año tras publicación |
| 2 | **Solo largo** (long-only) | Solo puede huir a cash en bajadas; no monetiza caídas | Long-short > long-only en retorno ajustado a riesgo (con AUM pequeño = ventaja retail) |
| 3 | **Large-caps líquidas** (SPY/QQQ/AAPL…) | El mercado más eficiente = el menos explotable | El decay es mucho menor en activos pequeños/ilíquidos |
| 4 | **Una sola clase de activo** (renta variable US) | Una sola fuente de retorno; todo cae junto en crisis | El Sharpe del trend-following viene de DIVERSIFICAR muchos mercados descorrelacionados |

**Conclusión:** el sistema está construido sobre las 4 condiciones donde el edge es más débil. No es un problema de parámetros; es de **diseño**.

---

## 1. Nueva meta (realista, medible, honesta)

**Ya NO:** "obtener más CAGR que SPY". (Imposible de forma robusta con señales de precio sobre ETFs líquidos — demostrado.)

**Ahora:** construir un sistema con **retorno esperado positivo real, Sharpe ~0.7–1.0, DESCORRELACIONADO de la renta variable** — rentable por sí mismo y que **gana cuando las acciones se hunden** ("crisis alpha"). Ese es el perfil del trend-following / managed futures, y es genuinamente valioso y alcanzable.

**Métricas de éxito nuevas:**
- Sharpe OOS ≥ 0.7, consistente en ≥4/5 folds.
- Correlación con SPY ≤ 0.3 (la diversificación ES el producto).
- Retorno positivo en años de bear de equities (2022, 2020-Q1).
- Sobrevive a costes realistas (incl. roll/borrow) y a validación purgada.

---

## 2. El edge elegido y por qué (evidencia, no fe)

**Trend-following / Time-Series Momentum (TSMOM) multi-activo, long & short.** Es el edge sistemático con **mejor evidencia out-of-sample que existe**:
- Un **siglo** de evidencia, en equity/bonos/divisas/materias primas.
- **Sin límites de capacidad** (mercados de futuros muy líquidos) → no se arbitra hasta desaparecer.
- Funciona **largo Y corto**, en **muchos mercados descorrelacionados** — y ahí está el Sharpe: no en predecir, sino en diversificar trends.
- Es un **premio de riesgo + sesgo conductual**, no un truco que se publica y muere.

**Track secundario (mayor riesgo): cripto.** Mercado retail-driven, estructuralmente menos eficiente → hay momentum explotable, **pero** la evidencia es fuerte para *time-series* momentum y débil/mixta para *cross-sectional*, y los costes de transacción se comen buena parte. Accesible y de bajo capital, a cambio de más riesgo y fragilidad.

---

## 3. Reformas por capa (qué cambia en el código)

| Capa | Hoy | Reforma |
|---|---|---|
| **Instrumentos** | ETFs long-only vía Alpaca | Long **y short**. Opción A: futuros (requiere broker tipo IBKR/Tradovate). Opción B (implementable ya): ETFs de clase de activo + **ETFs inversos** / short en Alpaca para replicar trend long/short. |
| **Universo** | 12 ETFs (sesgo equity) | ~15-25 mercados **descorrelacionados**: índices equity (US/intl/EM), bonos (corto/largo), divisas (DXY/EUR/JPY), materias primas (oro, petróleo, agrícolas, metales), y cripto opcional. |
| **Señal** | Cross-sectional ranking (top-N) | **Time-series momentum**: cada mercado largo si su propia tendencia es alcista, corto si bajista. Múltiples lookbacks (1/3/6/12m) combinados. Sin ranking entre activos. |
| **Sizing** | Peso igual / vol_sizing simple | **Volatility targeting / risk parity**: cada mercado contribuye el mismo riesgo; vol objetivo de cartera fija (p.ej. 10-15% anual). *Esto es lo que produce el Sharpe.* |
| **Frecuencia** | Rebalanceo semanal | Diaria-semanal con holding largo (semanas-meses). Trend = paciencia. |
| **Datos** | yfinance diario | yfinance basta para ETFs/cripto. Para **futuros reales** hace falta histórico de contratos continuos (Norgate, CSI, o vía broker) — yfinance NO sirve ahí. |
| **Validación** | Walk-forward + costes (ya hecho) | Mantener. Añadir **costes de roll** (futuros) o **borrow** (shorts ETF), y validación purgada/embargada. |
| **Ejecución** | Alpaca paper | Alpaca sirve para ETFs largos/cortos y cripto. Futuros NO (Alpaca no opera futuros) → decidir broker. |

---

## 4. Roadmap por fases (con gates)

**FASE R1 — Reorientar a TSMOM long/short sobre ETFs multi-activo (implementable en stack actual)**
- Universo diversificado con ETFs + inversos (o short Alpaca).
- Nueva señal `tsmom` en `compute_scores`: signo de la tendencia propia (no ranking), multi-lookback.
- Sizing por volatility targeting.
- **Gate R1:** Sharpe OOS ≥ 0.6, correlación con SPY ≤ 0.4, positivo en 2022.

**FASE R2 — Diversificar a verdaderas clases de activo (divisas, materias primas, bonos)**
- Ampliar universo con ETFs de FX/commodities/rates.
- Medir cuánto sube el Sharpe la diversificación.
- **Gate R2:** Sharpe OOS ≥ 0.7 estable en ≥4/5 folds.

**FASE R3 — Decidir instrumento de producción**
- Si R2 pasa: evaluar pasar a **futuros reales** (mejor coste/apalancamiento/short) → requiere broker + datos de futuros + más capital.
- O quedarse en ETFs (más simple, peor coste de short).

**FASE R4 — Track cripto (paralelo, opcional, alto riesgo)**
- TSMOM sobre top-N cripto por liquidez vía Alpaca crypto.
- Costes realistas agresivos (cripto es caro de operar).
- **Gate R4:** sobrevive a costes de 20-30 bps/operación.

**FASE R5 — Robustez, paper 60-90 días, capital real pequeño** (igual que el plan anterior).

---

## 5. Expectativas honestas (lee esto dos veces)

- **Sharpe realista del trend-following diversificado: ~0.5–1.0.** No 2-3. Quien promete más, miente o sobre-ajusta.
- **Tiene décadas flacas.** Los 2010s fueron mediocres para CTAs; 2022 fue excelente. Hay que aguantar años planos.
- **NO batirá a SPY en un bull sostenido** — y no es su trabajo. Su valor es ganar cuando SPY pierde y descorrelacionar.
- **El retorno viene de diversificación + disciplina + costes bajos**, no de predecir. Si buscas emoción o enriquecerte rápido, esto no es eso.
- **Riesgo de implementación:** shorts/futuros añaden complejidad (borrow, roll, margin, gaps). Más superficie de fallo que long-only.

---

## 6. Requisitos reales

| Recurso | Track ETF long/short | Track futuros | Track cripto |
|---|---|---|---|
| Capital mínimo razonable | $10-25k | $50-100k+ | $2-10k |
| Broker | Alpaca (actual) | IBKR/Tradovate | Alpaca/exchange |
| Datos | yfinance (ok) | futuros continuos (de pago) | yfinance/exchange API |
| Complejidad nueva | media (shorts) | alta | media-alta |
| Edge esperado | medio | mejor (coste/short) | alto pero frágil |

**Recomendación de arranque:** **Fase R1 en el stack actual** (TSMOM long/short sobre ETFs + inversos). Cero coste de cambio, prueba la tesis central (diversificación + trend + short) antes de invertir en broker de futuros o capital.

---

## 7. Qué se conserva del trabajo previo
- El **framework de validación** (walk-forward multi-ventana, costes realistas, ablation, gates) — es exactamente lo que se necesita y ya está construido.
- El **motor de riesgo** (circuit breaker por régimen, vol sizing) — reutilizable.
- La **arquitectura Kedro** (ingesta→features→señal→backtest→ejecución) — sólida; solo cambian universo, señal y sizing.
- Se mantienen **congelados**: LLM, TradingAgents, Polymarket (no aportan alfa, confirmado empíricamente).

---

### Fuentes (evidencia)
- [A Century of Evidence on Trend-Following Investing (Hurst/Ooi/Pedersen, AQR)](https://fairmodel.econ.yale.edu/ec439/hurst.pdf)
- [Time Series Momentum (aka Trend-Following): historical evidence — Alpha Architect](https://alphaarchitect.com/time-series-momentum-aka-trend-following-the-historical-evidence/)
- [Momentum Strategies in Futures Markets and Trend-following Funds (EFMA)](https://www.efmaefm.org/0efmameetings/efma%20annual%20meetings/2012-Barcelona/papers/BK_MOMF_Full.pdf)
- [Why and how systematic strategies decay (arXiv)](https://arxiv.org/pdf/2105.01380)
- [When do systematic strategies decay? (Quantitative Finance)](https://www.tandfonline.com/doi/full/10.1080/14697688.2022.2098810)
- [Equity Factors: To Short Or Not To Short (arXiv)](https://arxiv.org/pdf/2003.10419)
- [Long-Short vs Long-Only Implementation of Equity Factors — QuantPedia](https://quantpedia.com/long-short-vs-long-only-implementation-of-equity-factors/)
- [Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565)
- [A Trend Factor for the Cross Section of Cryptocurrency Returns (JFQA, Cambridge)](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178)
