# 🤖 Agente de Trading Algorítmico — TSMOM

> Sistema de inversión autónoma basado en **Time-Series Momentum (TSMOM)** multi-activo largo/corto, validado con walk-forward out-of-sample, desplegado en paper trading (Alpaca) vía GitHub Actions. Kedro + Python + UV.

---

## 📋 Descripción General

Investigación personal de trading algorítmico. **No es un caso de estudio — el objetivo es un sistema de inversión autónoma genuinamente rentable**, evaluado en paper trading antes de considerar capital real.

La estrategia en producción es **TSMOM** (managed-futures style, con evidencia académica de un siglo): cada activo de un universo de 20 instrumentos descorrelacionados entra **largo si su tendencia propia es alcista, corto si es bajista**, dimensionado por volatilidad objetivo de cartera (no por peso igualitario). El Sharpe no viene de predecir precios — viene de **diversificar entre clases de activo que no se mueven juntas**.

Dos cuentas Alpaca paper corren **el mismo código y la misma estrategia** en paralelo, como dos réplicas independientes para comparar tracking real vs. backtest:

| Cuenta | Secrets | Cron | Estado |
|---|---|---|---|
| **Polymarket** | `ALPACA_API_KEY_POLY` / `ALPACA_SECRET_KEY_POLY` | Lun-Vie 21:20 UTC | Corriendo desde 2026-06-02 |
| **Crypto** | `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Lun-Vie 13:15 UTC | Cuenta reiniciada a $100k el 2026-09-01 |

Los nombres son historia — ninguna de las dos opera "cripto" ni "sentimiento de Polymarket" específicamente; el universo mezcla renta variable, renta fija, materias primas, divisas y 2 criptomonedas.

**Documentación completa del historial y decisiones:** [`docs/FABLE_BRIEFING.md`](docs/FABLE_BRIEFING.md) — léelo antes de tocar cualquier config.

---

## 🏗️ Cómo llegó a ser esto (resumen — detalle completo en el briefing)

1. **Diseño original:** agentes LLM (técnico/sentimiento/riesgo/decisión) + confirmación multi-agente (TradingAgents) + sentimiento de Polymarket como boost de score.
2. **Auditoría (2026-06-01):** el 51% CAGR que decía el README era la métrica *in-sample*. El backtest real daba 23.4% CAGR, y **out-of-sample perdía contra comprar y mantener SPY** (7.2% vs 11.3%). Polymarket aportaba señal **0.0 literal** — 0 mercados relevantes matcheados. Bug adicional: el backtest validaba 2 posiciones pero la ejecución en vivo compraba 9.
3. **Reforma:** LLM/TradingAgents/Polymarket quedaron **congelados** (código conservado como referencia, fuera de producción). Se buscó edge con evidencia real → TSMOM, validado con walk-forward antes de desplegar (2026-06-02).
4. **Incidente de apalancamiento (corregido):** la cuenta Crypto se apalancó a 2x el mismo día del despliegue, sin validar. Se revirtió a 1x el 2026-08-27.
5. **Reset de cuenta (2026-09-01):** Alpaca eliminó el botón de reset paper (ahora requiere crear cuenta nueva). Se creó cuenta Crypto nueva en $100k, se regeneraron las API keys, y **esa fecha es el baseline oficial** del período de validación de 60-90 días.

---

## 🧠 Estrategia: TSMOM multi-activo largo/corto

**Universo (20 activos, multi-clase, descorrelacionados):**

| Clase | Activos |
|---|---|
| Renta variable | SPY, QQQ, IWM, EFA, EEM, VNQ |
| Renta fija | IEF, TLT, LQD, SHY |
| Reales | GLD, DBC, USO, DBA, SLV |
| Divisas | UUP, FXE, FXY |
| Cripto | BTC-USD, ETH-USD (solo largo — no shorteable en Alpaca) |

**Mecánica:** tendencia propia por activo (lookbacks 3/6/12 meses, `signal_mode: strength`) — sin ranking cross-sectional. Vol-targeting: cartera dimensionada a un riesgo objetivo anual (`pvt: 0.12` = 12%), no peso igualitario. Sleeve de reversión (20% peso, ~2 años lookback) como diversificador. Rebalanceo mensual.

**Validación out-of-sample:** FULL Sharpe **0.84** (CAGR 11.6%, MaxDD -25%) | **OOS Sharpe 0.68**. Objetivo de diseño: correlación con SPY ≤0.3 — gana cuando las acciones caen, sin depender de un gatillo frágil tipo VIX.

**Ejecución en vivo:** `kedro run --pipeline=tsmom_trade` (ingesta + pesos + ejecución Alpaca en un solo comando). Adaptador de órdenes delta, soporta largo y corto, `live_gross: 1.0` (sin apalancamiento) en ambas cuentas.

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Framework de pipelines | Kedro 1.3.1 |
| Gestión de entorno | UV (Astral) |
| Datos de mercado | yfinance |
| Ejecución de órdenes | Alpaca Trading API (`alpaca-py`), largo y corto |
| CI/CD | GitHub Actions |
| Notificaciones | Telegram Bot |
| Lenguaje | Python 3.12 |
| *(Legado, congelado)* | OpenAI GPT / OpenRouter, TradingAgents (LangGraph), Polymarket Gamma API — ver §Historia |

---

## 📁 Estructura del Proyecto

```
trading-agent/
├── .github/workflows/
│   ├── crypto-signals.yml          # Cuenta "Crypto" — corre TSMOM
│   └── polymarket-signals.yml      # Cuenta "Polymarket" — corre TSMOM
├── docs/
│   └── FABLE_BRIEFING.md           # Historial completo, decisiones, reglas
├── conf/base/
│   ├── catalog.yml                 # Datasets (incluye tsmom_weights, tsmom_orders...)
│   ├── parameters.yml              # Universo + sección `tsmom:` (config viva)
│   └── logging.yml
├── src/trading_agent/pipelines/
│   ├── tsmom/                      # Motor de estrategia (fuente única: strategy.py)
│   ├── alpaca_tsmom/                # Ejecución largo/corto en Alpaca
│   ├── ingestion/                  # Descarga OHLCV del universo
│   ├── llm_agents/, polymarket/,    # LEGADO — congelado, no operar
│   │   alpaca/ (parte long-only)
│   └── backtesting/                # Walk-forward, métricas (usado por tsmom_live tambien)
├── AUDITORIA_HALLAZGOS.md          # Por qué se abandonó el modelo LLM
├── PLAN_RENTABILIDAD.md            # Filosofía, gates de decisión
├── IMPLEMENTACION_TSMOM.md         # Detalle técnico de la estrategia en producción
└── pyproject.toml
```

---

## ⚙️ Flujo de Ejecución (GitHub Actions)

```
Ambas cuentas, Lun-Vie:
  cron → kedro run --pipeline=tsmom_trade --params="start_date=2015-01-01,end_date=$END"
       → calcula pesos objetivo (tsmom_weights.csv)
       → lleva la cuenta a esa posición via órdenes delta (largo/corto)
       → Notificación Telegram (señal, órdenes, equity)
```

Nota: el cron de GitHub Actions en este repo ha mostrado retrasos de varias horas de forma recurrente — no tratar un run tardío como fallo del sistema.

---

## 🚀 Instalación Local

```bash
git clone https://github.com/apotheosisss/Agente-Trading.git
cd Agente-Trading
git checkout feature/polymarket   # rama con el código de producción (TSMOM)
uv sync
mkdir -p conf/local
```

Crear `conf/local/credentials.yml`:

```yaml
alpaca:
  api_key: "TU_ALPACA_API_KEY"
  secret_key: "TU_ALPACA_SECRET_KEY"
  paper_trading: true

telegram:
  bot_token: "TU_BOT_TOKEN"
  chat_id: "TU_CHAT_ID"
```

---

## ▶️ Ejecución Local

```bash
# Estrategia completa: ingesta + pesos TSMOM (sin ejecutar órdenes)
uv run kedro run --pipeline=tsmom_live

# Estrategia + ejecución en Alpaca paper (ingesta + pesos + órdenes)
uv run kedro run --pipeline=tsmom_trade

# Solo ejecución, con tsmom_weights ya calculado
uv run kedro run --pipeline=alpaca_tsmom
```

Salidas: `data/07_model_output/tsmom_weights.csv`, `tsmom_orders.csv`, `tsmom_alpaca_log.csv` · `data/08_reporting/tsmom_report.txt`, `tsmom_validation.csv`.

---

## 🔑 Secrets de GitHub Actions

| Secret | Descripción |
|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Cuenta Alpaca "Crypto" |
| `ALPACA_API_KEY_POLY` / `ALPACA_SECRET_KEY_POLY` | Cuenta Alpaca "Polymarket" |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Notificaciones |
| `OPENAI_API_KEY` | Legado — solo si se reactiva el pipeline LLM congelado |

---

## 🔒 Reglas duras (con evidencia detrás — detalle en el briefing)

1. **No reintroducir LLM/TradingAgents/Polymarket** en producción sin datos históricos backtesteables.
2. **No subir apalancamiento sin validar** ≥60-90 días de paper limpio, subida gradual (1.0→1.3→1.5), nunca de un salto.
3. **No juzgar la estrategia con <6 meses de datos live** (rebalanceo mensual).
4. **No optimizar mirando métricas full/in-sample** — siempre walk-forward OOS con costes realistas.
5. **No diseñar controles de riesgo alrededor de una noticia puntual** — calibrar con datos de mercado.
6. **No rehacer el motor sin evidencia nueva** — ya pasó por auditoría → hipótesis → walk-forward → robustez → paper.
7. `conf/local/credentials.yml` **nunca** se sube a Git. Sistema exclusivamente en **paper trading**.

---

## 👤 Autor

**Claudio** — Ingeniería Informática mención Ciencia de Datos, DuocUC
Proyecto personal de inversión algorítmica — en período de validación activa (paper trading)
