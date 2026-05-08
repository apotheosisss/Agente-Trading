# 🤖 Agente de Trading con IA

> Sistema autónomo de inversión impulsado por múltiples agentes LLM, análisis técnico, datos de predicción (Polymarket) y ejecución automatizada en Alpaca paper trading. Arquitectura modular basada en Kedro con despliegue continuo via GitHub Actions.

---

## 📋 Descripción General

Sistema de trading algorítmico que combina indicadores técnicos cuantitativos, análisis de sentimiento y múltiples agentes de lenguaje (LLM) para generar señales de inversión (BUY / HOLD / SELL). Las señales pasan por un filtro de confirmación multi-agente (TradingAgents) antes de ejecutarse automáticamente en cuentas de paper trading de Alpaca.

El sistema opera **dos modelos de inversión paralelos e independientes**, cada uno con su propio universo de activos, cuenta de paper trading y horario de ejecución.

---

## 🏗️ Arquitectura

```
Yahoo Finance ──► Datos OHLCV (400 días)
     ^VIX ──────► Índice de volatilidad
Polymarket ──────► Sentimiento de mercados de predicción
        │
        ▼
┌─────────────────────────────────────────┐
│          Pipeline: signals              │
│                                         │
│  Ingesta → Limpieza → Features          │
│       │                                 │
│       ▼                                 │
│  Agente Técnico  ─┐                     │
│  Agente Sentimiento ─► Agente Decisión  │
│  Agente Riesgo   ─┘        │            │
│                             ▼           │
│              Filtro TradingAgents       │
│           (confirmación multi-agente)   │
│                    │                    │
│                    ▼                    │
│             verified_signal             │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│          Pipeline: alpaca               │
│                                         │
│  Verificar cuenta ─► Ejecutar órdenes  │
│  (no duplica posiciones abiertas)       │
│         ▼                              │
│  Sincronizar posiciones                 │
└─────────────────────────────────────────┘
        │
        ▼
  Alpaca Paper Trading API
        │
        ▼
  Notificación Telegram
```

---

## 🔀 Modelos de Inversión

El proyecto mantiene **dos ramas activas** con modelos independientes:

### 📈 Modelo Crypto (`feature/crypto`)

| Parámetro | Valor |
|---|---|
| Universo análisis | 11 activos: BTC, ETH, BNB, XRP, SOL, AVAX, LINK, DOGE, LTC, COIN, SPY |
| Universo ejecución | 10 activos crypto/COIN — **SPY excluido de órdenes** (referencia macro únicamente) |
| Enfoque | Ciclos bull/bear de criptomonedas |
| Frecuencia | Diario — **09:15 AM CLT** (13:15 UTC) |
| Max posiciones | 2 |
| Cuenta Alpaca | `ALPACA_API_KEY` |
| Backtest CAGR | ~61.5% \| Sharpe ~1.09 |
| Rebalanceo | Cada 15 días (óptimo walk-forward) |

### 📊 Modelo Polymarket (`feature/polymarket`)

| Parámetro | Valor |
|---|---|
| Universo | 18 activos: SPY, QQQ, NVDA, META, AMZN, GOOGL, AAPL, MSFT, BTC, ETH, XLK, SOXX, CVX, SLB, HAL, VLO, OXY, XLE |
| Enfoque | Tech + energía (tesis Venezuela) con sentimiento Polymarket |
| Frecuencia | Lunes a viernes — **17:20 CLT** (21:20 UTC) |
| Max posiciones | 2 |
| Cuenta Alpaca | `ALPACA_API_KEY_POLY` (cuenta separada) |
| Backtest CAGR | ~51.1% \| Sharpe ~1.13 \| MaxDD -15.9% |
| Rebalanceo | Cada 15 días |

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Framework de pipelines | Kedro 1.3.1 |
| Gestión de entorno | UV (Astral) |
| Datos de mercado | yfinance |
| Modelos LLM | OpenAI GPT via OpenRouter (`openai/gpt-5-nano`) |
| Confirmación multi-agente | TradingAgents (LangGraph) |
| Sentimiento predictivo | Polymarket Gamma API |
| Ejecución de órdenes | Alpaca Trading API (`alpaca-py`) |
| CI/CD | GitHub Actions |
| Notificaciones | Telegram Bot |
| Lenguaje | Python 3.12 |

---

## 📁 Estructura del Proyecto

```
trading-agent/
├── .github/
│   └── workflows/
│       ├── crypto-signals.yml      # Ejecución diaria modelo crypto
│       └── polymarket-signals.yml  # Ejecución diaria modelo polymarket
├── conf/
│   ├── base/
│   │   ├── catalog.yml             # Datasets (incluye verified_signal)
│   │   ├── parameters.yml          # Universo, riesgo, LLM, backtesting
│   │   └── logging.yml
│   └── local/
│       └── credentials.yml         # API keys — NO en Git
├── data/
│   ├── 01_raw/                     # OHLCV, VIX, Polymarket
│   ├── 02_intermediate/            # Datos limpios
│   ├── 03_primary/                 # Features técnicos y sentimiento
│   ├── 04_feature/                 # Feature vector unificado
│   ├── 07_model_output/            # Señales, verified_signal, órdenes Alpaca
│   └── 08_reporting/               # Métricas, equity curve, walk-forward
├── src/trading_agent/
│   └── pipelines/
│       ├── ingestion/              # Descarga OHLCV, VIX, Polymarket
│       ├── feature_engineering/    # RSI, MACD, BB, EMA200, ATR, sentimiento
│       ├── llm_agents/             # Agentes técnico/sentimiento/riesgo/decisión + TradingAgents
│       ├── backtesting/            # Backtest + walk-forward + métricas
│       └── alpaca/                 # Ejecución real en Alpaca
├── pyproject.toml
└── README.md
```

---

## ⚙️ Flujo Completo de Ejecución

### GitHub Actions (automático)

```
Cada día — modelo crypto:
  13:15 UTC → kedro run --pipeline=signals
           → kedro run --pipeline=alpaca
           → Notificación Telegram

Lun-Vie — modelo polymarket:
  21:20 UTC → kedro run --pipeline=signals
           → kedro run --pipeline=alpaca
           → Notificación Telegram
```

### Controles de seguridad en ejecución de órdenes

- Solo señales BUY con `confidence >= 0.65`
- Máximo `$5,000 USD` por orden
- Máximo `15%` del portfolio en un solo activo
- Reserva mínima de `5%` en cash
- **No duplica posiciones**: si el activo ya está en cartera, la orden se omite

---

## 🚀 Instalación Local

### Prerrequisitos
- [UV (Astral)](https://astral.sh/uv)
- Python 3.12+
- Git

```bash
# 1. Clonar el repositorio
git clone https://github.com/apotheosisss/Agente-Trading.git
cd Agente-Trading

# 2. Elegir modelo
git checkout feature/crypto       # modelo crypto
# o
git checkout feature/polymarket   # modelo polymarket

# 3. Instalar dependencias
uv sync

# 4. Configurar credenciales
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

openai:
  api_key: "TU_OPENROUTER_API_KEY"
```

---

## ▶️ Ejecución Local

```bash
# Pipeline de señales
END=$(date +%Y-%m-%d)
START=$(date -d "400 days ago" +%Y-%m-%d)
uv run kedro run --pipeline=signals --params="start_date=$START,end_date=$END"

# Pipeline de órdenes Alpaca
uv run kedro run --pipeline=alpaca

# Solo backtesting
uv run kedro run --pipeline=backtesting

# Pipeline completo
uv run kedro run
```

---

## 🔑 Secrets de GitHub Actions

Configurar en **Settings → Secrets and variables → Actions**:

| Secret | Descripción |
|---|---|
| `OPENAI_API_KEY` | API key de OpenRouter |
| `ALPACA_API_KEY` | API key cuenta Alpaca — modelo crypto |
| `ALPACA_SECRET_KEY` | Secret key cuenta Alpaca — modelo crypto |
| `ALPACA_API_KEY_POLY` | API key cuenta Alpaca — modelo polymarket |
| `ALPACA_SECRET_KEY_POLY` | Secret key cuenta Alpaca — modelo polymarket |
| `TELEGRAM_BOT_TOKEN` | Token del bot de Telegram |
| `TELEGRAM_CHAT_ID` | Chat ID para notificaciones |

---

## 📊 Indicadores Técnicos

| Indicador | Parámetro |
|---|---|
| RSI | 14 períodos |
| MACD | 12 / 26 / 9 |
| Bollinger Bands | 20 períodos, 2σ |
| EMA | 200 períodos (filtro de tendencia) |
| ATR | Stop-loss dinámico (4.5× ATR) |

---

## 🔒 Reglas de Seguridad

1. `conf/local/credentials.yml` **nunca** se sube a Git
2. El sistema opera exclusivamente en **paper trading** hasta revisión manual de ≥ 30 días
3. Para pasar a live: cambiar `paper_trading: false` en credentials.yml solo tras 30 días de revisión manual
4. Máximo `$5,000 USD` por orden, nunca más del 15% del portfolio en un activo

---

## 👤 Autor

**Claudio** — Ingeniería Informática mención Ciencia de Datos, DuocUC  
Proyecto personal de inversión algorítmica — en evaluación activa (paper trading)
