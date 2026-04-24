# Documento de Diseño de Software (SDD)
## Agente de Trading con IA — v0.1.0

**Proyecto:** Agente de Trading Autónomo con LLM  
**Autor:** Claudio  
**Institución:** DuocUC — Ingeniería Informática mención Ciencia de Datos  
**Fecha:** Abril 2026  
**Estado:** En desarrollo activo

---

## 1. Introducción

### 1.1 Propósito

Este documento describe el diseño técnico del Agente de Trading con IA. Sirve como referencia arquitectónica para el desarrollo, mantenimiento y extensión del sistema. Es la fuente de verdad para decisiones de diseño.

### 1.2 Alcance

El sistema cubre el ciclo completo de trading algorítmico:

- Extracción y validación de datos de mercado
- Cálculo de indicadores técnicos y análisis de sentimiento
- Orquestación de agentes LLM para toma de decisiones
- Validación histórica mediante backtesting
- Ejecución de órdenes en modo paper trading

### 1.3 Definiciones

| Término | Definición |
|---|---|
| OHLCV | Open, High, Low, Close, Volume — datos estándar de velas japonesas |
| Pipeline | Secuencia ordenada de nodos de procesamiento en Kedro |
| Nodo | Función Python pura que transforma un input en un output |
| Señal | Decisión de trading: BUY, SELL o HOLD |
| Paper trading | Simulación de operaciones sin dinero real |
| Backtesting | Evaluación de estrategia sobre datos históricos |
| LLM | Large Language Model — modelo de lenguaje de gran escala |

---

## 2. Arquitectura del Sistema

### 2.1 Visión General

El sistema sigue una arquitectura de **pipeline orientado a datos** implementada con Kedro. Cada capa es independiente y se comunica exclusivamente a través del Data Catalog.

```
┌─────────────────────────────────────────────────┐
│              FUENTES DE DATOS                   │
│  Yahoo Finance        NewsAPI / RSS             │
└──────────────┬──────────────────┬───────────────┘
               │                  │
               ▼                  ▼
┌─────────────────────────────────────────────────┐
│           PIPELINE: INGESTION                   │
│  obtener_datos_mercado()  obtener_noticias()    │
│  validar_datos_mercado()  validar_noticias()    │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│       PIPELINE: FEATURE ENGINEERING             │
│  calcular_indicadores_tecnicos()                │
│  calcular_sentimiento()                         │
│  ensamblar_vector_features()                    │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│           PIPELINE: LLM AGENTS                  │
│  agente_tecnico()    agente_sentimiento()       │
│  agente_riesgo()     agente_decision()          │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│           PIPELINE: BACKTESTING                 │
│  ejecutar_backtest()   calcular_metricas()      │
│  generar_reporte()                              │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│           PIPELINE: EXECUTION                   │
│  verificar_riesgo()    enviar_orden()           │
│  actualizar_portafolio()                        │
└─────────────────────────────────────────────────┘
```

### 2.2 Principios de Diseño

**Separación de responsabilidades:** Cada pipeline tiene una única responsabilidad. Ningún pipeline accede directamente al siguiente — toda comunicación es a través del Data Catalog.

**Funciones puras:** Los nodos Kedro son funciones Python sin efectos secundarios. Reciben datos, los transforman y retornan resultados. No leen ni escriben archivos directamente.

**Configuración centralizada:** Todos los parámetros del sistema se definen en `conf/base/parameters.yml`. Ningún valor está hardcodeado en el código.

**Trazabilidad total:** Todos los datasets están versionados en el Data Catalog. Cada ejecución genera una traza completa en MLflow.

---

## 3. Descripción de Módulos

### 3.1 Pipeline: Ingestion

**Responsabilidad:** Extraer y validar datos brutos desde fuentes externas.

**Nodos:**

| Nodo | Función | Input | Output |
|---|---|---|---|
| `nodo_obtener_datos_mercado` | `obtener_datos_mercado()` | `params:ticker`, `params:start_date`, `params:end_date` | `raw_ohlcv` |
| `nodo_validar_datos_mercado` | `validar_datos_mercado()` | `raw_ohlcv` | `clean_ohlcv` |

**Validaciones aplicadas:**
- Verificación de columnas requeridas (open, high, low, close, volume)
- Eliminación de valores NaN
- Verificación de precios positivos
- Orden cronológico del índice

---

### 3.2 Pipeline: Feature Engineering

**Responsabilidad:** Transformar datos limpios en features cuantitativas interpretables por los agentes.

**Nodos planificados:**

| Nodo | Función | Output |
|---|---|---|
| `nodo_indicadores_tecnicos` | `calcular_indicadores_tecnicos()` | `technical_features` |
| `nodo_sentimiento` | `calcular_sentimiento()` | `sentiment_scores` |
| `nodo_vector_features` | `ensamblar_vector_features()` | `feature_vector` |

**Indicadores técnicos:**
- RSI (período configurable, default: 14)
- MACD (fast: 12, slow: 26, signal: 9)
- Bandas de Bollinger (período: 20, desviaciones: 2)
- ATR — Average True Range
- EMA 20 y EMA 50

---

### 3.3 Pipeline: LLM Agents

**Responsabilidad:** Orquestar agentes LLM especializados para producir una señal de trading razonada.

**Agentes:**

| Agente | Especialidad | Output |
|---|---|---|
| `technical_agent` | Interpreta indicadores técnicos | Análisis técnico en texto |
| `sentiment_agent` | Interpreta score de sentimiento | Análisis de sentimiento en texto |
| `risk_agent` | Evalúa condiciones de riesgo | Evaluación de riesgo en texto |
| `decision_agent` | Sintetiza todas las entradas | `trading_signal` (BUY/SELL/HOLD + confianza) |

**Esquema de salida del `decision_agent`:**

```json
{
  "signal": "BUY",
  "confidence": 0.78,
  "reasoning": "RSI en zona de sobreventa (28)...",
  "timestamp": "2024-01-15T10:30:00Z",
  "ticker": "BTC-USD"
}
```

---

### 3.4 Pipeline: Backtesting

**Responsabilidad:** Validar la estrategia sobre datos históricos antes de operar.

**Métricas calculadas:**

| Métrica | Descripción |
|---|---|
| Sharpe Ratio | Retorno ajustado por riesgo (objetivo: > 1.0) |
| Max Drawdown | Pérdida máxima desde un pico (objetivo: < 20%) |
| Win Rate | Porcentaje de operaciones ganadoras |
| Profit Factor | Ratio ganancias / pérdidas |
| CAGR | Tasa de crecimiento anual compuesto |

---

### 3.5 Pipeline: Execution

**Responsabilidad:** Ejecutar órdenes de trading con controles de riesgo.

**Controles de riesgo implementados:**
- Stop-loss configurable (default: 3%)
- Take-profit configurable (default: 6%)
- Máxima exposición por activo (default: 10% del portafolio)
- Umbral de confianza mínima para ejecutar (default: 0.65)

---

## 4. Modelo de Datos

### 4.1 Data Catalog — Capas

```
data/
├── 01_raw/          # Datos directos de APIs sin modificar
├── 02_intermediate/ # Datos limpios y validados
├── 03_primary/      # Features calculadas (indicadores, sentimiento)
├── 04_feature/      # Vector de features unificado
├── 07_model_output/ # Señales de trading generadas por agentes
└── 08_reporting/    # Métricas de backtesting y reportes
```

### 4.2 Esquema del Dataset Principal: `clean_ohlcv`

| Columna | Tipo | Descripción |
|---|---|---|
| `date` | DatetimeIndex | Fecha de la vela (índice) |
| `open` | float64 | Precio de apertura |
| `high` | float64 | Precio máximo |
| `low` | float64 | Precio mínimo |
| `close` | float64 | Precio de cierre |
| `volume` | float64 | Volumen operado |

### 4.3 Esquema del Dataset: `feature_vector`

| Columna | Tipo | Descripción |
|---|---|---|
| `close` | float64 | Precio de cierre |
| `rsi` | float64 | RSI (0-100) |
| `macd` | float64 | Línea MACD |
| `macd_signal` | float64 | Línea de señal MACD |
| `bb_upper` | float64 | Banda de Bollinger superior |
| `bb_lower` | float64 | Banda de Bollinger inferior |
| `ema_20` | float64 | Media móvil exponencial 20 |
| `ema_50` | float64 | Media móvil exponencial 50 |
| `sentiment_score` | float64 | Score de sentimiento [-1, +1] |

---

## 5. Configuración del Sistema

### 5.1 Parámetros Globales (`parameters.yml`)

```yaml
ticker: "BTC-USD"
timeframe: "1d"
start_date: "2022-01-01"
end_date: "2024-12-31"

technical:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20

risk:
  max_position_pct: 0.10
  stop_loss_pct: 0.03
  take_profit_pct: 0.06

llm:
  model: "gpt-4o-mini"
  temperature: 0.1
  confidence_threshold: 0.65

backtesting:
  initial_capital: 10000
  commission: 0.001
```

---

## 6. Decisiones de Diseño

### 6.1 ¿Por qué Kedro?

Kedro fuerza buenas prácticas desde el inicio: funciones puras, configuración centralizada y linaje de datos automático. Esto hace el código reproducible y mantenible desde el primer día.

### 6.2 ¿Por qué CrewAI sobre LangChain?

CrewAI tiene una API más simple para definir roles de agentes y manejar comunicación entre ellos. Es más adecuado para un equipo de agentes especializados con roles definidos.

### 6.3 ¿Por qué VectorBT sobre Backtrader?

VectorBT es significativamente más rápido (operaciones vectorizadas con NumPy) y tiene mejor integración con pandas DataFrames.

### 6.4 ¿Por qué paper trading primero?

El backtesting valida la estrategia históricamente, pero el paper trading valida la implementación técnica en tiempo real sin riesgo financiero. Es un paso obligatorio antes de capital real.

---

## 7. Limitaciones Conocidas

- El sistema actualmente soporta un solo activo por ejecución (multi-activo en roadmap)
- El análisis de sentimiento depende de disponibilidad de la NewsAPI (límite de requests en plan gratuito)
- Los agentes LLM pueden generar razonamientos inconsistentes en condiciones de mercado extremas
- No hay manejo de splits ni dividendos en los datos de acciones (sí en crypto)

---

## 8. Historial de Versiones

| Versión | Fecha | Cambios |
|---|---|---|
| v0.1.0 | Abril 2026 | Setup inicial, pipeline de ingesta implementado |
