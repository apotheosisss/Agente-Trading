# Referencia de API — Nodos y Parámetros
## Agente de Trading con IA

---

## Convenciones

- Todos los nodos son **funciones Python puras** (sin efectos secundarios)
- Los inputs/outputs hacen referencia a entradas del `catalog.yml`
- Los parámetros se acceden como `params:nombre_parametro`
- Tipos de retorno siguen convenciones de pandas/Python estándar

---

## Pipeline: ingestion

### `obtener_datos_mercado()`

```python
def obtener_datos_mercado(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame
```

**Descripción:** Descarga datos OHLCV desde Yahoo Finance para el activo y rango de fechas especificado.

**Parámetros:**

| Parámetro | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `ticker` | `str` | Símbolo del activo en formato Yahoo Finance | `"BTC-USD"`, `"AAPL"` |
| `start_date` | `str` | Fecha de inicio en formato ISO 8601 | `"2022-01-01"` |
| `end_date` | `str` | Fecha de fin en formato ISO 8601 | `"2024-12-31"` |

**Retorna:** `pd.DataFrame` con columnas `open, high, low, close, volume` e índice `DatetimeIndex`.

**Excepciones:**
- `ValueError` si no se obtienen datos para el ticker especificado

---

### `validar_datos_mercado()`

```python
def validar_datos_mercado(
    df: pd.DataFrame
) -> pd.DataFrame
```

**Descripción:** Valida y limpia el DataFrame OHLCV. Elimina NaN, verifica precios positivos y asegura orden cronológico.

**Parámetros:**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `df` | `pd.DataFrame` | DataFrame OHLCV crudo desde `obtener_datos_mercado()` |

**Retorna:** `pd.DataFrame` validado y limpio.

**Excepciones:**
- `ValueError` si faltan columnas requeridas
- `ValueError` si se detectan precios negativos o cero

---

## Pipeline: feature_engineering

### `calcular_indicadores_tecnicos()`

```python
def calcular_indicadores_tecnicos(
    ohlcv: pd.DataFrame,
    parameters: dict
) -> pd.DataFrame
```

**Descripción:** Calcula indicadores técnicos sobre el DataFrame OHLCV limpio.

**Parámetros:**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `ohlcv` | `pd.DataFrame` | DataFrame limpio desde `validar_datos_mercado()` |
| `parameters` | `dict` | Sub-diccionario `technical` desde `parameters.yml` |

**Claves de `parameters`:**

| Clave | Tipo | Default | Descripción |
|---|---|---|---|
| `rsi_period` | `int` | `14` | Período del RSI |
| `macd_fast` | `int` | `12` | Período EMA rápida del MACD |
| `macd_slow` | `int` | `26` | Período EMA lenta del MACD |
| `macd_signal` | `int` | `9` | Período señal del MACD |
| `bb_period` | `int` | `20` | Período de Bandas de Bollinger |
| `bb_std` | `int` | `2` | Desviaciones estándar para BB |

**Retorna:** `pd.DataFrame` con columnas adicionales: `rsi, macd, macd_signal, bb_upper, bb_mid, bb_lower, ema_20, ema_50, atr`.

---

### `calcular_sentimiento()`

```python
def calcular_sentimiento(
    noticias: list[dict]
) -> pd.DataFrame
```

**Descripción:** Calcula el score de sentimiento de titulares de noticias usando FinBERT.

**Retorna:** `pd.DataFrame` con columnas `date, sentiment_score` donde el score está en el rango `[-1.0, +1.0]`.

---

### `ensamblar_vector_features()`

```python
def ensamblar_vector_features(
    technical: pd.DataFrame,
    sentiment: pd.DataFrame
) -> pd.DataFrame
```

**Descripción:** Combina features técnicas y de sentimiento en un único DataFrame alineado por fecha.

**Retorna:** `pd.DataFrame` unificado listo para consumo por los agentes LLM.

---

## Pipeline: llm_agents

### `decision_agent()`

```python
def decision_agent(
    tech_report: str,
    sent_report: str,
    risk_report: str,
    parameters: dict
) -> dict
```

**Descripción:** Sintetiza los reportes de los agentes especializados y genera la señal de trading final.

**Retorna:** Diccionario con el esquema:

```python
{
    "signal": str,        # "BUY" | "SELL" | "HOLD"
    "confidence": float,  # [0.0, 1.0]
    "reasoning": str,     # Explicación en lenguaje natural
    "timestamp": str,     # ISO 8601
    "ticker": str         # Activo analizado
}
```

---

## Pipeline: execution

### `verificar_riesgo()`

```python
def verificar_riesgo(
    signal: dict,
    portfolio: dict,
    parameters: dict
) -> bool
```

**Descripción:** Verifica si la señal supera los controles de riesgo antes de ejecutar.

**Retorna:** `True` si la señal puede ejecutarse, `False` si debe bloquearse.

**Controles aplicados:**
- `signal["confidence"] >= parameters["risk"]["confidence_threshold"]`
- Exposición actual del activo `< parameters["risk"]["max_position_pct"]`

---

## Parámetros Globales — Referencia Completa

Archivo: `conf/base/parameters.yml`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `ticker` | `str` | `"BTC-USD"` | Activo a operar |
| `timeframe` | `str` | `"1d"` | Granularidad temporal |
| `start_date` | `str` | `"2022-01-01"` | Inicio del período de datos |
| `end_date` | `str` | `"2024-12-31"` | Fin del período de datos |
| `technical.rsi_period` | `int` | `14` | Período RSI |
| `technical.macd_fast` | `int` | `12` | EMA rápida MACD |
| `technical.macd_slow` | `int` | `26` | EMA lenta MACD |
| `technical.macd_signal` | `int` | `9` | Señal MACD |
| `technical.bb_period` | `int` | `20` | Período Bollinger |
| `technical.bb_std` | `int` | `2` | Desviaciones estándar Bollinger |
| `risk.max_position_pct` | `float` | `0.10` | Máx. exposición por activo |
| `risk.stop_loss_pct` | `float` | `0.03` | Stop-loss (3%) |
| `risk.take_profit_pct` | `float` | `0.06` | Take-profit (6%) |
| `llm.model` | `str` | `"gpt-4o-mini"` | Modelo LLM a usar |
| `llm.temperature` | `float` | `0.1` | Temperatura del LLM |
| `llm.confidence_threshold` | `float` | `0.65` | Confianza mínima para ejecutar |
| `backtesting.initial_capital` | `int` | `10000` | Capital inicial en USD |
| `backtesting.commission` | `float` | `0.001` | Comisión por operación (0.1%) |

---

## Variables de Entorno y Credenciales

Archivo: `conf/local/credentials.yml` (no incluido en Git)

```yaml
openai:
  api_key: "sk-..."          # Requerida para GPT-4o-mini

alpaca:
  api_key: "..."             # Opcional — broker paper trading
  secret_key: "..."

newsapi:
  api_key: "..."             # Requerida para noticias financieras
```

**Alternativa sin costo:** Usar `ollama` localmente como backend LLM. Cambiar en `parameters.yml`:
```yaml
llm:
  model: "ollama/llama3"
```
