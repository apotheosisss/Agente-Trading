# TradingAgents + OpenAI Integration

## Configuración realizada

He configurado la integración de TradingAgents con OpenAI `gpt-4o-mini`. El sistema ahora está listo para usar un **filtro de confirmación selectiva** que analiza los top BUY signals a través de un multi-agente framework de análisis.

### Cambios realizados:

#### 1. **Configuración de TradingAgents → OpenAI** 
   - Archivo: `C:\Users\molte\Documents\Proyectos\TradingAgents-0.2.4\tradingagents\default_config.py`
   - Cambios:
     ```python
     "llm_provider": "openai"  # (confirmado)
     "deep_think_llm": "gpt-4o-mini"  # (modelo económico)
     "quick_think_llm": "gpt-4o-mini"  # (mismo modelo para ambos)
     ```
   - **Modelo seleccionado:**
     - `gpt-4o-mini`: Económico, rápido y muy competente para análisis multi-agente

#### 2. **Nuevo nodo Kedro: `filtrar_signals_tradingagents`**
   - Archivo: `src/trading_agent/pipelines/llm_agents/nodes.py`
   - Flujo:
     ```
     agente_decision → trading_signal
                   ↓
     filtrar_signals_tradingagents → verified_signal
     ```
   - Lógica:
     1. Toma top N BUY signals con confidence ≥ threshold
     2. Corre cada ticker a través de TradingAgents
     3. Si TradingAgents devuelve "Buy" u "Overweight" → mantiene la señal
     4. Si devuelve "Hold"/"Underweight"/"Sell" → descarta la señal
     5. Retorna solo señales CONFIRMADAS por ambos sistemas

#### 3. **Configuración de parámetros**
   - Archivo: `conf/base/parameters.yml`
   - Nueva sección `llm.tradingagents`:
     ```yaml
     llm:
       tradingagents:
         enabled: false              # CAMBIAR A true PARA ACTIVAR
         confidence_threshold: 0.60  # Solo analiza BUY con confidence ≥ 60%
         top_n_signals: 3            # Máximo 3 signals por corrida
     ```

#### 4. **Credenciales OpenAI**
   - Archivo: `conf/local/credentials.yml`
   - Campo nuevo:
     ```yaml
     openai:
       api_key: ""  # ← COMPLETAR CON TU CLAVE
     ```

---

## Próximos pasos (para activar)

### Paso 1: Obtener API key de OpenAI

1. Ve a https://platform.openai.com/api-keys
2. Crea una clave (si no tienes) o copia una existente (formato: `sk-...`)
3. Edita `conf/local/credentials.yml` y pega la clave:
   ```yaml
   openai:
     api_key: "sk-tu-clave-aqui"
   ```

**Nota de seguridad:** 
- Este archivo está en `.gitignore` (no se commitea)
- No compartas tu API key

### Paso 2: Verificar configuración (opcional)

Ejecuta el script de prueba:
```bash
cd C:\Users\molte\Documents\Proyectos\trading-agent\.claude\worktrees\polymarket-work
uv run python test_tradingagents.py
```

El test verifica:
- ✓ TradingAgents instalado
- ✓ LangChain OpenAI disponible
- ✓ Config de DEFAULT_CONFIG = OpenAI (gpt-4o-mini)
- ✓ API key cargada
- ✓ TradingAgentsGraph inicializado
- ✓ Un análisis de prueba (AAPL)

### Paso 3: Activar el filtro

Edita `conf/base/parameters.yml`:
```yaml
llm:
  tradingagents:
    enabled: true  # ← cambiar de false a true
```

### Paso 4: Ejecutar pipeline

```bash
# Con el nuevo filtro activado
uv run kedro run --pipeline=signals
```

---

## Flujo de ejecución con TradingAgents

```
[Datos históricos de yfinance]
        ↓
[Pipeline de features: RSI, MACD, Bollinger, EMA]
        ↓
[agente_tecnico, agente_sentimiento, agente_riesgo]
        ↓
[agente_decision] → trading_signal (BUY/HOLD/SELL)
        ↓
[filtrar_signals_tradingagents]
    ├─ Top 3 BUY signals (confidence ≥ 60%)
    ├─ Market Analyst: análisis técnico con yfinance
    ├─ News Analyst: análisis de sentimiento de noticias
    ├─ Fundamentals Analyst: reportes financieros
    ├─ Investment Debate: Bull vs Bear con judge
    ├─ Risk Debate: Aggressive vs Conservative
    └─ Portfolio Manager: Rating final (Buy/Overweight/Hold/Underweight/Sell)
        ↓
[verified_signal] → Solo BUY confirmados por AMBOS sistemas
        ↓
[Ejecución Alpaca] (en rama crypto)
```

---

## Arquitectura de TradingAgents

### Agentes involucrados:

| Agente | Rol | Entrada | Salida |
|--------|-----|---------|--------|
| **Market Analyst** | Análisis técnico (RSI, MACD, Bollinger, EMA, ATR) | Ticker + datos históricos | Market Report |
| **News Analyst** | Sentimiento de noticias financieras | Ticker + query Google News | Sentiment Report |
| **Fundamentals Analyst** | Análisis de ingresos, ganancias, ratios | Ticker + financials de yfinance | Fundamentals Report |
| **Bull Debater** | Argumento alcista | Todos los reports | Bull reasoning |
| **Bear Debater** | Argumento bajista | Todos los reports | Bear reasoning |
| **Investment Judge** | Resuelve debate | Bull + Bear | Investment decision |
| **Risk Debater (Aggressive)** | Estrategia agresiva | Investment decision | Risk reasoning |
| **Risk Debater (Conservative)** | Estrategia conservadora | Investment decision | Risk reasoning |
| **Risk Judge** | Resuelve debate de riesgo | Aggressive + Conservative | Final risk stance |
| **Portfolio Manager** | Rating final (5 tiers) | Final risk stance | Signal: Buy/Overweight/Hold/Underweight/Sell |

### Configuración para Pro Anthropic:

El sistema ya está configurado para usar:
- **Deep thinking**: `claude-opus-4-6` (análisis complejos multi-step)
- **Quick thinking**: `claude-sonnet-4-6` (respuestas rápidas)

Esto es óptimo porque:
- Opus es el modelo más inteligente (análisis sofisticados del debate de inversión)
- Sonnet es rápido y eficiente (análisis técnicos, reportes)
- Ambos soportan herramientas y reasoning

---

## Costo estimado

Con 3 signals por día, 5 días/semana (20 días/mes):

- **Tokens por análisis:** ~8,000 input + ~2,000 output
- **Costo por análisis:** 8k × $0.00015 + 2k × $0.0006 = ~$0.0036
- **Costo mensual (60 análisis):** ~$0.22/mes

Muy económico con `gpt-4o-mini`.

---

## Archivos modificados

### 1. TradingAgents config (sistema global)
```
C:\Users\molte\Documents\Proyectos\TradingAgents-0.2.4\
  tradingagents\default_config.py
```

### 2. Polymarket branch
```
.claude\worktrees\polymarket-work\
  ├─ src\trading_agent\pipelines\llm_agents\
  │  ├─ nodes.py (nuevo nodo: filtrar_signals_tradingagents)
  │  └─ pipeline.py (añadido el nodo al pipeline)
  ├─ conf\base\parameters.yml (nuevo: llm.tradingagents)
  ├─ conf\local\credentials.yml (nuevo: anthropic.api_key)
  └─ test_tradingagents.py (script de prueba)
```

---

## Troubleshooting

### Error: "OPENAI_API_KEY not found"
→ Asegúrate de que editaste `conf/local/credentials.yml` con tu clave real

### Error: "TradingAgents graph compilation timeout"
→ El análisis puede tomar 1-2 minutos por ticker (normal con multi-agent debate)
→ Reduce `top_n_signals: 1` si es muy lento

### Error: "gpt-4o-mini model not found"
→ Verifica que tu cuenta OpenAI tiene acceso a gpt-4o-mini
→ Si no, usa `gpt-4o` en default_config.py (más caro pero más potente)

### Error: "LangChain OpenAI import failed"
→ Instala: `uv add langchain-openai`

---

## Para probar primero (recomendado)

1. Activa el filtro SOLO en modo dry-run:
   ```bash
   uv run python -m streamlit run dashboard.py
   # Verifica que trading_signal se ve correctamente
   ```

2. Ejecuta con un solo ticker para minimizar costo:
   ```bash
   # En parameters.yml, reduce universe: a solo ["AAPL"]
   uv run kedro run --pipeline=signals
   ```

3. Observa los logs para ver decisiones de TradingAgents:
   ```
   [INFO] TradingAgents: analizando AAPL
   [INFO] ✓ AAPL confirmado por TradingAgents (Buy)
   [INFO] TradingAgents: 1/1 señales confirmadas
   ```

4. Una vez confiado, activa completamente:
   - Restaura `universe:` a todos los tickers
   - Usa con `parameters.yml` normal
   - El scheduler diario lo ejecutará automáticamente

---

## Referencias

- TradingAgents: https://github.com/LanguageGroupAI/TradingAgents
- Anthropic Models: https://docs.anthropic.com/en/docs/about/models/overview
- LangChain Anthropic: https://python.langchain.com/docs/integrations/llms/anthropic/

