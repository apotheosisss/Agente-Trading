# 🤖 Agente de Trading con IA

> Agente autónomo de inversión impulsado por modelos de lenguaje (LLM), con análisis técnico, análisis de sentimiento y backtesting. Arquitectura modular basada en Kedro.

---

## 📋 Descripción General

Sistema de trading algorítmico que utiliza múltiples agentes LLM especializados para analizar datos de mercado y generar señales de inversión (BUY / SELL / HOLD). El sistema integra indicadores técnicos cuantitativos con análisis de sentimiento de noticias financieras.

---

## 🏗️ Arquitectura

```
Fuentes externas (Yahoo Finance, NewsAPI)
        │
        ▼
[Pipeline: ingestion]          → data/01_raw/, data/02_intermediate/
        │
        ▼
[Pipeline: feature_engineering] → data/03_primary/, data/04_feature/
        │
        ▼
[Pipeline: llm_agents]          → data/07_model_output/
        │
        ▼
[Pipeline: backtesting]         → data/08_reporting/
        │
        ▼
[Pipeline: execution]           → Órdenes (paper trading)
```

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Framework de pipelines | Kedro 1.3.1 |
| Gestión de entorno | Astral UV |
| Datos de mercado | yfinance |
| Análisis técnico | pandas-ta |
| Orquestador LLM | CrewAI |
| Modelo LLM | GPT-4o-mini / Ollama |
| NLP / Sentimiento | HuggingFace Transformers (FinBERT) |
| Backtesting | VectorBT |
| Seguimiento de experimentos | MLflow |
| Visualización de pipelines | Kedro-Viz |
| Dashboard | Streamlit |
| Lenguaje | Python 3.12 |

---

## 📁 Estructura del Proyecto

```
trading-agent/
├── conf/
│   ├── base/
│   │   ├── catalog.yml         # Definición de datasets
│   │   ├── parameters.yml      # Parámetros globales
│   │   └── logging.yml         # Configuración de logs
│   └── local/
│       └── credentials.yml     # API keys (NO en Git)
├── data/
│   ├── 01_raw/                 # Datos crudos
│   ├── 02_intermediate/        # Datos limpios
│   ├── 03_primary/             # Features calculadas
│   ├── 04_feature/             # Vector de features unificado
│   ├── 07_model_output/        # Señales de trading
│   └── 08_reporting/           # Métricas y reportes
├── src/trading_agent/
│   └── pipelines/
│       ├── ingestion/
│       ├── feature_engineering/
│       ├── llm_agents/
│       ├── backtesting/
│       └── execution/
├── tests/
├── notebooks/
├── docs/
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Instalación desde Cero

### Prerrequisitos
- [UV (Astral)](https://astral.sh/uv) instalado
- Python 3.12+
- Git

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/apotheosisss/Agente-Trading.git
cd Agente-Trading

# 2. Crear entorno virtual
uv venv --python 3.12

# 3. Activar entorno
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\Activate.ps1       # Windows

# 4. Instalar dependencias
uv pip install -r requirements.txt

# 5. Configurar credenciales
cp conf/local/credentials.yml.example conf/local/credentials.yml
# Editar credentials.yml con tus API keys
```

---

## ⚙️ Configuración

Editar `conf/base/parameters.yml`:

```yaml
ticker: "BTC-USD"       # Activo a operar
start_date: "2022-01-01"
end_date: "2024-12-31"
```

Editar `conf/local/credentials.yml` (no se sube a Git):

```yaml
openai:
  api_key: "sk-..."
newsapi:
  api_key: "..."
```

---

## ▶️ Ejecución

```bash
# Ejecutar pipeline completo
kedro run

# Ejecutar pipeline específico
kedro run --pipeline ingestion
kedro run --pipeline feature_engineering
kedro run --pipeline llm_agents
kedro run --pipeline backtesting

# Visualizar pipelines
kedro viz run
# Abrir http://localhost:4141

# Correr tests
pytest tests/
```

---

## 🗺️ Roadmap

| Milestone | Estado | Descripción |
|---|---|---|
| M1 — Ingesta y Features | 🔄 En progreso | Pipeline de datos y análisis técnico |
| M2 — Agentes LLM | ⏳ Pendiente | Orquestador CrewAI con agentes especializados |
| M3 — Backtesting | ⏳ Pendiente | Validación histórica con VectorBT |
| M4 — Ejecución | ⏳ Pendiente | Paper trading y gestión de riesgo |
| M5 — Dashboard | ⏳ Pendiente | Interfaz Streamlit con monitoreo en tiempo real |

---

## 📐 Documentación Técnica

Ver carpeta [`docs/`](docs/):

- [`docs/arquitectura.md`](docs/arquitectura.md) — Diseño del sistema
- [`docs/diagramas/`](docs/diagramas/) — Diagramas PlantUML
- [`docs/sdd.md`](docs/sdd.md) — Documento de Diseño de Software
- [`docs/api.md`](docs/api.md) — Referencia de nodos y parámetros

---

## 🔒 Reglas de Seguridad

1. **Nunca** subir `conf/local/credentials.yml` a Git
2. **Nunca** operar con dinero real antes de completar el Milestone 3 (backtesting)
3. Todo nuevo módulo debe tener tests en `/tests/`
4. Usar `loguru` en lugar de `print()`
5. Commits frecuentes con mensajes descriptivos en español

---

## 👤 Autor

**Claudio** — Estudiante de Ingeniería Informática mención Ciencia de Datos, DuocUC  
Proyecto académico — v0.1.0 (en desarrollo activo)