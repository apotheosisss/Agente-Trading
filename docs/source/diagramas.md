# Diagramas del Sistema — Agente de Trading con IA
> Todos los diagramas están en formato PlantUML. Renderizar en https://www.plantuml.com/plantuml

---

## 1. Diagrama de Arquitectura General

```plantuml
@startuml arquitectura_general
!theme plain
skinparam backgroundColor #FAFAFA
skinparam componentStyle rectangle
skinparam defaultFontName Arial

title Arquitectura General — Agente de Trading con IA

package "Fuentes Externas" {
  [Yahoo Finance\nyfinance] as YF
  [NewsAPI\nRSS Feeds] as NEWS
}

package "Pipeline: Ingestion" #E8F4FD {
  [obtener_datos_mercado()] as ODM
  [validar_datos_mercado()] as VDM
  database "01_raw\nohlcv.parquet" as RAW
  database "02_intermediate\nclean_ohlcv.parquet" as CLEAN
}

package "Pipeline: Feature Engineering" #E8FDE8 {
  [calcular_indicadores_tecnicos()] as CIT
  [calcular_sentimiento()] as CS
  [ensamblar_vector_features()] as EVF
  database "03_primary\nfeatures" as PRIMARY
  database "04_feature\nfeature_vector" as FEATURE
}

package "Pipeline: LLM Agents" #FDF3E8 {
  [technical_agent] as TA
  [sentiment_agent] as SA
  [risk_agent] as RA
  [decision_agent] as DA
  database "07_model_output\nsignal.json" as SIGNAL
}

package "Pipeline: Backtesting" #F3E8FD {
  [ejecutar_backtest()] as EB
  [calcular_metricas()] as CM
  database "08_reporting\nmetricas.csv" as REPORT
}

package "Pipeline: Execution" #FDE8E8 {
  [verificar_riesgo()] as VR
  [enviar_orden()] as EO
  [actualizar_portafolio()] as AP
}

YF --> ODM
NEWS --> CS

ODM --> RAW
RAW --> VDM
VDM --> CLEAN

CLEAN --> CIT
CIT --> PRIMARY
NEWS --> CS
CS --> PRIMARY
PRIMARY --> EVF
EVF --> FEATURE

FEATURE --> TA
FEATURE --> SA
FEATURE --> RA
TA --> DA
SA --> DA
RA --> DA
DA --> SIGNAL

SIGNAL --> EB
EB --> CM
CM --> REPORT

SIGNAL --> VR
VR --> EO
EO --> AP

@enduml
```

---

## 2. Diagrama de Flujo de Datos

```plantuml
@startuml flujo_datos
!theme plain
skinparam backgroundColor #FAFAFA
skinparam defaultFontName Arial

title Flujo de Datos — De la API a la Decisión Final

|Capa 1: Ingestion|
start
:Descargar OHLCV\n(yfinance);
:Validar integridad\n(NaN, outliers, orden);
:Guardar raw_ohlcv.parquet\ny clean_ohlcv.parquet;

|Capa 2: Feature Engineering|
:Calcular RSI, MACD,\nBollinger Bands, ATR, EMA;
:Analizar sentimiento\nde noticias (FinBERT);
:Ensamblar vector\nde features unificado;

|Capa 3: LLM Agents|
:Agente Técnico\nanaliza indicadores;
:Agente Sentimiento\nanaliza noticias;
:Agente Riesgo\nevalúa condiciones;
:Agente Decisión\nsintentiza señal;

|Capa 4: Backtesting|
:Simular estrategia\nen datos históricos;
:Calcular Sharpe,\nDrawdown, Win Rate;

|Capa 5: Execution|
:Verificar controles\nde riesgo;
if (¿Supera umbral\nde confianza?) then (sí)
  :Enviar orden\n(paper trading);
  :Actualizar portafolio;
else (no)
  :Mantener posición\n(HOLD);
endif
stop

@enduml
```

---

## 3. Diagrama de Componentes — Pipeline de Agentes LLM

```plantuml
@startuml componentes_llm
!theme plain
skinparam backgroundColor #FAFAFA
skinparam defaultFontName Arial
skinparam componentStyle rectangle

title Pipeline LLM Agents — Componentes y Comunicación

component "feature_vector\n(DataFrame)" as FV #E8F4FD

package "Orquestador CrewAI" {
  component "technical_agent" as TA {
    [Prompt: analista\ntécnico cuantitativo]
  }
  component "sentiment_agent" as SA {
    [Prompt: analista\nde sentimiento]
  }
  component "risk_agent" as RA {
    [Prompt: gestor\nde riesgo]
  }
  component "decision_agent" as DA {
    [Prompt: director\nde inversiones]
  }
}

component "GPT-4o-mini\n/ Ollama" as LLM #FDF3E8

component "trading_signal\n(JSON)" as SIG #E8FDE8

FV --> TA : "indicadores\ntécnicos"
FV --> SA : "score de\nsentimiento"
FV --> RA : "volatilidad\ny riesgo"

TA --> LLM : "consulta"
SA --> LLM : "consulta"
RA --> LLM : "consulta"

LLM --> TA : "análisis técnico"
LLM --> SA : "análisis sentimiento"
LLM --> RA : "evaluación riesgo"

TA --> DA : "reporte técnico"
SA --> DA : "reporte sentimiento"
RA --> DA : "reporte riesgo"

DA --> LLM : "síntesis final"
LLM --> DA : "decisión razonada"
DA --> SIG : "BUY/SELL/HOLD\n+ confianza"

@enduml
```

---

## 4. Diagrama de Secuencia — Ciclo Completo de Trading

```plantuml
@startuml secuencia_trading
!theme plain
skinparam backgroundColor #FAFAFA
skinparam defaultFontName Arial

title Secuencia Completa — Un Ciclo de Trading

actor "Scheduler" as SCH
participant "Pipeline\nIngestion" as ING
participant "Pipeline\nFeature Eng." as FE
participant "Pipeline\nLLM Agents" as LLM
participant "Pipeline\nExecution" as EXE
database "Data Catalog\n(Kedro)" as DC
participant "Broker API\n(Paper)" as BRK

SCH -> ING: kedro run
activate ING

ING -> ING: yf.download(ticker)
ING -> ING: validar_datos()
ING -> DC: guardar raw_ohlcv\ny clean_ohlcv
deactivate ING

ING -> FE: trigger automático
activate FE

FE -> DC: cargar clean_ohlcv
FE -> FE: calcular RSI, MACD, BB
FE -> FE: score sentimiento (FinBERT)
FE -> DC: guardar feature_vector
deactivate FE

FE -> LLM: trigger automático
activate LLM

LLM -> DC: cargar feature_vector
LLM -> LLM: technical_agent()
LLM -> LLM: sentiment_agent()
LLM -> LLM: risk_agent()
LLM -> LLM: decision_agent() → BUY
LLM -> DC: guardar signal.json
deactivate LLM

LLM -> EXE: trigger automático
activate EXE

EXE -> DC: cargar signal.json
EXE -> EXE: verificar_riesgo()\nconfianza >= 0.65?
EXE -> EXE: calcular tamaño\nde posición
EXE -> BRK: enviar_orden(BUY, qty)
BRK --> EXE: orden confirmada
EXE -> DC: actualizar portafolio
deactivate EXE

@enduml
```

---

## 5. Diagrama de Clases — Estructura de Nodos

```plantuml
@startuml clases_nodos
!theme plain
skinparam backgroundColor #FAFAFA
skinparam defaultFontName Arial

title Estructura de Nodos por Pipeline

package "ingestion.nodes" {
  class obtener_datos_mercado {
    +ticker: str
    +start_date: str
    +end_date: str
    --
    +return: pd.DataFrame
  }
  class validar_datos_mercado {
    +df: pd.DataFrame
    --
    +return: pd.DataFrame
    -verificar_columnas()
    -eliminar_nan()
    -validar_precios()
    -ordenar_indice()
  }
}

package "feature_engineering.nodes" {
  class calcular_indicadores_tecnicos {
    +ohlcv: pd.DataFrame
    +parameters: dict
    --
    +return: pd.DataFrame
    -calc_rsi()
    -calc_macd()
    -calc_bollinger()
    -calc_ema()
  }
  class calcular_sentimiento {
    +noticias: list
    --
    +return: pd.DataFrame
    -tokenizar()
    -inferir_finbert()
    -agregar_scores()
  }
  class ensamblar_vector_features {
    +technical: pd.DataFrame
    +sentiment: pd.DataFrame
    --
    +return: pd.DataFrame
  }
}

package "llm_agents.nodes" {
  class technical_agent {
    +feature_vector: pd.DataFrame
    +parameters: dict
    --
    +return: str
  }
  class sentiment_agent {
    +feature_vector: pd.DataFrame
    +parameters: dict
    --
    +return: str
  }
  class risk_agent {
    +feature_vector: pd.DataFrame
    +parameters: dict
    --
    +return: str
  }
  class decision_agent {
    +tech_report: str
    +sent_report: str
    +risk_report: str
    --
    +return: dict
  }
}

package "execution.nodes" {
  class verificar_riesgo {
    +signal: dict
    +portfolio: dict
    +parameters: dict
    --
    +return: bool
  }
  class enviar_orden {
    +signal: dict
    +qty: float
    --
    +return: dict
  }
}

obtener_datos_mercado --> validar_datos_mercado
validar_datos_mercado --> calcular_indicadores_tecnicos
calcular_indicadores_tecnicos --> ensamblar_vector_features
calcular_sentimiento --> ensamblar_vector_features
ensamblar_vector_features --> technical_agent
ensamblar_vector_features --> sentiment_agent
ensamblar_vector_features --> risk_agent
technical_agent --> decision_agent
sentiment_agent --> decision_agent
risk_agent --> decision_agent
decision_agent --> verificar_riesgo
verificar_riesgo --> enviar_orden

@enduml
```

---

## 6. Diagrama de Despliegue

```plantuml
@startuml despliegue
!theme plain
skinparam backgroundColor #FAFAFA
skinparam defaultFontName Arial

title Diagrama de Despliegue — Entorno de Desarrollo

node "Máquina Local\n(Windows 11)" {
  package "Entorno Virtual (.venv)" {
    component "Kedro 1.3.1" as KEDRO
    component "CrewAI" as CREW
    component "VectorBT" as VBT
    component "Streamlit\nDashboard" as DASH
    component "MLflow\nTracking" as MLF
  }

  folder "Proyecto" {
    folder "conf/base" as CONF
    folder "data/" as DATA
    folder "src/" as SRC
  }

  component "Kedro-Viz\n:4141" as VIZ
  component "MLflow UI\n:5000" as MLUI
  component "Streamlit\n:8501" as STR
}

cloud "APIs Externas" {
  [Yahoo Finance\nAPI] as YF
  [OpenAI\nAPI] as OAI
  [NewsAPI] as NAPI
}

node "GitHub" {
  [Repositorio\nAgente-Trading] as GH
}

KEDRO --> CONF : "lee parámetros"
KEDRO --> DATA : "lee/escribe datasets"
KEDRO --> SRC : "ejecuta nodos"
KEDRO --> VIZ : "visualización"
KEDRO --> MLF : "tracking"
MLF --> MLUI : "interfaz web"
DASH --> STR : "servidor"

SRC --> YF : "HTTPS"
SRC --> OAI : "HTTPS + API Key"
SRC --> NAPI : "HTTPS + API Key"

SRC --> GH : "git push"

@enduml
```

---

## Cómo Renderizar

**Opción 1 — Online:**
1. Copiar cualquier bloque entre `@startuml` y `@enduml`
2. Pegar en https://www.plantuml.com/plantuml/uml/

**Opción 2 — VS Code:**
```bash
# Instalar extensión: "PlantUML" de jebbs
# Instalar Java (requerido por PlantUML)
# Alt+D para previsualizar
```

**Opción 3 — CLI con UV:**
```bash
uv pip install plantuml
python -m plantuml docs/diagramas/diagramas.md
```
