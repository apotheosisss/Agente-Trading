# src/trading_agent/pipelines/llm_agents/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import agente_decision, agente_riesgo, agente_sentimiento, agente_tecnico


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=agente_tecnico,
                inputs="feature_vector",
                outputs="tech_report",
                name="nodo_agente_tecnico",
            ),
            node(
                func=agente_sentimiento,
                inputs="feature_vector",
                outputs="sent_report",
                name="nodo_agente_sentimiento",
            ),
            node(
                func=agente_riesgo,
                inputs="feature_vector",
                outputs="risk_report",
                name="nodo_agente_riesgo",
            ),
            node(
                func=agente_decision,
                inputs=[
                    "tech_report",
                    "sent_report",
                    "risk_report",
                    "feature_vector",
                    "params:llm",
                    "params:universe",
                    "poly_report",
                    "polymarket_signals",
                ],
                outputs="trading_signal",
                name="nodo_agente_decision",
            ),
        ]
    )
