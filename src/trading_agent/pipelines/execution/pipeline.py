# src/trading_agent/pipelines/execution/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import actualizar_portafolio, enviar_orden, verificar_riesgo


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=verificar_riesgo,
                inputs=["trading_signal", "parameters"],
                outputs="risk_result",
                name="nodo_verificar_riesgo",
            ),
            node(
                func=enviar_orden,
                inputs=["risk_result", "trading_signal", "parameters"],
                outputs="execution_record",
                name="nodo_enviar_orden",
            ),
            node(
                func=actualizar_portafolio,
                inputs=["execution_record", "parameters"],
                outputs="portfolio_state",
                name="nodo_actualizar_portafolio",
            ),
        ]
    )
