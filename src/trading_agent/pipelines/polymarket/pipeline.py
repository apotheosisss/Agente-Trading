# src/trading_agent/pipelines/polymarket/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import generar_reporte_polymarket, obtener_seniales_polymarket


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline de seniales Polymarket.

    Outputs:
        polymarket_signals — DataFrame con poly_score por ticker.
        poly_report        — Reporte texto para agente_decision.

    Nota: este pipeline corre ANTES de llm_agents porque
    agente_decision consume poly_report como input adicional.
    """
    return Pipeline(
        [
            node(
                func=obtener_seniales_polymarket,
                inputs="parameters",
                outputs="polymarket_signals",
                name="nodo_polymarket_signals",
            ),
            node(
                func=generar_reporte_polymarket,
                inputs="polymarket_signals",
                outputs="poly_report",
                name="nodo_poly_report",
            ),
        ]
    )
