# src/trading_agent/pipelines/alpaca_tsmom/pipeline.py
"""Ejecución de la estrategia TSMOM long/short en Alpaca (paper).

Consume `tsmom_weights` (pesos objetivo) y lleva la cuenta a esa posición
mediante órdenes delta. Se ejecuta SOLO cuando se invoca explícitamente:

    kedro run --pipeline tsmom_trade   (ingestión + tsmom + esta ejecución)
    kedro run --pipeline alpaca_tsmom  (solo ejecución, con tsmom_weights ya calculado)

Requiere credenciales en conf/local/credentials.yml (alpaca.paper_trading: true).
"""
from kedro.pipeline import Pipeline, node

from trading_agent.pipelines.alpaca.nodes import (
    ejecutar_tsmom_alpaca,
    sincronizar_posiciones_alpaca,
    verificar_cuenta_alpaca,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=verificar_cuenta_alpaca,
            inputs=[],
            outputs="alpaca_account_state",
            name="nodo_tsmom_verificar_cuenta",
        ),
        node(
            func=ejecutar_tsmom_alpaca,
            inputs=["tsmom_weights", "alpaca_account_state", "parameters"],
            outputs="tsmom_alpaca_log",
            name="nodo_tsmom_ejecutar_alpaca",
        ),
        node(
            func=sincronizar_posiciones_alpaca,
            inputs=[],
            outputs="alpaca_positions",
            name="nodo_tsmom_sincronizar_posiciones",
        ),
    ])
