# src/trading_agent/pipelines/alpaca/pipeline.py
"""Pipeline de ejecución via Alpaca Trading API.

Este pipeline se ejecuta independientemente del pipeline principal:
    kedro run --pipeline=alpaca

Requiere credenciales en conf/local/credentials.yml (ver ejemplo en
conf/base/credentials.yml.example).

Flujo:
  1. verificar_cuenta_alpaca → obtiene estado actual de la cuenta
  2. ejecutar_ordenes_alpaca → envía BUY orders aprobadas
  3. sincronizar_posiciones_alpaca → confirma posiciones abiertas
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    ejecutar_ordenes_alpaca,
    sincronizar_posiciones_alpaca,
    verificar_cuenta_alpaca,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=verificar_cuenta_alpaca,
                inputs=[],
                outputs="alpaca_account_state",
                name="nodo_verificar_cuenta_alpaca",
            ),
            node(
                func=ejecutar_ordenes_alpaca,
                inputs=[
                    "verified_signal",
                    "alpaca_account_state",
                    "parameters",
                ],
                outputs="alpaca_execution_log",
                name="nodo_ejecutar_ordenes_alpaca",
            ),
            node(
                func=sincronizar_posiciones_alpaca,
                inputs=[],
                outputs="alpaca_positions",
                name="nodo_sincronizar_posiciones_alpaca",
            ),
        ]
    )
