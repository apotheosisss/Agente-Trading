# src/trading_agent/pipelines/ingestion/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import obtener_datos_mercado, validar_datos_mercado


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=obtener_datos_mercado,
                inputs=[
                    "params:universe",
                    "params:start_date",
                    "params:end_date",
                ],
                outputs="raw_ohlcv",
                name="nodo_obtener_datos_mercado",
            ),
            node(
                func=validar_datos_mercado,
                inputs="raw_ohlcv",
                outputs="clean_ohlcv",
                name="nodo_validar_datos_mercado",
            ),
        ]
    )