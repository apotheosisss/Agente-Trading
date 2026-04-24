# src/trading_agent/pipelines/feature_engineering/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import calcular_indicadores_tecnicos, calcular_sentimiento, ensamblar_vector_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=calcular_indicadores_tecnicos,
                inputs=["clean_ohlcv", "params:technical"],
                outputs="technical_features",
                name="nodo_indicadores_tecnicos",
            ),
            node(
                func=calcular_sentimiento,
                inputs="clean_ohlcv",
                outputs="sentiment_scores",
                name="nodo_sentimiento",
            ),
            node(
                func=ensamblar_vector_features,
                inputs=["technical_features", "sentiment_scores"],
                outputs="feature_vector",
                name="nodo_vector_features",
            ),
        ]
    )
