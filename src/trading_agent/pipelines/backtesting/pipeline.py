# src/trading_agent/pipelines/backtesting/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import calcular_metricas, ejecutar_backtest


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ejecutar_backtest,
                inputs=["feature_vector", "parameters"],
                outputs="backtest_portfolio",
                name="nodo_ejecutar_backtest",
            ),
            node(
                func=calcular_metricas,
                inputs=["backtest_portfolio", "parameters"],
                outputs=["backtest_metrics", "equity_curve"],
                name="nodo_calcular_metricas",
            ),
        ]
    )
