# src/trading_agent/pipelines/backtesting/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import (
    calcular_benchmark,
    calcular_metricas,
    calcular_walk_forward,
    ejecutar_backtest,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ejecutar_backtest,
                inputs=["feature_vector", "vix_data", "parameters"],
                outputs="backtest_portfolio",
                name="nodo_ejecutar_backtest",
            ),
            node(
                func=calcular_metricas,
                inputs=["backtest_portfolio", "parameters"],
                outputs=["backtest_metrics", "equity_curve"],
                name="nodo_calcular_metricas",
            ),
            node(
                func=calcular_benchmark,
                inputs=["feature_vector", "parameters"],
                outputs="benchmark_curve",
                name="nodo_calcular_benchmark",
            ),
            node(
                func=calcular_walk_forward,
                inputs=["backtest_portfolio", "parameters"],
                outputs="walk_forward_results",
                name="nodo_walk_forward",
            ),
        ]
    )
