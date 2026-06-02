# src/trading_agent/pipelines/tsmom/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import (
    calcular_pesos_tsmom,
    generar_ordenes_tsmom,
    generar_reporte_tsmom,
    validar_tsmom,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline TSMOM long/short (estrategia de producción validada).

    Outputs:
        tsmom_weights  — pesos objetivo (con signo) por activo, hoy.
        tsmom_report   — reporte legible.
        tsmom_orders   — órdenes/posiciones objetivo para ejecución.
        tsmom_validation — métricas de backtest (full + OOS) para monitoreo.
    """
    return Pipeline([
        node(
            func=calcular_pesos_tsmom,
            inputs=["clean_ohlcv", "parameters"],
            outputs="tsmom_weights",
            name="nodo_tsmom_pesos",
        ),
        node(
            func=generar_reporte_tsmom,
            inputs="tsmom_weights",
            outputs="tsmom_report",
            name="nodo_tsmom_reporte",
        ),
        node(
            func=generar_ordenes_tsmom,
            inputs=["tsmom_weights", "parameters"],
            outputs="tsmom_orders",
            name="nodo_tsmom_ordenes",
        ),
        node(
            func=validar_tsmom,
            inputs=["clean_ohlcv", "parameters"],
            outputs="tsmom_validation",
            name="nodo_tsmom_validacion",
        ),
    ])
