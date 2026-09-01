"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Pipelines excluidos del run completo por defecto (requieren configuracion especial)
_OPTIONAL_PIPELINES = {"alpaca", "alpaca_tsmom"}

# Pipelines que NO forman parte del pipeline de señales en vivo
_BACKTEST_PIPELINES = {"backtesting"}


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Pipelines de PRODUCCIÓN (usar estos):
    - tsmom_live   : ingestion + tsmom — calcula pesos objetivo, sin ejecutar ordenes
    - tsmom_trade  : ingestion + tsmom + alpaca_tsmom — calcula Y ejecuta en Alpaca
                     (esto es lo que corren los workflows de GitHub Actions)
    - alpaca_tsmom : solo ejecucion, con tsmom_weights ya calculado

    Pipelines LEGADO (congelados desde 2026-06, ver AUDITORIA_HALLAZGOS.md —
    conservados como referencia/comparacion, NO usar para operar):
    - __default__  : pipeline completo LLM (señales + backtesting)
    - signals      : señales via agentes LLM, sin backtesting
    - alpaca       : ejecucion long-only via el modelo LLM viejo

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)

    default_pipelines = {
        name: pipe
        for name, pipe in pipelines.items()
        if name not in _OPTIONAL_PIPELINES
    }
    pipelines["__default__"] = sum(default_pipelines.values())

    # Pipeline ligero para el scheduler diario (sin backtesting historico)
    signals_pipelines = {
        name: pipe
        for name, pipe in pipelines.items()
        if name not in _OPTIONAL_PIPELINES and name not in _BACKTEST_PIPELINES
    }
    pipelines["signals"] = sum(signals_pipelines.values())

    # ── PRODUCCIÓN: estrategia TSMOM long/short (ingesta → pesos → órdenes) ──
    # Pipeline autocontenido recomendado para operar. Solo necesita ingestión
    # (descarga del universo) + el módulo tsmom. No usa LLM/Polymarket/legacy.
    if "ingestion" in pipelines and "tsmom" in pipelines:
        pipelines["tsmom_live"] = pipelines["ingestion"] + pipelines["tsmom"]
        # tsmom_trade: calcula señales Y ejecuta en Alpaca (paper). Invocar a mano.
        if "alpaca_tsmom" in pipelines:
            pipelines["tsmom_trade"] = (
                pipelines["ingestion"] + pipelines["tsmom"] + pipelines["alpaca_tsmom"]
            )

    return pipelines
