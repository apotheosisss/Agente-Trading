"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Pipelines excluidos del run completo por defecto (requieren configuracion especial)
_OPTIONAL_PIPELINES = {"alpaca"}

# Pipelines que NO forman parte del pipeline de señales en vivo
_BACKTEST_PIPELINES = {"backtesting"}


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Pipelines disponibles:
    - __default__  : pipeline completo (señales + backtesting)
    - signals      : solo señales en vivo, sin backtesting (~30s)
    - alpaca       : ejecucion real via Alpaca (requiere credenciales)

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

    return pipelines
