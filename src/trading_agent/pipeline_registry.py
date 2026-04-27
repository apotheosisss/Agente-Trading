"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Pipelines excluidos del run completo por defecto (requieren configuracion especial)
_OPTIONAL_PIPELINES = {"alpaca"}


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    El pipeline ``alpaca`` requiere credenciales en conf/local/credentials.yml
    y se ejecuta por separado:  kedro run --pipeline=alpaca

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
    return pipelines
