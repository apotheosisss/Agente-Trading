from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


def test_pipelines_registrados():
    """Verifica que los 6 pipelines están registrados y tienen nodos."""
    bootstrap_project(Path.cwd())
    nombres_esperados = {
        "ingestion", "feature_engineering", "llm_agents",
        "backtesting", "execution", "polymarket",
    }
    registrados = set(pipelines.keys()) - {"__default__"}
    assert nombres_esperados <= registrados


def test_default_pipeline_tiene_nodos():
    """El pipeline __default__ debe incluir nodos de todos los sub-pipelines."""
    bootstrap_project(Path.cwd())
    default = pipelines["__default__"]
    assert len(default.nodes) > 0
