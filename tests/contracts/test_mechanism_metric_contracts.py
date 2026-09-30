from __future__ import annotations

import json
from pathlib import Path

from dimebench.config import load_yaml
from dimebench.evaluators.mechanism import available_mechanism_metrics
from dimebench.schemas.task import TaskSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mechanism_registry_covers_protocol_and_task_configs() -> None:
    protocol = json.loads(
        (PROJECT_ROOT / "specification/benchmark-v1.json").read_text()
    )
    protocol_metrics = {
        metric
        for track in protocol["tracks"]
        for metric in track["metric_ids"]
        if metric in available_mechanism_metrics()
    }
    configured = set()
    for path in sorted((PROJECT_ROOT / "configs/tasks").glob("*/*.yaml")):
        task = TaskSpec.model_validate(load_yaml(path))
        configured.update(
            metric for metric in task.metrics if metric in available_mechanism_metrics()
        )
    assert protocol_metrics == set(available_mechanism_metrics())
    assert configured == protocol_metrics - {"reasoning_regime_gap"}


def test_neural_mechanism_modules_import_without_loading_optional_dependencies() -> (
    None
):
    from dimebench.evaluators.mechanism import (
        BoundaryConsistencyEvaluator,
        ContradictionRateEvaluator,
        ContradictionReductionEvaluator,
        TransformersNLIBackend,
        TransformersPairClassifier,
    )

    assert BoundaryConsistencyEvaluator.METRIC_ID == "boundary_consistency"
    assert ContradictionRateEvaluator.METRIC_ID == "contradiction_rate"
    assert ContradictionReductionEvaluator.METRIC_ID == "contradiction_reduction_rate"
    assert TransformersNLIBackend.backend_id == "transformers_nli"
    assert TransformersPairClassifier.backend_id == "transformers_pair_classifier"
