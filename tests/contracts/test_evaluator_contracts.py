from __future__ import annotations

from pathlib import Path

from dimebench.config import load_yaml
from dimebench.evaluators.standard import available_standard_evaluators
from dimebench.schemas.task import TaskSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_every_standard_task_metric_resolves_to_an_evaluator() -> None:
    standard = set(available_standard_evaluators())
    expected = {
        "accuracy",
        "normalized_exact_match",
        "pass_at_1",
        "instruction_following_rate",
        "token_f1",
        "rouge_l",
        "bertscore_f1",
        "option_accuracy",
        "path_validity",
    }
    configured = set()
    for path in sorted((PROJECT_ROOT / "configs/tasks").glob("*/*.yaml")):
        task = TaskSpec.model_validate(load_yaml(path))
        configured.update(metric for metric in task.metrics if metric in expected)
    assert configured == expected
    assert configured <= standard


def test_standard_evaluator_versions_and_hashes_are_stable() -> None:
    registry = available_standard_evaluators()
    assert len(registry) == 9
    assert set(registry.values()) == {"1.0.0"}
