from __future__ import annotations

import json
from pathlib import Path

import pytest

from dimebench.evaluators import EvaluationError
from dimebench.evaluators.mechanism import (
    BoundaryConsistencyEvaluator,
    ContradictionRateEvaluator,
    ContradictionReductionEvaluator,
)
from tests.mechanism_metric_fixtures import (
    infilling_sample,
    localized_edit_sample,
)
from tests.standard_metric_fixtures import result_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FrozenProbabilityBackend:
    backend_id = "step16_frozen_probability_backend"
    revision = "1.0.0"

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = iter(probabilities)

    def score_pair(self, left: str, right: str) -> float:
        del left, right
        return next(self._probabilities)

    def contradiction_probability(self, premise: str, hypothesis: str) -> float:
        del premise, hypothesis
        return next(self._probabilities)


@pytest.mark.golden
def test_neural_mechanism_metrics_match_frozen_expected_values() -> None:
    fixture_path = (
        PROJECT_ROOT / "tests" / "fixtures" / "mechanism_metrics" / "neural-golden.json"
    )
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    infilling = infilling_sample("fixture.test.neural-golden")
    editing = localized_edit_sample()

    for case in cases:
        backend = _FrozenProbabilityBackend(case["probabilities"])
        if case["metric"] == "boundary_consistency":
            outcome = BoundaryConsistencyEvaluator(
                backend, threshold=case["threshold"]
            ).evaluate(result_for(infilling, "candidate middle"), infilling)
        elif case["metric"] == "contradiction_rate":
            outcome = ContradictionRateEvaluator(
                backend, threshold=case["threshold"]
            ).evaluate(result_for(infilling, "candidate middle"), infilling)
        else:
            outcome = ContradictionReductionEvaluator(
                backend, threshold=case["threshold"]
            ).evaluate(
                result_for(editing, "Water freezes at 0 C."),
                editing,
            )
        assert outcome.value == pytest.approx(case["expected"]), case["name"]
        assert outcome.eligible is True


@pytest.mark.golden
def test_missing_edit_evidence_has_frozen_null_semantics() -> None:
    sample = localized_edit_sample().model_copy(update={"evidence": ()})
    outcome = ContradictionReductionEvaluator(_FrozenProbabilityBackend([])).evaluate(
        result_for(sample, "Water freezes at 0 C."), sample
    )
    assert outcome.value is None
    assert outcome.denominator == 0
    assert outcome.eligible is False
    assert outcome.details["reason"] == "no_contradiction_evidence"


@pytest.mark.golden
def test_out_of_range_backend_probability_is_rejected() -> None:
    sample = infilling_sample("fixture.test.invalid-probability")
    evaluator = BoundaryConsistencyEvaluator(_FrozenProbabilityBackend([1.1, 0.5]))
    with pytest.raises(EvaluationError, match="invalid probability"):
        evaluator.evaluate(result_for(sample, "candidate middle"), sample)
