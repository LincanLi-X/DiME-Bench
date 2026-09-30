"""Track 3 contradiction-removal diagnostic."""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from dimebench.datasets import EditingRecord, SampleRecord
from dimebench.evaluators.base import (
    EvaluationError,
    IneligibleMetric,
    MechanismEvaluator,
)
from dimebench.evaluators.standard._text import prediction_text
from dimebench.schemas.result import ResultSpec

from .contradiction_rate import ContradictionBackend


class ContradictionReductionEvaluator(MechanismEvaluator):
    """Score removal of an evidence contradiction without a new one."""

    METRIC_ID = "contradiction_reduction_rate"

    def __init__(
        self, backend: ContradictionBackend, *, threshold: float = 0.5
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.backend = backend
        self.threshold = threshold

    def configuration(self) -> dict[str, JsonValue]:
        backend_config = getattr(self.backend, "configuration", None)
        details = (
            backend_config()
            if callable(backend_config)
            else {
                "backend_id": self.backend.backend_id,
                "revision": self.backend.revision,
            }
        )
        return {
            **super().configuration(),
            "backend": cast(JsonValue, details),
            "threshold": self.threshold,
            "aggregation": "all_evidence",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, EditingRecord):
            raise EvaluationError(
                "contradiction_reduction_rate requires an editing sample"
            )
        if not sample.evidence:
            raise IneligibleMetric("no_contradiction_evidence")
        output = prediction_text(result)
        source_probabilities = [
            self.backend.contradiction_probability(evidence, sample.source_text)
            for evidence in sample.evidence
        ]
        output_probabilities = [
            self.backend.contradiction_probability(evidence, output)
            for evidence in sample.evidence
        ]
        probabilities = source_probabilities + output_probabilities
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
            raise EvaluationError("contradiction backend returned invalid probability")
        source_has_contradiction = any(
            probability >= self.threshold for probability in source_probabilities
        )
        if not source_has_contradiction:
            raise IneligibleMetric(
                "source_contradiction_not_detected",
                details={"source_probabilities": cast(JsonValue, source_probabilities)},
            )
        output_has_contradiction = any(
            probability >= self.threshold for probability in output_probabilities
        )
        return float(not output_has_contradiction), {
            "source_probabilities": cast(JsonValue, source_probabilities),
            "output_probabilities": cast(JsonValue, output_probabilities),
            "source_contradiction": source_has_contradiction,
            "output_contradiction": output_has_contradiction,
        }


__all__ = ["ContradictionReductionEvaluator"]
