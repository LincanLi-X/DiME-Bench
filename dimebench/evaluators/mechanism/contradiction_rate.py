"""Track 2 contradiction-rate diagnostic."""

from __future__ import annotations

from typing import Protocol, cast

from pydantic import JsonValue

from dimebench.datasets import InfillingRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, MechanismEvaluator
from dimebench.evaluators.mechanism.boundary_consistency import (
    TransformersPairClassifier,
)
from dimebench.evaluators.standard._text import prediction_text
from dimebench.schemas.result import ResultSpec


class ContradictionBackend(Protocol):
    """Versioned NLI boundary returning contradiction probability."""

    backend_id: str
    revision: str

    def contradiction_probability(self, premise: str, hypothesis: str) -> float:
        """Return contradiction probability in ``[0, 1]``."""


class TransformersNLIBackend(TransformersPairClassifier):
    """GPU-capable frozen NLI checkpoint adapter."""

    backend_id = "transformers_nli"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        super().__init__(
            model_id,
            revision=revision,
            positive_label="contradiction",
            device=device,
            max_length=max_length,
        )

    def contradiction_probability(self, premise: str, hypothesis: str) -> float:
        return self.score_pair(premise, hypothesis)


class ContradictionRateEvaluator(MechanismEvaluator):
    """Flag contradiction between a generated middle and either boundary."""

    METRIC_ID = "contradiction_rate"

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
            "aggregation": "any_boundary",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, InfillingRecord):
            raise EvaluationError("contradiction_rate requires an infilling sample")
        middle = prediction_text(result)
        left_probability = self.backend.contradiction_probability(
            sample.prefix,
            middle,
        )
        right_probability = self.backend.contradiction_probability(
            sample.suffix,
            middle,
        )
        if not all(
            0.0 <= probability <= 1.0
            for probability in (left_probability, right_probability)
        ):
            raise EvaluationError("contradiction backend returned invalid probability")
        contradicted = max(left_probability, right_probability) >= self.threshold
        return float(contradicted), {
            "left_probability": left_probability,
            "right_probability": right_probability,
            "contradiction_detected": contradicted,
        }


__all__ = [
    "ContradictionBackend",
    "ContradictionRateEvaluator",
    "TransformersNLIBackend",
]
