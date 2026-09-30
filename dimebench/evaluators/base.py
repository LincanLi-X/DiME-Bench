"""Versioned evaluator contracts and sample-level metric orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from typing import ClassVar

from pydantic import Field, JsonValue, model_validator

from dimebench.artifacts import SampleMetricRecord
from dimebench.artifacts.hashing import canonical_json, hash_json
from dimebench.datasets import SampleRecord
from dimebench.schemas.base import StrictSchema
from dimebench.schemas.result import ResultSpec


class EvaluationError(ValueError):
    """Raised when a metric cannot evaluate an otherwise eligible sample."""


class MetricOutcome(StrictSchema):
    """Auditable outcome for one metric on one retained sample."""

    metric_id: str = Field(min_length=1)
    metric_version: str = Field(min_length=1)
    value: float | None
    denominator: int = Field(default=1, ge=0, le=1)
    eligible: bool = True
    evaluator_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_eligibility_contract(self) -> MetricOutcome:
        if self.eligible and (self.value is None or self.denominator != 1):
            raise ValueError("eligible outcomes require a value and denominator=1")
        if not self.eligible and (self.value is not None or self.denominator != 0):
            raise ValueError("ineligible outcomes require value=null and denominator=0")
        return self


class StandardEvaluator(ABC):
    """Base class that enforces identity, failure-zero, and score contracts."""

    METRIC_ID: ClassVar[str]
    VERSION: ClassVar[str] = "1.0.0"
    MIN_VALUE: ClassVar[float] = 0.0
    MAX_VALUE: ClassVar[float] = 1.0

    @property
    def metric_id(self) -> str:
        return self.METRIC_ID

    def configuration(self) -> dict[str, JsonValue]:
        """Return every setting that can affect scores."""
        return {
            "metric_id": self.metric_id,
            "metric_version": self.VERSION,
        }

    @property
    def config_hash(self) -> str:
        return hash_json(self.configuration())

    def evaluate(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> MetricOutcome:
        """Evaluate one retained sample, assigning zero to terminal failures."""
        if result.sample_id != sample.sample_id:
            raise EvaluationError("result and sample IDs do not match")
        if result.dataset_id != sample.dataset_id:
            raise EvaluationError("result and sample dataset IDs do not match")
        if result.status != "success":
            details: dict[str, JsonValue] = {
                "route": "failure_zero",
                "result_status": result.status,
            }
            if result.failure_class is not None:
                details["failure_class"] = result.failure_class
            return self._outcome(0.0, details=details)
        value, details = self._score(result, sample)
        if not isfinite(value):
            raise EvaluationError(f"{self.metric_id} returned a non-finite value")
        if not self.MIN_VALUE <= value <= self.MAX_VALUE:
            raise EvaluationError(
                f"{self.metric_id} returned {value}, outside "
                f"[{self.MIN_VALUE}, {self.MAX_VALUE}]"
            )
        return self._outcome(value, details=details)

    def _outcome(
        self,
        value: float,
        *,
        details: dict[str, JsonValue] | None = None,
    ) -> MetricOutcome:
        return MetricOutcome(
            metric_id=self.metric_id,
            metric_version=self.VERSION,
            value=value,
            evaluator_config_hash=self.config_hash,
            details=details or {},
        )

    @abstractmethod
    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        """Return a successful sample's score and audit details."""


class IneligibleMetric(Exception):
    """Signal that a mechanism metric lacks required semantic inputs."""

    def __init__(
        self,
        reason: str,
        *,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class MechanismEvaluator(StandardEvaluator):
    """Evaluator whose missing semantic inputs remain null instead of zero."""

    def evaluate(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> MetricOutcome:
        if result.sample_id != sample.sample_id:
            raise EvaluationError("result and sample IDs do not match")
        if result.dataset_id != sample.dataset_id:
            raise EvaluationError("result and sample dataset IDs do not match")
        if result.status != "success":
            details: dict[str, JsonValue] = {
                "route": "ineligible_result",
                "reason": "result_not_successful",
                "result_status": result.status,
            }
            if result.failure_class is not None:
                details["failure_class"] = result.failure_class
            return self._ineligible(details)
        try:
            value, details = self._score(result, sample)
        except IneligibleMetric as exc:
            details = {"reason": exc.reason, **exc.details}
            return self._ineligible(details)
        if not isfinite(value):
            raise EvaluationError(f"{self.metric_id} returned a non-finite value")
        if not self.MIN_VALUE <= value <= self.MAX_VALUE:
            raise EvaluationError(
                f"{self.metric_id} returned {value}, outside "
                f"[{self.MIN_VALUE}, {self.MAX_VALUE}]"
            )
        return self._outcome(value, details=details)

    def _ineligible(self, details: dict[str, JsonValue]) -> MetricOutcome:
        return MetricOutcome(
            metric_id=self.metric_id,
            metric_version=self.VERSION,
            value=None,
            denominator=0,
            eligible=False,
            evaluator_config_hash=self.config_hash,
            details=details,
        )


def evaluate_sample(
    result: ResultSpec,
    sample: SampleRecord,
    evaluators: tuple[StandardEvaluator, ...] | list[StandardEvaluator],
) -> SampleMetricRecord:
    """Evaluate selected metrics and emit the standard sample-metric artifact."""
    prediction_hash = result.metadata.get("prediction_hash")
    if not isinstance(prediction_hash, str):
        raise EvaluationError("result metadata is missing prediction_hash")
    if len({evaluator.metric_id for evaluator in evaluators}) != len(evaluators):
        raise EvaluationError("evaluator metric IDs must be unique")

    outcomes = [evaluator.evaluate(result, sample) for evaluator in evaluators]
    metrics = {
        outcome.metric_id: outcome.value
        for outcome in outcomes
        if outcome.value is not None
    }
    metadata: dict[str, str] = {}
    for outcome in outcomes:
        prefix = outcome.metric_id
        metadata[f"{prefix}.version"] = outcome.metric_version
        metadata[f"{prefix}.config_hash"] = outcome.evaluator_config_hash
        metadata[f"{prefix}.denominator"] = str(outcome.denominator)
        metadata[f"{prefix}.eligible"] = str(outcome.eligible).lower()
        metadata[f"{prefix}.details"] = canonical_json(outcome.details)

    return SampleMetricRecord(
        run_id=result.run_id,
        config_hash=result.config_hash,
        sample_id=result.sample_id,
        model_id=result.model_id,
        dataset_id=result.dataset_id,
        task_id=result.task_id,
        prediction_hash=prediction_hash,
        status=result.status,
        metrics=metrics,
        evaluator_metadata=metadata,
    )


__all__ = [
    "EvaluationError",
    "IneligibleMetric",
    "MechanismEvaluator",
    "MetricOutcome",
    "StandardEvaluator",
    "evaluate_sample",
]
