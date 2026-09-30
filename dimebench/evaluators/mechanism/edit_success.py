"""Dataset-routed edit success."""

from __future__ import annotations

from typing import Protocol, cast

from pydantic import JsonValue

from dimebench.datasets import EditingRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, MechanismEvaluator
from dimebench.evaluators.standard._text import (
    normalized_exact_text,
    prediction_text,
)
from dimebench.schemas.result import ResultSpec


class EditSuccessChecker(Protocol):
    """Backend boundary for reference, rule, M2, or entailment checkers."""

    checker_id: str
    revision: str

    def check(self, output: str, sample: EditingRecord) -> bool:
        """Return whether the requested edit was successfully completed."""


class ReferenceEditChecker:
    """Deterministic normalized-reference checker used by local fixtures."""

    checker_id = "normalized_reference_exact"
    revision = "1.0.0"

    def check(self, output: str, sample: EditingRecord) -> bool:
        return normalized_exact_text(output) == normalized_exact_text(
            sample.reference_edit
        )


class EditSuccessEvaluator(MechanismEvaluator):
    """Evaluate edit success through a frozen dataset-appropriate checker."""

    METRIC_ID = "edit_success"

    def __init__(self, checker: EditSuccessChecker | None = None) -> None:
        self.checker = checker or ReferenceEditChecker()

    def configuration(self) -> dict[str, JsonValue]:
        checker_config = getattr(self.checker, "configuration", None)
        details = (
            checker_config()
            if callable(checker_config)
            else {
                "checker_id": self.checker.checker_id,
                "revision": self.checker.revision,
            }
        )
        return {**super().configuration(), "checker": cast(JsonValue, details)}

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, EditingRecord):
            raise EvaluationError("edit_success requires an editing sample")
        passed = self.checker.check(prediction_text(result), sample)
        return float(passed), {"edit_succeeded": passed}


__all__ = [
    "EditSuccessChecker",
    "EditSuccessEvaluator",
    "ReferenceEditChecker",
]
