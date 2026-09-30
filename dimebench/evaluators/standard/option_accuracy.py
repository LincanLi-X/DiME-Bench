"""WinoGrande-style option accuracy."""

from __future__ import annotations

from pydantic import JsonValue

from dimebench.datasets import ReasoningRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, StandardEvaluator
from dimebench.schemas.result import ResultSpec


class OptionAccuracyEvaluator(StandardEvaluator):
    """Score parsed option indices for planning-style reasoning records."""

    METRIC_ID = "option_accuracy"

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, ReasoningRecord) or sample.answer_index is None:
            raise EvaluationError("option_accuracy requires reasoning choices")
        parsed = result.parsed_output
        if not isinstance(parsed, dict) or not isinstance(parsed.get("index"), int):
            raise EvaluationError("option_accuracy requires parsed_output.index")
        predicted = parsed["index"]
        return float(predicted == sample.answer_index), {
            "predicted_index": predicted,
            "answer_index": sample.answer_index,
        }


__all__ = ["OptionAccuracyEvaluator"]
