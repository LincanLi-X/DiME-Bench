"""Multiple-choice accuracy."""

from __future__ import annotations

from pydantic import JsonValue

from dimebench.datasets import MultipleChoiceRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, StandardEvaluator
from dimebench.schemas.result import ResultSpec


class AccuracyEvaluator(StandardEvaluator):
    """Score parsed multiple-choice indices against the frozen answer index."""

    METRIC_ID = "accuracy"

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, MultipleChoiceRecord):
            raise EvaluationError("accuracy requires a multiple-choice sample")
        parsed = result.parsed_output
        if not isinstance(parsed, dict) or not isinstance(parsed.get("index"), int):
            raise EvaluationError("accuracy requires parsed_output.index")
        predicted = parsed["index"]
        return float(predicted == sample.answer_index), {
            "predicted_index": predicted,
            "answer_index": sample.answer_index,
        }


__all__ = ["AccuracyEvaluator"]
