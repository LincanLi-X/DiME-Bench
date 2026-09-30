"""Track 3 unnecessary-edit diagnostic."""

from __future__ import annotations

from pydantic import JsonValue

from dimebench.datasets import EditingRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, MechanismEvaluator
from dimebench.evaluators.mechanism._editing import token_edit_distance
from dimebench.evaluators.standard._text import prediction_text, word_tokens
from dimebench.schemas.result import ResultSpec


class OverEditRateEvaluator(MechanismEvaluator):
    """Compute excess token edits relative to the frozen minimum distance."""

    METRIC_ID = "over_edit_rate"

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "tokenizer": "nfkc_alphanumeric_v1",
            "distance": "token_levenshtein_v1",
            "formula": "max(actual-minimum,0)/max(actual,1)",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, EditingRecord):
            raise EvaluationError("over_edit_rate requires an editing sample")
        source = word_tokens(sample.source_text)
        output = word_tokens(prediction_text(result))
        actual = token_edit_distance(source, output)
        minimum = sample.minimum_edit_distance
        value = max(actual - minimum, 0) / max(actual, 1)
        return value, {
            "actual_edit_distance": actual,
            "minimum_edit_distance": minimum,
        }


__all__ = ["OverEditRateEvaluator"]
