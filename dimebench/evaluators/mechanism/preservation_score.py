"""Track 3 non-target preservation score."""

from __future__ import annotations

from pydantic import JsonValue

from dimebench.datasets import EditingRecord, SampleRecord
from dimebench.evaluators.base import (
    EvaluationError,
    IneligibleMetric,
    MechanismEvaluator,
)
from dimebench.evaluators.mechanism._editing import (
    lcs_length,
    non_target_source_tokens,
)
from dimebench.evaluators.standard._text import prediction_text, word_tokens
from dimebench.schemas.result import ResultSpec


class PreservationScoreEvaluator(MechanismEvaluator):
    """Measure how much frozen non-target source content remains in output."""

    METRIC_ID = "preservation_score"

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "tokenizer": "nfkc_alphanumeric_v1",
            "alignment": "gold_spans_else_source_reference_diff_v1",
            "formula": "lcs(non_target_source,output)/len(non_target_source)",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, EditingRecord):
            raise EvaluationError("preservation_score requires an editing sample")
        non_target = non_target_source_tokens(sample)
        if not non_target:
            raise IneligibleMetric("no_non_target_content")
        output = word_tokens(prediction_text(result))
        overlap = lcs_length(non_target, output)
        return overlap / len(non_target), {
            "non_target_token_count": len(non_target),
            "preserved_lcs_tokens": overlap,
            "target_route": (
                "declared_non_target_spans"
                if sample.non_target_spans
                else "source_reference_diff"
            ),
        }


__all__ = ["PreservationScoreEvaluator"]
