"""Versioned normalized exact match."""

from __future__ import annotations

from pydantic import JsonValue

from dimebench.datasets import SampleRecord
from dimebench.evaluators.base import StandardEvaluator
from dimebench.evaluators.standard._text import (
    normalized_exact_text,
    prediction_text,
    references_for,
)
from dimebench.schemas.result import ResultSpec


class ExactMatchEvaluator(StandardEvaluator):
    """Match normalized text or canonical numeric values against references."""

    METRIC_ID = "normalized_exact_match"

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "normalization": "nfkc_casefold_whitespace_numeric_v1",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        prediction = normalized_exact_text(prediction_text(result))
        references = tuple(
            normalized_exact_text(reference) for reference in references_for(sample)
        )
        return float(prediction in references), {
            "normalized_prediction": prediction,
            "normalized_references": list(references),
        }


__all__ = ["ExactMatchEvaluator", "normalized_exact_text"]
