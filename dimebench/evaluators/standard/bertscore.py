"""BERTScore F1 with lazy optional loading and frozen backend metadata."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

from pydantic import JsonValue

from dimebench.datasets import SampleRecord
from dimebench.evaluators.base import EvaluationError, StandardEvaluator
from dimebench.evaluators.standard._text import prediction_text, references_for
from dimebench.schemas.result import ResultSpec


class BERTScoreBackend(Protocol):
    """Backend contract that allows deterministic fixture and production scorers."""

    backend_id: str
    revision: str

    def score(
        self,
        prediction: str,
        reference: str,
        *,
        model_type: str,
        idf: bool,
        rescale_with_baseline: bool,
        device: str | None,
    ) -> float:
        """Return BERTScore F1 for one prediction/reference pair."""


class BERTScorePackageBackend:
    """Lazy adapter for the optional ``bert-score`` package."""

    backend_id = "bert_score.package"
    revision = "0.3.13-api"

    def score(
        self,
        prediction: str,
        reference: str,
        *,
        model_type: str,
        idf: bool,
        rescale_with_baseline: bool,
        device: str | None,
    ) -> float:
        try:
            score = import_module("bert_score").score
        except ImportError as exc:
            raise EvaluationError(
                "BERTScore requires `pip install dime-bench[metrics]`"
            ) from exc
        _, _, f1 = score(
            [prediction],
            [reference],
            model_type=model_type,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
            verbose=False,
        )
        return float(f1[0].item())


class BERTScoreEvaluator(StandardEvaluator):
    """Return the maximum BERTScore F1 over frozen references."""

    METRIC_ID = "bertscore_f1"
    MIN_VALUE = -1.0

    def __init__(
        self,
        backend: BERTScoreBackend | None = None,
        *,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        model_revision: str = "frozen-by-release-manifest",
        tokenizer_revision: str = "frozen-by-release-manifest",
        idf: bool = False,
        rescale_with_baseline: bool = True,
        device: str | None = None,
    ) -> None:
        self.backend = backend or BERTScorePackageBackend()
        self.model_type = model_type
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.idf = idf
        self.rescale_with_baseline = rescale_with_baseline
        self.device = device

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "backend_id": self.backend.backend_id,
            "backend_revision": self.backend.revision,
            "model_type": self.model_type,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "idf": self.idf,
            "rescale_with_baseline": self.rescale_with_baseline,
            "device": self.device,
            "reference_aggregation": "max",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        prediction = prediction_text(result)
        scores = [
            self.backend.score(
                prediction,
                reference,
                model_type=self.model_type,
                idf=self.idf,
                rescale_with_baseline=self.rescale_with_baseline,
                device=self.device,
            )
            for reference in references_for(sample)
        ]
        return max(scores), {
            "reference_scores": cast(JsonValue, scores),
            "aggregation": "max",
        }


__all__ = [
    "BERTScoreBackend",
    "BERTScoreEvaluator",
    "BERTScorePackageBackend",
]
