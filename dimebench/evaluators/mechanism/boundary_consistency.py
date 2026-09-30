"""Track 2 left/right boundary-consistency diagnostic."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, cast

from pydantic import JsonValue

from dimebench.datasets import InfillingRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, MechanismEvaluator
from dimebench.evaluators.standard._text import prediction_text
from dimebench.schemas.result import ResultSpec


class BoundaryPairBackend(Protocol):
    """Versioned pair classifier used for one text boundary."""

    backend_id: str
    revision: str

    def score_pair(self, left: str, right: str) -> float:
        """Return a consistency probability in ``[0, 1]``."""


class TransformersPairClassifier:
    """Lazy GPU-capable sequence-classification backend.

    The selected checkpoint must expose a label matching ``positive_label``.
    Loading is delayed until the first score so CPU-only installations can
    import and configure the evaluator without Transformers or Torch.
    """

    backend_id = "transformers_pair_classifier"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        positive_label: str = "consistent",
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        if not revision:
            raise ValueError("a frozen model revision is required")
        self.model_id = model_id
        self.revision = revision
        self.positive_label = positive_label.casefold()
        self.device = device
        self.max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def configuration(self) -> dict[str, JsonValue]:
        return {
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "positive_label": self.positive_label,
            "device": self.device,
            "max_length": self.max_length,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            transformers = import_module("transformers")
            torch = import_module("torch")
        except ImportError as exc:
            raise EvaluationError(
                "neural boundary diagnostics require dime-bench[hf]"
            ) from exc
        tokenizer_type = transformers.AutoTokenizer
        model_type = transformers.AutoModelForSequenceClassification
        self._tokenizer = tokenizer_type.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        self._model = model_type.from_pretrained(
            self.model_id,
            revision=self.revision,
        ).to(self.device)
        self._model.eval()
        self._torch = torch

    def label_probability(self, left: str, right: str, label: str) -> float:
        """Return the probability for one unambiguously named class label."""
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        if tokenizer is None or model is None or torch is None:
            raise EvaluationError("pair classifier did not initialize")
        encoded = tokenizer(
            left,
            right,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
        labels = {
            int(index): str(label).casefold()
            for index, label in model.config.id2label.items()
        }
        normalized_label = label.casefold()
        matching = [
            index
            for index, configured_label in labels.items()
            if normalized_label in configured_label
        ]
        if len(matching) != 1:
            raise EvaluationError(
                f"expected one label containing {normalized_label!r}, got {labels}"
            )
        return float(probabilities[matching[0]].item())

    def score_pair(self, left: str, right: str) -> float:
        return self.label_probability(left, right, self.positive_label)


class BoundaryConsistencyEvaluator(MechanismEvaluator):
    """Average binary consistency at the left and right generated boundaries."""

    METRIC_ID = "boundary_consistency"

    def __init__(self, backend: BoundaryPairBackend, *, threshold: float = 0.5) -> None:
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
            "aggregation": "mean_left_right_binary",
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, InfillingRecord):
            raise EvaluationError("boundary_consistency requires an infilling sample")
        middle = prediction_text(result)
        left_probability = self.backend.score_pair(sample.prefix, middle)
        right_probability = self.backend.score_pair(middle, sample.suffix)
        if not all(
            0.0 <= probability <= 1.0
            for probability in (left_probability, right_probability)
        ):
            raise EvaluationError("boundary backend returned an invalid probability")
        left_passed = left_probability >= self.threshold
        right_passed = right_probability >= self.threshold
        value = (float(left_passed) + float(right_passed)) / 2
        return value, {
            "left_probability": left_probability,
            "right_probability": right_probability,
            "left_consistent": left_passed,
            "right_consistent": right_passed,
        }


__all__ = [
    "BoundaryConsistencyEvaluator",
    "BoundaryPairBackend",
    "TransformersPairClassifier",
]
