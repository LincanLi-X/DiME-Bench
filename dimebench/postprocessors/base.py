"""Common parser interface and raw-prediction postprocessing pipeline."""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from dimebench.artifacts import PredictionRecord
from dimebench.artifacts.hashing import canonical_json, hash_json
from dimebench.datasets import SampleRecord
from dimebench.postprocessors.failure_detection import (
    FailureDetection,
    detect_failures,
    primary_failure,
)
from dimebench.schemas.result import ResultSpec
from dimebench.schemas.task import TaskSpec


class ParseError(ValueError):
    """Raised when non-empty model output violates a parser contract."""


class PostprocessConfigurationError(ValueError):
    """Raised for incompatible parser configuration or record identity."""


class OutputParser(ABC):
    """Versioned deterministic output parser."""

    PARSER_ID = "base"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str | None = None) -> None:
        self.parser_id = parser_id or self.PARSER_ID

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        """Return every configuration-facing ID handled by this parser."""
        return (cls.PARSER_ID,)

    @abstractmethod
    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        """Parse raw model output or raise :class:`ParseError`."""


def normalize_text(text: str) -> str:
    """Normalize newlines and surrounding whitespace without changing content."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def strip_single_fence(text: str) -> str:
    """Remove one complete fence while preserving meaningful code indentation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if "```" not in normalized:
        return normalized
    if normalized.count("```") != 2 or not normalized.startswith("```"):
        raise ParseError("output contains an incomplete or ambiguous code fence")
    opening_end = normalized.find("\n")
    if opening_end < 0 or not normalized.endswith("```"):
        raise ParseError("output contains an incomplete code fence")
    opening = normalized[3:opening_end].strip().lower()
    if opening and opening not in {"python", "py", "text", "plaintext"}:
        raise ParseError(f"unsupported fenced output language {opening!r}")
    return normalized[opening_end + 1 : -3].strip("\n")


def _parser_types() -> tuple[type[OutputParser], ...]:
    from dimebench.postprocessors.code_extractor import CodeExtractorParser
    from dimebench.postprocessors.edit_output import EditOutputParser, RawTextParser
    from dimebench.postprocessors.infill_span import InfillSpanParser
    from dimebench.postprocessors.multiple_choice import MultipleChoiceParser
    from dimebench.postprocessors.numeric_answer import NumericAnswerParser
    from dimebench.postprocessors.path_answer import PathAnswerParser

    return (
        MultipleChoiceParser,
        NumericAnswerParser,
        CodeExtractorParser,
        InfillSpanParser,
        EditOutputParser,
        RawTextParser,
        PathAnswerParser,
    )


def available_parsers() -> dict[str, str]:
    """Return parser IDs and frozen implementation versions."""
    parsers: dict[str, str] = {}
    for parser_type in _parser_types():
        for parser_id in parser_type.parser_ids():
            if parser_id in parsers:
                raise PostprocessConfigurationError(
                    f"duplicate output parser registration {parser_id!r}"
                )
            parsers[parser_id] = parser_type.VERSION
    return parsers


def create_parser(parser_id: str, parser_version: str) -> OutputParser:
    """Create a parser only when its configured version exactly matches."""
    for parser_type in _parser_types():
        if parser_id in parser_type.parser_ids():
            if parser_version != parser_type.VERSION:
                raise PostprocessConfigurationError(
                    f"parser {parser_id!r} requires version {parser_type.VERSION!r}, "
                    f"got {parser_version!r}"
                )
            return parser_type(parser_id)
    known = ", ".join(sorted(available_parsers()))
    raise PostprocessConfigurationError(
        f"unknown parser {parser_id!r}; available parsers: {known}"
    )


def _finish_reason(prediction: PredictionRecord) -> str | None:
    inference = prediction.decoding_metadata.get("inference")
    if not isinstance(inference, dict):
        return None
    value = inference.get("finish_reason")
    return value if isinstance(value, str) else None


def _identity_check(
    prediction: PredictionRecord,
    sample: SampleRecord,
    task: TaskSpec,
) -> None:
    mismatches = []
    if prediction.sample_id != sample.sample_id:
        mismatches.append("sample_id")
    if prediction.dataset_id != sample.dataset_id:
        mismatches.append("dataset_id")
    if prediction.task_id != task.id:
        mismatches.append("task_id")
    if task.dataset_id != sample.dataset_id:
        mismatches.append("task.dataset_id")
    if mismatches:
        raise PostprocessConfigurationError(
            "postprocessing identity mismatch: " + ", ".join(mismatches)
        )


def _metadata(
    prediction: PredictionRecord,
    task: TaskSpec,
    detection: FailureDetection,
    parser_error: str | None,
) -> dict[str, JsonValue]:
    raw_hash = (
        hash_json(prediction.raw_output) if prediction.raw_output is not None else None
    )
    return {
        "prediction_hash": prediction.record_hash,
        "raw_output_hash": raw_hash,
        "parser": {
            "id": task.parser_id,
            "version": task.parser_version,
        },
        "failure_labels": cast(JsonValue, list(detection.labels)),
        "failure_evidence": cast(JsonValue, detection.evidence),
        "parser_error": parser_error,
    }


def postprocess_prediction(
    prediction: PredictionRecord,
    sample: SampleRecord,
    task: TaskSpec,
) -> ResultSpec:
    """Parse one raw prediction while retaining all failure evidence."""
    _identity_check(prediction, sample, task)
    parser = create_parser(task.parser_id, task.parser_version)

    if prediction.status == "skipped":
        return ResultSpec(
            run_id=prediction.run_id,
            config_hash=prediction.config_hash,
            sample_id=prediction.sample_id,
            model_id=prediction.model_id,
            dataset_id=prediction.dataset_id,
            task_id=prediction.task_id,
            status="skipped",
            raw_output=prediction.raw_output,
            metadata=_metadata(
                prediction,
                task,
                FailureDetection(),
                None,
            ),
        )

    if prediction.status == "failure":
        failure = prediction.failure_class or "runtime_error"
        detection = FailureDetection(
            labels=(failure,),
            evidence={
                failure: {
                    "source": "inference",
                    "error_message": prediction.error_message,
                }
            },
        )
        return ResultSpec(
            run_id=prediction.run_id,
            config_hash=prediction.config_hash,
            sample_id=prediction.sample_id,
            model_id=prediction.model_id,
            dataset_id=prediction.dataset_id,
            task_id=prediction.task_id,
            status="failure",
            raw_output=prediction.raw_output,
            failure_class=failure,
            error_message=prediction.error_message or "inference failed",
            metadata=_metadata(prediction, task, detection, None),
        )

    raw_output = prediction.raw_output or ""
    parsed: JsonValue | None = None
    parser_error: str | None = None
    try:
        parsed = parser.parse(raw_output, sample)
    except ParseError as exc:
        parser_error = str(exc)

    detection = detect_failures(
        raw_output,
        sample,
        task,
        finish_reason=_finish_reason(prediction),
        parse_succeeded=parser_error is None,
        parser_error=parser_error,
    )
    metadata = _metadata(prediction, task, detection, parser_error)
    if detection.labels:
        primary = primary_failure(detection.labels)
        return ResultSpec(
            run_id=prediction.run_id,
            config_hash=prediction.config_hash,
            sample_id=prediction.sample_id,
            model_id=prediction.model_id,
            dataset_id=prediction.dataset_id,
            task_id=prediction.task_id,
            status="failure",
            raw_output=prediction.raw_output,
            parsed_output=parsed,
            failure_class=primary,
            error_message=parser_error or f"detected {primary}",
            metadata=metadata,
        )
    return ResultSpec(
        run_id=prediction.run_id,
        config_hash=prediction.config_hash,
        sample_id=prediction.sample_id,
        model_id=prediction.model_id,
        dataset_id=prediction.dataset_id,
        task_id=prediction.task_id,
        status="success",
        raw_output=prediction.raw_output,
        parsed_output=parsed,
        metadata=metadata,
    )


def postprocess_predictions(
    predictions: Iterable[PredictionRecord],
    samples: Mapping[str, SampleRecord],
    task: TaskSpec,
) -> tuple[ResultSpec, ...]:
    """Postprocess an ordered run without dropping failed samples."""
    results = []
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.sample_id in seen:
            raise PostprocessConfigurationError(
                f"duplicate prediction sample_id {prediction.sample_id!r}"
            )
        seen.add(prediction.sample_id)
        try:
            sample = samples[prediction.sample_id]
        except KeyError as exc:
            raise PostprocessConfigurationError(
                f"missing normalized sample {prediction.sample_id!r}"
            ) from exc
        results.append(postprocess_prediction(prediction, sample, task))
    return tuple(results)


def write_postprocessed_results(
    path: str | Path,
    results: Iterable[ResultSpec],
) -> None:
    """Atomically write ordered sample results as canonical JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            for result in results:
                stream.write(canonical_json(result) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "OutputParser",
    "ParseError",
    "PostprocessConfigurationError",
    "available_parsers",
    "create_parser",
    "normalize_text",
    "postprocess_prediction",
    "postprocess_predictions",
    "strip_single_fence",
    "write_postprocessed_results",
]
