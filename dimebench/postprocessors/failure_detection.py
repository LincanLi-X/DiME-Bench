"""Conservative, evidence-retaining output failure detectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import cast

from pydantic import JsonValue

from dimebench.datasets import InfillingRecord, SampleRecord
from dimebench.schemas.result import FailureClass
from dimebench.schemas.task import TaskSpec


@dataclass(frozen=True)
class FailureDetection:
    """All detected labels plus JSON-compatible evidence for each label."""

    labels: tuple[FailureClass, ...] = ()
    evidence: dict[str, JsonValue] = field(default_factory=dict)


_PRECEDENCE: tuple[FailureClass, ...] = (
    "runtime_error",
    "empty_output",
    "full_document_output",
    "context_copy",
    "truncated_output",
    "parse_error",
    "unsafe_code_result",
)


def _normalized_comparison(text: str) -> str:
    return " ".join(text.casefold().split())


def _context_failures(raw_output: str, sample: InfillingRecord) -> FailureDetection:
    output = _normalized_comparison(raw_output)
    prefix = _normalized_comparison(sample.prefix)
    suffix = _normalized_comparison(sample.suffix)
    if not output:
        return FailureDetection()

    exact_boundary = output == prefix or output == suffix
    copied_fragment = len(output.split()) >= 3 and (
        (output in prefix and output != prefix)
        or (output in suffix and output != suffix)
    )
    if exact_boundary or copied_fragment:
        return FailureDetection(
            labels=("context_copy",),
            evidence={
                "context_copy": {
                    "matched_boundary": "prefix" if output in prefix else "suffix",
                    "normalized_output": output,
                }
            },
        )

    matched = []
    if prefix and prefix in output:
        matched.append("prefix")
    if suffix and suffix in output:
        matched.append("suffix")
    if matched:
        return FailureDetection(
            labels=("full_document_output",),
            evidence={
                "full_document_output": {
                    "matched_boundaries": cast(JsonValue, matched),
                }
            },
        )
    return FailureDetection()


def _looks_truncated(
    raw_output: str,
    task: TaskSpec,
    finish_reason: str | None,
    parse_succeeded: bool,
) -> bool:
    if finish_reason != "length":
        return False
    if not parse_succeeded:
        return True
    if task.type in {"infilling", "editing", "code"}:
        return True
    if task.parser_id == "raw_text":
        return True
    stripped = raw_output.rstrip()
    return stripped.endswith(("```", "\\", "{", "[", "("))


def detect_failures(
    raw_output: str,
    sample: SampleRecord,
    task: TaskSpec,
    *,
    finish_reason: str | None,
    parse_succeeded: bool,
    parser_error: str | None,
) -> FailureDetection:
    """Detect failures after conservative normalization and before scoring."""
    labels: list[FailureClass] = []
    evidence: dict[str, JsonValue] = {}
    if not raw_output.strip():
        labels.append("empty_output")
        evidence["empty_output"] = {"whitespace_only": bool(raw_output)}
    elif isinstance(sample, InfillingRecord):
        context = _context_failures(raw_output, sample)
        labels.extend(context.labels)
        evidence.update(context.evidence)

    if _looks_truncated(raw_output, task, finish_reason, parse_succeeded):
        labels.append("truncated_output")
        evidence["truncated_output"] = {
            "finish_reason": finish_reason,
            "output_characters": len(raw_output),
        }
    if parser_error is not None and "empty_output" not in labels:
        labels.append("parse_error")
        evidence["parse_error"] = {
            "message": parser_error,
            "message_digest": hashlib.sha256(parser_error.encode("utf-8")).hexdigest(),
        }
    ordered = tuple(label for label in _PRECEDENCE if label in set(labels))
    return FailureDetection(labels=ordered, evidence=evidence)


def primary_failure(labels: tuple[FailureClass, ...]) -> FailureClass:
    """Return the frozen primary-label precedence for a non-empty label set."""
    if not labels:
        raise ValueError("primary failure requires at least one label")
    for label in _PRECEDENCE:
        if label in labels:
            return label
    raise AssertionError("unknown failure label")


__all__ = [
    "FailureDetection",
    "detect_failures",
    "primary_failure",
]
