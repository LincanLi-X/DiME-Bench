"""Shared deterministic text and reference normalization."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from dimebench.datasets import (
    EditingRecord,
    GenerationRecord,
    InfillingRecord,
    ReasoningRecord,
    SampleRecord,
)
from dimebench.evaluators.base import EvaluationError
from dimebench.schemas.result import ResultSpec

_TOKEN = re.compile(r"[a-z0-9]+")


def prediction_text(result: ResultSpec) -> str:
    """Extract the semantic prediction from a parsed successful result."""
    parsed = result.parsed_output
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("value", "code", "text"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    raise EvaluationError("parsed output does not contain a text prediction")


def references_for(sample: SampleRecord) -> tuple[str, ...]:
    """Return one or more frozen references for a supported sample."""
    if isinstance(sample, GenerationRecord):
        return sample.references
    if isinstance(sample, InfillingRecord):
        return (sample.middle,)
    if isinstance(sample, EditingRecord):
        return (sample.reference_edit,)
    if isinstance(sample, ReasoningRecord):
        return (sample.answer,)
    raise EvaluationError(f"sample kind {sample.kind!r} has no text reference")


def word_tokens(text: str) -> tuple[str, ...]:
    """Tokenize like the no-stemming ROUGE basic tokenizer."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN.findall(normalized))


def canonical_number(text: str) -> str | None:
    """Canonicalize a complete decimal, fraction, percent, or LaTeX fraction."""
    value = unicodedata.normalize("NFKC", text).strip()
    boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}", value)
    if boxed:
        value = boxed.group(1).strip()
    latex_fraction = re.fullmatch(
        r"\\frac\s*\{\s*([+\-]?\d+(?:\.\d+)?)\s*\}"
        r"\s*\{\s*([+\-]?\d+(?:\.\d+)?)\s*\}",
        value,
    )
    if latex_fraction:
        value = f"{latex_fraction.group(1)}/{latex_fraction.group(2)}"
    value = value.replace(",", "").replace("−", "-").replace(" ", "")
    value = value.removeprefix("$").removesuffix("$")
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    if not re.fullmatch(
        r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
        r"(?:/[+\-]?(?:\d+(?:\.\d+)?|\.\d+))?",
        value,
    ):
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", maxsplit=1)
            number = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
            canonical = (
                str(number.numerator)
                if number.denominator == 1
                else f"{number.numerator}/{number.denominator}"
            )
        else:
            decimal = Decimal(value)
            number = Fraction(decimal)
            canonical = (
                str(number.numerator)
                if number.denominator == 1
                else f"{number.numerator}/{number.denominator}"
            )
            if canonical in {"-0", "+0", ""}:
                canonical = "0"
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return canonical + ("%" if percent else "")


def normalized_exact_text(text: str) -> str:
    """Apply v1 numeric canonicalization or conservative text normalization."""
    number = canonical_number(text)
    if number is not None:
        return number
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return " ".join(normalized.split())


__all__ = [
    "canonical_number",
    "normalized_exact_text",
    "prediction_text",
    "references_for",
    "word_tokens",
]
