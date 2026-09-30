"""Deterministic final numeric-answer parsing."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from pydantic import JsonValue

from dimebench.datasets import SampleRecord
from dimebench.postprocessors.base import OutputParser, ParseError, normalize_text

_NUMBER = (
    r"[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][+\-]?\d+)?(?:\s*/\s*[+\-]?\d+)?%?"
)


def _canonical_number(value: str) -> str:
    cleaned = value.replace(",", "").replace("−", "-").replace(" ", "")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            fraction = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
            canonical = (
                str(fraction.numerator)
                if fraction.denominator == 1
                else f"{fraction.numerator}/{fraction.denominator}"
            )
        else:
            decimal = Decimal(cleaned)
            canonical = format(decimal.normalize(), "f")
            if "." in canonical:
                canonical = canonical.rstrip("0").rstrip(".")
            if canonical in {"-0", "+0", ""}:
                canonical = "0"
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise ParseError(f"invalid numeric answer {value!r}") from exc
    return canonical + ("%" if percent else "")


class NumericAnswerParser(OutputParser):
    """Extract one final number using frozen marker and last-number rules."""

    PARSER_ID = "numeric_answer"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str = PARSER_ID) -> None:
        self.parser_id = parser_id

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        return (cls.PARSER_ID,)

    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        del sample
        text = normalize_text(raw_output)
        if not text:
            raise ParseError("numeric output is empty")
        latex_fraction = re.findall(
            r"\\frac\s*\{\s*([+\-]?\d+)\s*\}\s*\{\s*([+\-]?\d+)\s*\}",
            text,
        )
        boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
        marker = re.findall(
            rf"(?i)(?:####|final\s+answer|answer)\s*(?:is|:|=)?\s*({_NUMBER})",
            text,
        )
        route = "last_number"
        if boxed:
            candidate = boxed[-1]
            route = "boxed"
        elif latex_fraction:
            numerator, denominator = latex_fraction[-1]
            candidate = f"{numerator}/{denominator}"
            route = "latex_fraction"
        elif marker:
            candidate = marker[-1]
            route = "answer_marker"
        else:
            numbers = re.findall(_NUMBER, text)
            if not numbers:
                raise ParseError("no numeric answer was found")
            candidate = numbers[-1]
        return {"value": _canonical_number(candidate), "route": route}


__all__ = ["NumericAnswerParser"]
