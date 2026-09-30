"""Missing-middle span extraction."""

from __future__ import annotations

import re

from pydantic import JsonValue

from dimebench.datasets import InfillingRecord, SampleRecord
from dimebench.postprocessors.base import (
    OutputParser,
    ParseError,
    normalize_text,
    strip_single_fence,
)


class InfillSpanParser(OutputParser):
    """Extract only the generated middle span, retaining its text verbatim."""

    PARSER_ID = "middle_only"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str = PARSER_ID) -> None:
        self.parser_id = parser_id

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        return (cls.PARSER_ID,)

    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        if not isinstance(sample, InfillingRecord):
            raise ParseError("middle-only parser requires an infilling sample")
        text = strip_single_fence(raw_output)
        text = re.sub(
            r"(?is)^\s*(?:<middle>|<middle_output>|\[middle(?:\s+output)?\]|"
            r"missing\s+middle|middle\s+span)\s*:?\s*",
            "",
            text,
            count=1,
        )
        text = re.sub(
            r"(?is)\s*(?:</middle>|</middle_output>|\[/middle(?:\s+output)?\])\s*$",
            "",
            text,
            count=1,
        )
        parsed = normalize_text(text)
        if not parsed:
            raise ParseError("middle span is empty after wrapper removal")
        return parsed


__all__ = ["InfillSpanParser"]
