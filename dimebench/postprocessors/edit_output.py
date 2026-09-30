"""Full-text edit and unconstrained raw-text parsing."""

from __future__ import annotations

import re

from pydantic import JsonValue

from dimebench.datasets import EditingRecord, SampleRecord
from dimebench.postprocessors.base import (
    OutputParser,
    ParseError,
    normalize_text,
    strip_single_fence,
)


class EditOutputParser(OutputParser):
    """Extract a complete revised document without commentary wrappers."""

    PARSER_ID = "full_text_edit"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str = PARSER_ID) -> None:
        self.parser_id = parser_id

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        return (cls.PARSER_ID,)

    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        if not isinstance(sample, EditingRecord):
            raise ParseError("full-text edit parser requires an editing sample")
        text = strip_single_fence(raw_output)
        text = re.sub(
            r"(?is)^\s*(?:\[revised\s+text\]|<revised_text>|revised\s+text)\s*:?\s*",
            "",
            text,
            count=1,
        )
        parsed = normalize_text(text)
        if not parsed:
            raise ParseError("revised text is empty after wrapper removal")
        return parsed


class RawTextParser(OutputParser):
    """Preserve normalized free-form text for official downstream checkers."""

    PARSER_ID = "raw_text"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str = PARSER_ID) -> None:
        self.parser_id = parser_id

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        return (cls.PARSER_ID,)

    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        del sample
        parsed = normalize_text(raw_output)
        if not parsed:
            raise ParseError("text output is empty")
        return parsed


__all__ = ["EditOutputParser", "RawTextParser"]
