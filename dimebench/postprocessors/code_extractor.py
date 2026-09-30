"""Python completion extraction without executing generated code."""

from __future__ import annotations

from pydantic import JsonValue

from dimebench.datasets import SampleRecord
from dimebench.postprocessors.base import (
    OutputParser,
    ParseError,
    strip_single_fence,
)


class CodeExtractorParser(OutputParser):
    """Extract plain code or one complete Markdown code block."""

    PARSER_ID = "code_extractor"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str = PARSER_ID) -> None:
        self.parser_id = parser_id

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        return (cls.PARSER_ID,)

    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        del sample
        normalized = raw_output.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        fenced = "```" in normalized
        code = strip_single_fence(normalized)
        if not code.strip():
            raise ParseError("code completion is empty")
        return {"code": code, "route": "fenced" if fenced else "plain"}


__all__ = ["CodeExtractorParser"]
