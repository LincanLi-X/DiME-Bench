"""Planning-path output parsing."""

from __future__ import annotations

import json
import re

from pydantic import JsonValue

from dimebench.datasets import SampleRecord
from dimebench.postprocessors.base import OutputParser, ParseError, normalize_text


class PathAnswerParser(OutputParser):
    """Parse JSON or arrow/hyphen/comma-separated node paths."""

    PARSER_ID = "path_answer"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str = PARSER_ID) -> None:
        self.parser_id = parser_id

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        return (cls.PARSER_ID,)

    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        del sample
        text = normalize_text(raw_output)
        text = re.sub(r"(?i)^\s*(?:final\s+)?(?:answer|path)\s*:?\s*", "", text)
        if not text:
            raise ParseError("path output is empty")
        route = "delimited"
        if text.startswith("["):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ParseError("path JSON is invalid") from exc
            if not isinstance(payload, list) or not all(
                isinstance(node, str) for node in payload
            ):
                raise ParseError("path JSON must be a list of node strings")
            nodes = [node.strip() for node in payload]
            route = "json"
        else:
            nodes = [node.strip() for node in re.split(r"\s*(?:->|→|—|–|-|,)\s*", text)]
        if len(nodes) < 2 or any(not node for node in nodes):
            raise ParseError("path requires at least two non-empty nodes")
        if any(not re.fullmatch(r"[A-Za-z0-9_.]+", node) for node in nodes):
            raise ParseError("path contains an invalid node identifier")
        return {"nodes": nodes, "route": route}


__all__ = ["PathAnswerParser"]
