"""Multiple-choice option parsing."""

from __future__ import annotations

import json
import re

from pydantic import JsonValue

from dimebench.datasets import MultipleChoiceRecord, ReasoningRecord, SampleRecord
from dimebench.postprocessors.base import OutputParser, ParseError, normalize_text


class MultipleChoiceParser(OutputParser):
    """Parse a direct option label, exact option text, or option-score mapping."""

    PARSER_ID = "option_letter"
    VERSION = "1.0.0"

    def __init__(self, parser_id: str = PARSER_ID) -> None:
        self.parser_id = parser_id

    @classmethod
    def parser_ids(cls) -> tuple[str, ...]:
        return ("option_letter", "option_letter_or_text")

    @staticmethod
    def _choices(sample: SampleRecord) -> tuple[str, ...]:
        if (
            isinstance(sample, (MultipleChoiceRecord, ReasoningRecord))
            and sample.choices
        ):
            return sample.choices
        raise ParseError("multiple-choice parser requires sample choices")

    @staticmethod
    def _result(index: int, choices: tuple[str, ...], route: str) -> JsonValue:
        if index >= len(choices):
            raise ParseError("parsed option label is outside the available choices")
        return {
            "index": index,
            "label": chr(ord("A") + index),
            "text": choices[index],
            "route": route,
        }

    def _score_mapping(self, text: str, choices: tuple[str, ...]) -> JsonValue | None:
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or set(payload) != set(choices):
            return None
        if any(not isinstance(value, (int, float)) for value in payload.values()):
            raise ParseError("option-score mapping contains a non-numeric score")
        maximum = max(float(value) for value in payload.values())
        winners = [key for key, value in payload.items() if float(value) == maximum]
        if len(winners) != 1:
            raise ParseError("option-score mapping has a tie")
        return self._result(choices.index(winners[0]), choices, "option_scores")

    def parse(self, raw_output: str, sample: SampleRecord) -> JsonValue:
        choices = self._choices(sample)
        if len(choices) > 26:
            raise ParseError("option-letter parser supports at most 26 choices")
        text = normalize_text(raw_output)
        if not text:
            raise ParseError("option output is empty")
        score_result = self._score_mapping(text, choices)
        if score_result is not None:
            return score_result

        direct = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?[.)]?", text)
        if direct:
            return self._result(
                ord(direct.group(1).upper()) - ord("A"), choices, "label"
            )
        marker = re.fullmatch(
            r"(?i)(?:final\s+)?(?:answer|option)\s*(?:is|:|=|-)?\s*"
            r"\(?\s*([A-Za-z])\s*\)?[.)]?",
            text,
        )
        if marker:
            return self._result(
                ord(marker.group(1).upper()) - ord("A"), choices, "label"
            )

        normalized = " ".join(text.casefold().split())
        matches = [
            index
            for index, choice in enumerate(choices)
            if normalized == " ".join(choice.casefold().split())
        ]
        if len(matches) == 1 and self.parser_id == "option_letter_or_text":
            return self._result(matches[0], choices, "exact_text")
        raise ParseError("output is not one unambiguous option label or allowed text")


__all__ = ["MultipleChoiceParser"]
