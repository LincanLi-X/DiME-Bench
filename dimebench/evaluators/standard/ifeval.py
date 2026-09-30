"""Versioned IFEval rule aggregation with an injectable official checker."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import JsonValue

from dimebench.datasets import GenerationRecord, SampleRecord
from dimebench.evaluators.base import EvaluationError, StandardEvaluator
from dimebench.evaluators.standard._text import prediction_text
from dimebench.schemas.result import ResultSpec


@dataclass(frozen=True)
class IFEvalCheckResult:
    """Per-instruction rule outcomes from an IFEval checker."""

    passed: tuple[bool, ...]
    instruction_ids: tuple[str, ...]


class IFEvalChecker(Protocol):
    """Boundary for the frozen official IFEval implementation."""

    checker_id: str
    revision: str

    def check(self, response: str, sample: GenerationRecord) -> IFEvalCheckResult:
        """Evaluate all instruction IDs attached to a sample."""


class BuiltinIFEvalChecker:
    """Deterministic checker for the two rules in the committed smoke fixture."""

    checker_id = "dimebench_ifeval_fixture_rules"
    revision = "1.0.0"

    def check(self, response: str, sample: GenerationRecord) -> IFEvalCheckResult:
        raw_ids = sample.metadata.get("instruction_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) for item in raw_ids
        ):
            raise EvaluationError("IFEval sample requires string instruction_ids")
        instruction_ids = tuple(cast(list[str], raw_ids))
        outcomes = []
        for instruction_id in instruction_ids:
            if instruction_id == "lowercase":
                letters = [character for character in response if character.isalpha()]
                outcomes.append(
                    bool(letters) and all(char.islower() for char in letters)
                )
            elif instruction_id == "exact_word_count":
                expected = self._word_count(sample)
                outcomes.append(len(response.split()) == expected)
            else:
                raise EvaluationError(
                    f"built-in IFEval checker does not support {instruction_id!r}; "
                    "inject the frozen official checker for full IFEval"
                )
        return IFEvalCheckResult(tuple(outcomes), instruction_ids)

    @staticmethod
    def _word_count(sample: GenerationRecord) -> int:
        explicit = sample.metadata.get("exact_word_count")
        if isinstance(explicit, int) and explicit >= 0:
            return explicit
        names = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        match = re.search(
            r"exactly\s+(\d+|[a-z]+)(?:\s+[a-z-]+){0,3}\s+words?",
            sample.instruction.lower(),
        )
        if match is None:
            raise EvaluationError("exact_word_count is absent from IFEval metadata")
        token = match.group(1)
        if token.isdigit():
            return int(token)
        if token in names:
            return names[token]
        raise EvaluationError(f"unsupported word-count token {token!r}")


class IFEvalEvaluator(StandardEvaluator):
    """Compute official strict-prompt IFEval success (all rules must pass)."""

    METRIC_ID = "instruction_following_rate"

    def __init__(self, checker: IFEvalChecker | None = None) -> None:
        self.checker = checker or BuiltinIFEvalChecker()

    def configuration(self) -> dict[str, JsonValue]:
        return {
            **super().configuration(),
            "aggregation": "strict_prompt",
            "checker_id": self.checker.checker_id,
            "checker_revision": self.checker.revision,
        }

    def _score(
        self,
        result: ResultSpec,
        sample: SampleRecord,
    ) -> tuple[float, dict[str, JsonValue]]:
        if not isinstance(sample, GenerationRecord):
            raise EvaluationError("IFEval requires a generation sample")
        checked = self.checker.check(prediction_text(result), sample)
        if not checked.passed:
            raise EvaluationError("IFEval sample has no instruction checks")
        return float(all(checked.passed)), {
            "instruction_ids": cast(JsonValue, list(checked.instruction_ids)),
            "instruction_passed": cast(JsonValue, list(checked.passed)),
            "aggregation": "strict_prompt",
        }


__all__ = [
    "BuiltinIFEvalChecker",
    "IFEvalCheckResult",
    "IFEvalChecker",
    "IFEvalEvaluator",
]
