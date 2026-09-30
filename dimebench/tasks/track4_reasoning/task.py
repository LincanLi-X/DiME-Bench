"""Prompt field extraction for deterministic direct-answer reasoning."""

from __future__ import annotations

from dimebench.datasets.records import ReasoningRecord, SampleRecord
from dimebench.tasks.base import BaseTask, TaskError


class ReasoningTask(BaseTask):
    """Render chain and planning inputs without requesting hidden reasoning."""

    def prompt_fields(self, sample: SampleRecord) -> dict[str, str]:
        if not isinstance(sample, ReasoningRecord):
            raise TaskError(f"Track 4 cannot render sample kind {sample.kind!r}")
        choices = "\n".join(
            f"{chr(ord('A') + index)}. {choice}"
            for index, choice in enumerate(sample.choices)
        )
        return {
            "problem": sample.problem,
            "choices": choices or "None provided.",
        }


__all__ = ["ReasoningTask"]
