"""Prompt field extraction for instruction-only full-text editing."""

from __future__ import annotations

from dimebench.datasets.records import EditingRecord, SampleRecord
from dimebench.tasks.base import BaseTask, TaskError


class EditingTask(BaseTask):
    """Render source, instruction, and public evidence without gold edits."""

    def prompt_fields(self, sample: SampleRecord) -> dict[str, str]:
        if not isinstance(sample, EditingRecord):
            raise TaskError(f"Track 3 cannot render sample kind {sample.kind!r}")
        return {
            "instruction": sample.instruction,
            "source_text": sample.source_text,
            "evidence": "\n".join(sample.evidence) or "None provided.",
        }


__all__ = ["EditingTask"]
