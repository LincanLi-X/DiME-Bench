"""Builders shared by CPU-only Step 11 mechanism tests."""

from __future__ import annotations

from dimebench.datasets import EditingRecord, InfillingRecord, TextSpan


def localized_edit_sample() -> EditingRecord:
    return EditingRecord(
        sample_id="fixture.test.localized-edit",
        dataset_id="fixture",
        split="test",
        source_id="localized-edit",
        source_text="Water freezes at 10 C.",
        instruction="Correct only the temperature.",
        reference_edit="Water freezes at 0 C.",
        target_spans=(TextSpan(start=17, end=21, text="10 C"),),
        non_target_spans=(
            TextSpan(start=0, end=17, text="Water freezes at "),
            TextSpan(start=21, end=22, text="."),
        ),
        minimum_edit_distance=1,
        edit_type="contradiction_repair",
        evidence=("Water freezes at 0 C under standard pressure.",),
    )


def over_edit_sample() -> EditingRecord:
    return EditingRecord(
        sample_id="fixture.test.over-edit",
        dataset_id="fixture",
        split="test",
        source_id="over-edit",
        source_text="a b c",
        instruction="Replace b with x.",
        reference_edit="a x c",
        minimum_edit_distance=1,
        edit_type="replacement",
    )


def infilling_sample(
    sample_id: str,
    *,
    span_length_bin: str = "short",
    boundary_density: str = "low",
) -> InfillingRecord:
    middle = "gold middle" if span_length_bin == "short" else "one two three four"
    return InfillingRecord(
        sample_id=sample_id,
        dataset_id="fixture",
        split="test",
        source_id=sample_id,
        prefix="left boundary",
        middle=middle,
        suffix="right boundary",
        domain="fixture",
        span_length=len(middle.split()),
        prefix_length=2,
        suffix_length=2,
        span_length_bin=span_length_bin,
        boundary_density=boundary_density,
    )


__all__ = ["infilling_sample", "localized_edit_sample", "over_edit_sample"]
