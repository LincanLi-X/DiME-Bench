from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dimebench.datasets import (
    EditingRecord,
    InfillingRecord,
    MultipleChoiceRecord,
    TextSpan,
    load_data_registry,
    load_dataset_manifest,
)
from dimebench.datasets.download import load_bundled_sample_rows
from dimebench.datasets.preprocessing import preprocess_rows
from dimebench.datasets.records import expected_sample_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def test_all_registered_manifests_and_sample_records_are_valid() -> None:
    registry = load_data_registry(DATA_ROOT / "registry.yaml")
    assert len(registry.datasets) == 15
    kinds: set[str] = set()
    for dataset_id, relative_path in registry.datasets.items():
        manifest = load_dataset_manifest(DATA_ROOT / relative_path)
        assert manifest.id == dataset_id
        for split in manifest.splits:
            rows = load_bundled_sample_rows(manifest, DATA_ROOT, split)
            dataset = preprocess_rows(manifest, rows, split)
            assert len(dataset) == 1
            record = dataset.records[0]
            assert record.kind == manifest.record_kind
            assert expected_sample_id(record) == record.sample_id
            assert len(record.record_hash) == 64
            kinds.add(record.kind)
    assert kinds == {
        "multiple_choice",
        "generation",
        "infilling",
        "editing",
        "reasoning",
    }


def test_multiple_choice_rejects_out_of_range_answer() -> None:
    with pytest.raises(ValidationError, match="available choice"):
        MultipleChoiceRecord(
            sample_id="dataset.test.abc",
            dataset_id="dataset",
            split="test",
            source_id="one",
            question="Question?",
            choices=("A", "B"),
            answer_index=2,
        )


def test_infilling_lengths_must_match_content() -> None:
    with pytest.raises(ValidationError, match="token lengths"):
        InfillingRecord(
            sample_id="dataset.test.abc",
            dataset_id="dataset",
            split="test",
            source_id="one",
            prefix="one two",
            middle="three",
            suffix="four five",
            domain="fixture",
            span_length=2,
            prefix_length=2,
            suffix_length=2,
            span_length_bin="short",
            boundary_density="low",
        )


def test_editing_spans_must_be_disjoint_and_bounded() -> None:
    with pytest.raises(ValidationError, match="cannot overlap"):
        EditingRecord(
            sample_id="dataset.test.abc",
            dataset_id="dataset",
            split="test",
            source_id="one",
            source_text="abcdef",
            instruction="edit",
            reference_edit="abXdef",
            target_spans=(TextSpan(start=1, end=4),),
            non_target_spans=(TextSpan(start=3, end=6),),
            minimum_edit_distance=1,
            edit_type="fixture",
        )
