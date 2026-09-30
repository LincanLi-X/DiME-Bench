from __future__ import annotations

from pathlib import Path

import pytest

from dimebench.datasets import DatasetManifest, NormalizedDataset
from dimebench.datasets.base import DatasetError
from dimebench.datasets.preprocessing import preprocess_rows
from dimebench.datasets.split_lock import (
    apply_split_lock,
    freeze_split,
    load_split_lock,
    write_split_lock,
)


def _manifest() -> DatasetManifest:
    return DatasetManifest.model_validate(
        {
            "id": "fixture",
            "display_name": "Fixture",
            "version": "v1",
            "track": "track1_general",
            "record_kind": "generation",
            "license": "fixture",
            "homepage": "https://example.invalid",
            "citation": "fixture",
            "source": {"type": "local", "name_or_path": "fixture.jsonl"},
            "splits": ["test"],
            "sample_path": "samples/fixture.jsonl",
            "preprocessing": {
                "strategy": "generation",
                "field_map": {
                    "instruction": "question",
                    "references": "answer",
                },
            },
        }
    )


def _dataset(reverse: bool = False) -> NormalizedDataset:
    rows = [
        {"id": f"row-{index}", "question": f"Question {index}", "answer": str(index)}
        for index in range(8)
    ]
    if reverse:
        rows.reverse()
    return preprocess_rows(_manifest(), rows, "test")


def test_split_freezing_is_independent_of_raw_row_order() -> None:
    first, first_lock = freeze_split(
        _dataset(), _manifest().manifest_hash, seed=17, shuffle=True, sample_limit=4
    )
    second, second_lock = freeze_split(
        _dataset(reverse=True),
        _manifest().manifest_hash,
        seed=17,
        shuffle=True,
        sample_limit=4,
    )
    assert first.dataset_hash == second.dataset_hash
    assert first_lock.lock_hash == second_lock.lock_hash
    assert first_lock.sample_ids == second_lock.sample_ids


def test_split_lock_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    dataset = _dataset()
    frozen, lock = freeze_split(dataset, _manifest().manifest_hash)
    lock_path = tmp_path / "split-lock.json"
    write_split_lock(lock_path, lock)
    restored = load_split_lock(lock_path)
    assert apply_split_lock(dataset, restored).dataset_hash == frozen.dataset_hash

    changed = list(dataset.records)
    changed.pop()
    with pytest.raises(DatasetError, match="missing sample"):
        apply_split_lock(
            NormalizedDataset(dataset.dataset_id, dataset.split, changed),
            restored,
        )
