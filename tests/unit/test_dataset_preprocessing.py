from __future__ import annotations

from dimebench.datasets.base import DatasetManifest
from dimebench.datasets.preprocessing import preprocess_rows


def test_raw_infilling_preprocessing_is_deterministic() -> None:
    manifest = DatasetManifest.model_validate(
        {
            "id": "fixture_infill",
            "display_name": "Fixture",
            "version": "v1",
            "track": "track2_infilling",
            "record_kind": "infilling",
            "license": "fixture",
            "homepage": "https://example.invalid",
            "citation": "fixture",
            "source": {"type": "local", "name_or_path": "fixture.jsonl"},
            "splits": ["test"],
            "sample_path": "samples/fixture.jsonl",
            "preprocessing": {
                "strategy": "infilling",
                "field_map": {"text": "document"},
                "parameters": {"domain": "test"},
            },
        }
    )
    rows = [
        {
            "id": "doc-1",
            "document": (
                "Alpha beta gamma delta epsilon zeta eta theta iota kappa "
                "lambda mu nu xi omicron."
            ),
        }
    ]
    first = preprocess_rows(manifest, rows, "test")
    second = preprocess_rows(manifest, rows, "test")
    assert first.dataset_hash == second.dataset_hash
    assert first.records[0].sample_id == second.records[0].sample_id
    assert first.records[0].kind == "infilling"


def test_editing_preprocessor_derives_diff_spans() -> None:
    manifest = DatasetManifest.model_validate(
        {
            "id": "fixture_edit",
            "display_name": "Fixture",
            "version": "v1",
            "track": "track3_editing",
            "record_kind": "editing",
            "license": "fixture",
            "homepage": "https://example.invalid",
            "citation": "fixture",
            "source": {"type": "local", "name_or_path": "fixture.jsonl"},
            "splits": ["test"],
            "sample_path": "samples/fixture.jsonl",
            "preprocessing": {
                "strategy": "editing",
                "field_map": {
                    "source_text": "source",
                    "instruction": "instruction",
                    "reference_edit": "reference",
                },
            },
        }
    )
    dataset = preprocess_rows(
        manifest,
        [
            {
                "id": "edit-1",
                "source": "Bird fly.",
                "instruction": "Correct grammar.",
                "reference": "Birds fly.",
            }
        ],
        "test",
    )
    record = dataset.records[0]
    assert record.kind == "editing"
    assert record.minimum_edit_distance == 1
    assert record.target_spans
    assert record.non_target_spans


def test_editing_preprocessor_uses_token_not_character_distance() -> None:
    manifest = DatasetManifest.model_validate(
        {
            "id": "fixture_edit",
            "display_name": "Fixture",
            "version": "v1",
            "track": "track3_editing",
            "record_kind": "editing",
            "license": "fixture",
            "homepage": "https://example.invalid",
            "citation": "fixture",
            "source": {"type": "local", "name_or_path": "fixture.jsonl"},
            "splits": ["test"],
            "sample_path": "samples/fixture.jsonl",
            "preprocessing": {
                "strategy": "editing",
                "field_map": {
                    "source_text": "source",
                    "instruction": "instruction",
                    "reference_edit": "reference",
                },
            },
        }
    )
    dataset = preprocess_rows(
        manifest,
        [
            {
                "id": "edit-2",
                "source": "a b c",
                "instruction": "Replace two tokens.",
                "reference": "a x y",
            }
        ],
        "test",
    )
    record = dataset.records[0]
    assert record.kind == "editing"
    assert record.minimum_edit_distance == 2


def test_reasoning_preprocessor_handles_winogrande_fields() -> None:
    manifest = DatasetManifest.model_validate(
        {
            "id": "fixture_reasoning",
            "display_name": "Fixture",
            "version": "v1",
            "track": "track4_reasoning",
            "record_kind": "reasoning",
            "license": "fixture",
            "homepage": "https://example.invalid",
            "citation": "fixture",
            "source": {"type": "local", "name_or_path": "fixture.jsonl"},
            "splits": ["validation"],
            "sample_path": "samples/fixture.jsonl",
            "preprocessing": {
                "strategy": "reasoning",
                "field_map": {
                    "problem": "sentence",
                    "answer": "answer",
                    "answer_index": "answer",
                },
                "parameters": {
                    "reasoning_type": "planning",
                    "choice_fields": ["option1", "option2"],
                    "answer_index_offset": 1,
                },
            },
        }
    )
    dataset = preprocess_rows(
        manifest,
        [
            {
                "id": "reasoning-1",
                "sentence": "The trophy does not fit because _ is too large.",
                "option1": "the trophy",
                "option2": "the suitcase",
                "answer": "1",
            }
        ],
        "validation",
    )
    record = dataset.records[0]
    assert record.kind == "reasoning"
    assert record.choices == ("the trophy", "the suitcase")
    assert record.answer_index == 0
    assert record.answer == "the trophy"


def test_infilling_preprocessor_filters_short_rows() -> None:
    manifest = DatasetManifest.model_validate(
        {
            "id": "fixture_filter",
            "display_name": "Fixture",
            "version": "v1",
            "track": "track2_infilling",
            "record_kind": "infilling",
            "license": "fixture",
            "homepage": "https://example.invalid",
            "citation": "fixture",
            "source": {"type": "local", "name_or_path": "fixture.jsonl"},
            "splits": ["test"],
            "sample_path": "samples/fixture.jsonl",
            "preprocessing": {
                "strategy": "infilling",
                "field_map": {"text": "text"},
                "parameters": {"skip_short": True, "min_words": 9},
            },
        }
    )
    dataset = preprocess_rows(
        manifest,
        [
            {"id": "heading", "text": "= Short heading ="},
            {
                "id": "body",
                "text": "One two three four five six seven eight nine ten eleven.",
            },
        ],
        "test",
    )
    assert len(dataset.records) == 1
    assert dataset.records[0].source_id == "body"
