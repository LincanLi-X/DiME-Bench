from __future__ import annotations

from pathlib import Path

from dimebench.datasets import load_dataset_manifest
from dimebench.datasets.download import acquire_full_source, read_jsonl_rows
from dimebench.datasets.preprocessing import preprocess_rows
from dimebench.datasets.synthetic_repair import generate_synthetic_repair_rows
from dimebench.tracks import load_step12_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_step12_matrix_is_exactly_four_tracks_n25_nfe16() -> None:
    matrix = load_step12_matrix(
        PROJECT_ROOT / "configs/experiments/step12_minimal.yaml",
        PROJECT_ROOT,
    )

    assert matrix.sample_count == 25
    assert matrix.dllm_nfes == 16
    assert {case.track for case in matrix.tracks} == {
        "track1_general",
        "track2_infilling",
        "track3_editing",
        "track4_reasoning",
    }


def test_every_diffusion_budget_is_valid_for_sixteen_nfes() -> None:
    matrix = load_step12_matrix(
        PROJECT_ROOT / "configs/experiments/step12_minimal.yaml",
        PROJECT_ROOT,
    )

    for case in matrix.tracks:
        blocks = case.max_output_tokens // case.dllm_block_length
        assert case.max_output_tokens % case.dllm_block_length == 0
        assert matrix.dllm_nfes % blocks == 0


def test_synthetic_repair_full_builder_has_at_least_25_stable_rows(
    tmp_path: Path,
) -> None:
    manifest = load_dataset_manifest(
        PROJECT_ROOT / "data/manifests/synthetic_repair.yaml"
    )
    first = tuple(generate_synthetic_repair_rows())
    second = tuple(generate_synthetic_repair_rows())
    assert first == second
    assert len(first) >= 25

    source = acquire_full_source(manifest, "test", tmp_path)
    normalized = preprocess_rows(manifest, read_jsonl_rows(source), "test")
    assert len(normalized) >= 25
    assert len({record.sample_id for record in normalized}) == len(normalized)
    assert all(record.kind == "editing" for record in normalized)


def test_gsm8k_general_keeps_only_the_final_numeric_reference() -> None:
    manifest = load_dataset_manifest(PROJECT_ROOT / "data/manifests/gsm8k_general.yaml")
    normalized = preprocess_rows(
        manifest,
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "Compute the sum. 40 + 2 = 42.\n#### 42",
            }
        ],
        "test",
    )

    assert normalized.records[0].references == ("42",)
