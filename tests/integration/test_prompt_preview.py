from __future__ import annotations

import json
from pathlib import Path

from dimebench.cli.main import main
from dimebench.datasets import prepare_suite
from dimebench.tasks import TaskCatalog, preview_prepared_suite

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _catalog() -> TaskCatalog:
    return TaskCatalog.from_roots(
        PROJECT_ROOT / "configs" / "tasks",
        PROJECT_ROOT / "dimebench" / "prompts",
    )


def test_full_suite_prompt_preview_is_deterministic(tmp_path: Path) -> None:
    prepare_suite(
        PROJECT_ROOT / "data" / "registry.yaml",
        "dime_bench_v1",
        tmp_path,
    )
    suite_root = tmp_path / "dime_bench_v1"
    first = preview_prepared_suite(_catalog(), suite_root)
    second = preview_prepared_suite(_catalog(), suite_root)
    assert len(first) == 30
    assert len({prompt.sample_id for prompt in first}) == 15
    assert [prompt.prompt_hash for prompt in first] == [
        prompt.prompt_hash for prompt in second
    ]


def test_task_preview_cli_writes_both_families(tmp_path: Path) -> None:
    prepare_suite(
        PROJECT_ROOT / "data" / "registry.yaml",
        "dime_bench_v1_smoke",
        tmp_path,
    )
    sample_path = tmp_path / "dime_bench_v1_smoke" / "mmlu_pro" / "test.jsonl"
    output = tmp_path / "previews.jsonl"
    assert (
        main(
            [
                "task",
                "preview",
                "--task",
                "mmlu_pro",
                "--samples",
                str(sample_path),
                "--limit",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    previews = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(previews) == 2
    assert {item["model_family"] for item in previews} == {
        "autoregressive",
        "diffusion",
    }
    assert all(len(item["prompt_hash"]) == 64 for item in previews)
