from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dimebench.datasets import load_data_registry
from dimebench.tasks import TaskCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_task_assets_cover_the_frozen_benchmark() -> None:
    catalog = TaskCatalog.from_roots(
        PROJECT_ROOT / "configs" / "tasks",
        PROJECT_ROOT / "dimebench" / "prompts",
    )
    registry = load_data_registry(PROJECT_ROOT / "data" / "registry.yaml")
    benchmark = json.loads(
        (PROJECT_ROOT / "specification" / "benchmark-v1.json").read_text()
    )
    frozen_tasks = {
        task["id"] for track in benchmark["tracks"] for task in track["tasks"]
    }
    assert set(catalog.ids) == set(registry.suites["dime_bench_v1"].datasets)
    assert set(catalog.ids) == frozen_tasks
    track_counts = Counter(catalog.get(task_id).spec.track for task_id in catalog.ids)
    assert track_counts == {
        "track1_general": 5,
        "track2_infilling": 3,
        "track3_editing": 3,
        "track4_reasoning": 4,
    }


def test_task_primary_metrics_match_frozen_specification() -> None:
    catalog = TaskCatalog.from_roots(
        PROJECT_ROOT / "configs" / "tasks",
        PROJECT_ROOT / "dimebench" / "prompts",
    )
    benchmark = json.loads(
        (PROJECT_ROOT / "specification" / "benchmark-v1.json").read_text()
    )
    expected = {
        task["id"]: task["primary_metric"]
        for track in benchmark["tracks"]
        for task in track["tasks"]
    }
    observed = {
        task_id: catalog.get(task_id).spec.primary_metric for task_id in catalog.ids
    }
    assert observed == expected
