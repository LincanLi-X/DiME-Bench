from __future__ import annotations

import json
from pathlib import Path

from dimebench.config import load_config, load_model_config
from dimebench.datasets import NormalizedDataset, prepare_suite
from dimebench.models import (
    MODEL_ADAPTER_REGISTRY,
    AdapterState,
    ModelAdapter,
    create_adapter,
)
from dimebench.models.registry import load_builtin_adapters
from dimebench.tasks import BaseTask, TaskCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_every_builtin_adapter_satisfies_the_common_static_contract() -> None:
    load_builtin_adapters()
    assert set(MODEL_ADAPTER_REGISTRY.names()) == {
        "huggingface",
        "llada",
        "mock",
    }
    model_specs = (
        load_model_config(
            PROJECT_ROOT / "configs/models/autoregressive/llama3_8b_instruct.yaml"
        ),
        load_model_config(
            PROJECT_ROOT / "configs/models/diffusion/llada_8b_instruct.yaml"
        ),
        load_config(PROJECT_ROOT / "configs/smoke.yaml").model,
    )
    for spec in model_specs:
        adapter = create_adapter(spec)
        assert isinstance(adapter, ModelAdapter)
        assert adapter.ADAPTER_NAME == spec.adapter
        assert adapter.state is AdapterState.CREATED
        assert all(adapter.supports(capability) for capability in spec.capabilities)
        assert not type(adapter).__abstractmethods__
        adapter.close()
        assert adapter.state is AdapterState.CLOSED


def test_every_tracked_task_renders_through_the_common_interface(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "prepared"
    manifest = prepare_suite(
        PROJECT_ROOT / "data/registry.yaml",
        "dime_bench_v1",
        prepared_root,
        mode="sample",
    )
    suite_root = prepared_root / "dime_bench_v1"
    data_paths = {
        entry["dataset_id"]: suite_root / str(entry["data_path"])
        for entry in manifest["datasets"]
    }
    catalog = TaskCatalog.from_roots(
        PROJECT_ROOT / "configs/tasks",
        PROJECT_ROOT / "dimebench/prompts",
    )
    assert len(catalog.ids) == 15

    for task_id in catalog.ids:
        task = catalog.get(task_id)
        assert isinstance(task, BaseTask)
        dataset = NormalizedDataset.read_jsonl(data_paths[task.spec.dataset_id])
        sample = dataset.records[0]
        rendered = {
            family: task.render(sample, family)
            for family in ("autoregressive", "diffusion")
        }
        assert all(item.text.strip() for item in rendered.values())
        assert all(item.task_id == task_id for item in rendered.values())
        assert all(item.sample_id == sample.sample_id for item in rendered.values())
        assert len({item.prompt_hash for item in rendered.values()}) == 2

    suite_manifest = json.loads(
        (suite_root / "suite-manifest.json").read_text(encoding="utf-8")
    )
    assert len(suite_manifest["datasets"]) == len(catalog.ids)
