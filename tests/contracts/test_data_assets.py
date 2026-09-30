from __future__ import annotations

from pathlib import Path

from dimebench.datasets import load_data_registry, load_dataset_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def test_data_registry_assets_are_complete() -> None:
    registry = load_data_registry(DATA_ROOT / "registry.yaml")
    assert set(registry.suites["dime_bench_v1"].datasets) == set(registry.datasets)
    for relative_path in registry.datasets.values():
        manifest = load_dataset_manifest(DATA_ROOT / relative_path)
        assert (DATA_ROOT / manifest.sample_path).is_file()
        assert manifest.homepage.startswith("https://")
    assert (DATA_ROOT / "cards" / "README.md").is_file()
