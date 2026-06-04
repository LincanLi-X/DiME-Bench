from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_track_config(track: str = "track1_general") -> dict[str, Any]:
    return load_yaml(Path("configs/tracks") / f"{track}.yaml")


def load_model_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["_config_path"] = str(path)
    return cfg


def load_track_model_configs(track_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [load_model_config(path) for path in track_cfg["models"]]


def get_model_config(track_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_name_norm = model_name.lower()
    for cfg in load_track_model_configs(track_cfg):
        aliases = {
            cfg["name"].lower(),
            Path(cfg["_config_path"]).stem.lower(),
            cfg["repo_id"].split("/")[-1].lower(),
        }
        if model_name_norm in aliases:
            return cfg
    available = ", ".join(cfg["name"] for cfg in load_track_model_configs(track_cfg))
    raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")


def get_dataset_config(track_cfg: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    if dataset_name not in track_cfg["datasets"]:
        available = ", ".join(track_cfg["datasets"].keys())
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")
    cfg = dict(track_cfg["datasets"][dataset_name])
    cfg["name"] = dataset_name
    return cfg
