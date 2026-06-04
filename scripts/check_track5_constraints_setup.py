#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tracks.track5_constraints import load_track5_model_configs, safe_name
from src.utils.config import load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text
from src.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--allow-missing-datasets", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track5_constraints")
    report = {
        "track": "track5_constraints",
        "cuda": cuda_summary(),
        "nvidia_smi": nvidia_smi_text(),
        "models": [],
        "datasets": [],
        "prompts": [],
        "output_dirs": {},
    }

    for prompt_key, dataset_cfg in track_cfg["datasets"].items():
        path = resolve_path(dataset_cfg["prompt_template"])
        report["prompts"].append({"dataset": prompt_key, "path": str(path), "exists": path.exists()})

    for model_cfg in load_track5_model_configs(track_cfg):
        local_dir = resolve_path(model_cfg["local_dir"])
        safetensors = sorted(local_dir.glob("*.safetensors")) if local_dir.exists() else []
        bins = sorted(local_dir.glob("pytorch_model*.bin")) if local_dir.exists() else []
        has_tokenizer = (
            (local_dir / "tokenizer.json").exists()
            or (local_dir / "vocab.json").exists()
            or (local_dir / "tokenizer.model").exists()
        )
        missing_required = [name for name in ["config.json"] if not (local_dir / name).exists()]
        ok = local_dir.exists() and not missing_required and has_tokenizer and (bool(safetensors) or bool(bins))
        report["models"].append(
            {
                "name": model_cfg["name"],
                "safe_name": safe_name(model_cfg["name"]),
                "repo_id": model_cfg.get("repo_id"),
                "local_dir": str(local_dir),
                "exists": local_dir.exists(),
                "num_safetensors": len(safetensors),
                "num_pytorch_bins": len(bins),
                "has_tokenizer": has_tokenizer,
                "missing_required_files": missing_required,
                "ok": ok,
            }
        )

    for dataset_name, dataset_cfg in track_cfg["datasets"].items():
        data_dir = resolve_path(Path("data/processed/track5_constraints") / dataset_name)
        samples_path = data_dir / "samples.jsonl"
        metadata_path = data_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        report["datasets"].append(
            {
                "name": dataset_name,
                "task_type": dataset_cfg["task_type"],
                "samples_path": str(samples_path),
                "metadata_path": str(metadata_path),
                "exists": samples_path.exists(),
                "num_rows": metadata.get("num_rows"),
                "fixture": metadata.get("fixture"),
                "ok": samples_path.exists() and bool(metadata.get("num_rows")),
            }
        )

    for key in ["output_dir", "log_dir", "metric_dir", "trajectory_dir"]:
        path = ensure_dir(resolve_path(track_cfg["reporting"][key]))
        report["output_dirs"][key] = str(path)

    models_ok = all(model["ok"] for model in report["models"]) or args.allow_missing_models
    datasets_ok = all(dataset["ok"] for dataset in report["datasets"]) or args.allow_missing_datasets
    prompts_ok = all(prompt["exists"] for prompt in report["prompts"])
    report["ok"] = bool(models_ok and datasets_ok and prompts_ok)

    out_path = resolve_path("logs/track5_constraints/setup_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("Track 5 setup check failed. See logs/track5_constraints/setup_check.json")


if __name__ == "__main__":
    main()
