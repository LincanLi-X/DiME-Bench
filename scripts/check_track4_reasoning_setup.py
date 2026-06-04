#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tracks.track4_reasoning import load_track4_model_configs
from src.utils.config import load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text
from src.utils.io import read_json


TRACK = "track4_reasoning"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--allow-missing-datasets", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config(TRACK)
    report = {
        "track": TRACK,
        "cuda": cuda_summary(),
        "nvidia_smi": nvidia_smi_text(),
        "models": [],
        "datasets": [],
    }

    for model_cfg in load_track4_model_configs(track_cfg):
        local_dir = resolve_path(model_cfg["local_dir"])
        safetensors = sorted(local_dir.glob("*.safetensors")) if local_dir.exists() else []
        bins = sorted(local_dir.glob("pytorch_model*.bin")) if local_dir.exists() else []
        required = ["config.json", "tokenizer_config.json"]
        missing = [name for name in required if not (local_dir / name).exists()]
        has_tokenizer = (local_dir / "tokenizer.json").exists() or (local_dir / "vocab.json").exists()
        size_bytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file()) if local_dir.exists() else 0
        ok = local_dir.exists() and not missing and has_tokenizer and (bool(safetensors) or bool(bins))
        report["models"].append(
            {
                "name": model_cfg["name"],
                "repo_id": model_cfg["repo_id"],
                "local_dir": str(local_dir),
                "exists": local_dir.exists(),
                "size_gb": round(size_bytes / (1024**3), 2),
                "num_safetensors": len(safetensors),
                "num_pytorch_bins": len(bins),
                "has_tokenizer": has_tokenizer,
                "missing_required_files": missing,
                "ok": ok,
            }
        )

    for dataset_name, dataset_cfg in track_cfg["datasets"].items():
        data_dir = resolve_path(Path("data/processed/track4_reasoning") / dataset_name)
        samples_path = data_dir / "samples.jsonl"
        metadata_path = data_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        report["datasets"].append(
            {
                "name": dataset_name,
                "repo_id": dataset_cfg["repo_id"],
                "split": dataset_cfg["split"],
                "reasoning_type": dataset_cfg["reasoning_type"],
                "samples_path": str(samples_path),
                "metadata_path": str(metadata_path),
                "exists": samples_path.exists(),
                "num_rows": metadata.get("num_rows"),
                "ok": samples_path.exists() and bool(metadata.get("num_rows")),
            }
        )

    models_ok = all(model["ok"] for model in report["models"]) or args.allow_missing_models
    datasets_ok = all(dataset["ok"] for dataset in report["datasets"]) or args.allow_missing_datasets
    report["ok"] = bool(models_ok and datasets_ok)
    out_path = resolve_path("logs/track4_reasoning/setup_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("Track 4 setup check failed. See logs/track4_reasoning/setup_check.json")


if __name__ == "__main__":
    main()
