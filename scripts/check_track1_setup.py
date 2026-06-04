#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import load_track_config, load_track_model_configs, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text
from src.utils.io import read_json


def main() -> None:
    track_cfg = load_track_config("track1_general")
    report = {
        "cuda": cuda_summary(),
        "nvidia_smi": nvidia_smi_text(),
        "models": [],
        "datasets": [],
    }

    for model_cfg in load_track_model_configs(track_cfg):
        local_dir = resolve_path(model_cfg["local_dir"])
        safetensors = sorted(local_dir.glob("*.safetensors"))
        bins = sorted(local_dir.glob("pytorch_model*.bin"))
        required = ["config.json", "tokenizer_config.json"]
        missing = [name for name in required if not (local_dir / name).exists()]
        has_tokenizer = (local_dir / "tokenizer.json").exists() or (local_dir / "vocab.json").exists()
        size_bytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file()) if local_dir.exists() else 0
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
                "ok": local_dir.exists()
                and not missing
                and has_tokenizer
                and (bool(safetensors) or bool(bins)),
            }
        )

    for dataset_name, dataset_cfg in track_cfg["datasets"].items():
        data_dir = resolve_path(Path("data/processed/track1_general") / dataset_name)
        samples_path = data_dir / "samples.jsonl"
        metadata_path = data_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        report["datasets"].append(
            {
                "name": dataset_name,
                "repo_id": dataset_cfg["repo_id"],
                "split": dataset_cfg["split"],
                "samples_path": str(samples_path),
                "metadata_path": str(metadata_path),
                "exists": samples_path.exists(),
                "num_rows": metadata.get("num_rows"),
                "ok": samples_path.exists() and bool(metadata.get("num_rows")),
            }
        )

    all_ok = (
        report["cuda"].get("cuda_available")
        and all(model["ok"] for model in report["models"])
        and all(dataset["ok"] for dataset in report["datasets"])
    )
    report["ok"] = bool(all_ok)
    out_path = resolve_path("logs/track1_general/setup_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_ok:
        raise SystemExit("Track 1 setup check failed. See logs/track1_general/setup_check.json")


if __name__ == "__main__":
    main()
