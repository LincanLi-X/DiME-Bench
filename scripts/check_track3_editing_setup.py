#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tracks.track3_editing import load_track3_model_configs
from src.utils.config import load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track3_editing")
    report = {
        "track": "track3_editing",
        "cuda": cuda_summary(),
        "nvidia_smi": nvidia_smi_text(),
        "models": [],
        "datasets": [],
        "prompts": [],
    }
    for model_cfg in load_track3_model_configs(track_cfg):
        local_dir = resolve_path(model_cfg["local_dir"])
        safetensors = sorted(local_dir.glob("*.safetensors")) if local_dir.exists() else []
        bins = sorted(local_dir.glob("pytorch_model*.bin")) if local_dir.exists() else []
        has_tokenizer = (
            (local_dir / "tokenizer.json").exists()
            or (local_dir / "vocab.json").exists()
            or (local_dir / "tokenizer.model").exists()
        )
        missing_required = [name for name in ["config.json"] if not (local_dir / name).exists()]
        size_bytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file()) if local_dir.exists() else 0
        report["models"].append(
            {
                "name": model_cfg["name"],
                "family": model_cfg.get("family"),
                "repo_id": model_cfg.get("repo_id"),
                "local_dir": str(local_dir),
                "exists": local_dir.exists(),
                "size_gb": round(size_bytes / (1024**3), 2),
                "num_safetensors": len(safetensors),
                "num_pytorch_bins": len(bins),
                "has_tokenizer": has_tokenizer,
                "missing_required_files": missing_required,
                "declared_status": model_cfg.get("status"),
                "ok": local_dir.exists()
                and not missing_required
                and has_tokenizer
                and (bool(safetensors) or bool(bins)),
            }
        )

    for dataset_name, dataset_cfg in track_cfg["datasets"].items():
        data_dir = resolve_path(Path("data/processed/track3_editing") / dataset_name)
        samples_path = data_dir / "samples.jsonl"
        metadata_path = data_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        report["datasets"].append(
            {
                "name": dataset_name,
                "display_name": dataset_cfg.get("display_name"),
                "samples_path": str(samples_path),
                "metadata_path": str(metadata_path),
                "exists": samples_path.exists(),
                "num_rows": metadata.get("num_rows"),
                "ok": samples_path.exists() and bool(metadata.get("num_rows")),
            }
        )
        for key in ["prompt_template", "target_marked_prompt_template"]:
            if dataset_cfg.get(key):
                prompt_path = resolve_path(dataset_cfg[key])
                report["prompts"].append({"dataset": dataset_name, "path": str(prompt_path), "ok": prompt_path.exists()})

    cuda_ok = bool(report["cuda"].get("cuda_available")) or args.no_require_cuda
    report["ok"] = (
        cuda_ok
        and all(model["ok"] for model in report["models"])
        and all(dataset["ok"] for dataset in report["datasets"])
        and all(prompt["ok"] for prompt in report["prompts"])
    )
    out_path = resolve_path("logs/track3_editing/setup_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"] and not args.allow_missing:
        raise SystemExit("Track 3 setup check failed. See logs/track3_editing/setup_check.json")


if __name__ == "__main__":
    main()
