#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tracks.track2_infilling import load_track2_model_configs
from src.utils.config import load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--allow-no-cuda", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track2_infilling")
    report = {
        "track": "track2_infilling",
        "cuda": cuda_summary(),
        "nvidia_smi": nvidia_smi_text(),
        "models": [],
        "datasets": [],
        "prompts": [],
    }

    for model_cfg in load_track2_model_configs(track_cfg):
        local_dir = resolve_path(model_cfg["local_dir"])
        safetensors = sorted(local_dir.glob("*.safetensors"))
        bins = sorted(local_dir.glob("pytorch_model*.bin"))
        missing = [name for name in ["config.json"] if not (local_dir / name).exists()]
        has_tokenizer = (
            (local_dir / "tokenizer.json").exists()
            or (local_dir / "tokenizer.model").exists()
            or (local_dir / "vocab.json").exists()
        )
        size_bytes = sum(path.stat().st_size for path in local_dir.rglob("*") if path.is_file()) if local_dir.exists() else 0
        ok = local_dir.exists() and not missing and has_tokenizer and (bool(safetensors) or bool(bins))
        report["models"].append(
            {
                "name": model_cfg["name"],
                "family": model_cfg.get("family"),
                "repo_id": model_cfg["repo_id"],
                "local_dir": str(local_dir),
                "config_path": model_cfg.get("_config_path"),
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
        data_dir = resolve_path(Path("data/processed/track2_infilling") / dataset_name)
        samples_path = data_dir / "samples.jsonl"
        metadata_path = data_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        report["datasets"].append(
            {
                "name": dataset_name,
                "repo_id": dataset_cfg["repo_id"],
                "subset": dataset_cfg.get("subset"),
                "split": dataset_cfg["split"],
                "samples_path": str(samples_path),
                "metadata_path": str(metadata_path),
                "exists": samples_path.exists(),
                "num_rows": metadata.get("num_rows"),
                "ok": samples_path.exists() and bool(metadata.get("num_rows")),
            }
        )

    for name, path in track_cfg.get("prompt_templates", {}).items():
        resolved = resolve_path(path)
        report["prompts"].append({"name": name, "path": str(resolved), "exists": resolved.exists(), "ok": resolved.exists()})

    models_ok = all(model["ok"] for model in report["models"]) or args.allow_missing_models
    datasets_ok = all(dataset["ok"] for dataset in report["datasets"]) or args.allow_missing_datasets
    cuda_ok = bool(report["cuda"].get("cuda_available")) or args.allow_no_cuda
    prompts_ok = all(prompt["ok"] for prompt in report["prompts"])
    report["ok"] = bool(models_ok and datasets_ok and cuda_ok and prompts_ok)
    report["allowances"] = {
        "allow_missing_models": args.allow_missing_models,
        "allow_missing_datasets": args.allow_missing_datasets,
        "allow_no_cuda": args.allow_no_cuda,
    }

    out_path = resolve_path("logs/track2_infilling/setup_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("Track 2 setup check failed. See logs/track2_infilling/setup_check.json")


if __name__ == "__main__":
    main()
