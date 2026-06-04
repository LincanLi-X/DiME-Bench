#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tracks.track6_efficiency import load_track6_model_configs
from src.utils.config import load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text
from src.utils.io import read_json, write_json


TRACK = "track6_efficiency"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--allow-missing-upstream", action="store_true")
    parser.add_argument("--allow-no-cuda", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config(TRACK)
    report = {
        "track": TRACK,
        "cuda": cuda_summary(),
        "nvidia_smi": nvidia_smi_text(),
        "models": [],
        "processed_datasets": [],
        "upstream_sources": [],
    }

    for model_cfg in load_track6_model_configs(track_cfg):
        local_dir = resolve_path(model_cfg["local_dir"])
        safetensors = sorted(local_dir.glob("*.safetensors"))
        bins = sorted(local_dir.glob("pytorch_model*.bin"))
        has_tokenizer = (local_dir / "tokenizer.json").exists() or (local_dir / "vocab.json").exists()
        size_bytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file()) if local_dir.exists() else 0
        ok = local_dir.exists() and has_tokenizer and (bool(safetensors) or bool(bins))
        report["models"].append(
            {
                "name": model_cfg["name"],
                "family": model_cfg.get("family"),
                "repo_id": model_cfg["repo_id"],
                "local_dir": str(local_dir),
                "exists": local_dir.exists(),
                "size_gb": round(size_bytes / (1024**3), 2),
                "num_safetensors": len(safetensors),
                "num_pytorch_bins": len(bins),
                "has_tokenizer": has_tokenizer,
                "ok": ok,
            }
        )

    for dataset_name, dataset_cfg in track_cfg["datasets"].items():
        processed_dir = resolve_path(Path("data/processed/track6_efficiency") / dataset_name)
        samples_path = processed_dir / "samples.jsonl"
        metadata_path = processed_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        report["processed_datasets"].append(
            {
                "name": dataset_name,
                "samples_path": str(samples_path),
                "metadata_path": str(metadata_path),
                "exists": samples_path.exists(),
                "num_rows": metadata.get("num_rows"),
                "result_status": metadata.get("result_status"),
                "ok": samples_path.exists() and bool(metadata.get("num_rows")),
            }
        )
        for pattern in dataset_cfg.get("upstream_paths", []):
            matches = sorted(resolve_path(".").glob(str(Path(pattern)))) if "**" not in pattern else []
            report["upstream_sources"].append(
                {
                    "dataset": dataset_name,
                    "pattern": pattern,
                    "note": "recursive glob checked during preparation" if "**" in pattern else "",
                    "matches": [str(path) for path in matches[:10]],
                    "num_matches_listed": len(matches),
                }
            )

    models_ok = all(model["ok"] for model in report["models"]) or args.allow_missing_models
    data_ok = all(dataset["ok"] for dataset in report["processed_datasets"]) or args.allow_missing_upstream
    cuda_ok = bool(report["cuda"].get("cuda_available")) or args.allow_no_cuda
    report["ok"] = bool(cuda_ok and models_ok and data_ok)
    report["allowed_missing_models"] = args.allow_missing_models
    report["allowed_missing_upstream"] = args.allow_missing_upstream
    report["allowed_no_cuda"] = args.allow_no_cuda
    out_path = resolve_path("logs/track6_efficiency/setup_check.json")
    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("Track 6 setup check failed. See logs/track6_efficiency/setup_check.json")


if __name__ == "__main__":
    main()
