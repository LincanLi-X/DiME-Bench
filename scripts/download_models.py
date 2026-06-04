#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from src.utils.config import load_track_config, load_track_model_configs, resolve_path
from src.utils.io import ensure_dir, write_json


def download_model(model_cfg: dict, hf_token: str | None = None, resume: bool = True) -> dict:
    local_dir = resolve_path(model_cfg["local_dir"])
    ensure_dir(local_dir)
    t0 = time.perf_counter()
    try:
        path = snapshot_download(
            repo_id=model_cfg["repo_id"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=resume,
        )
        result = {
            "model": model_cfg["name"],
            "repo_id": model_cfg["repo_id"],
            "local_dir": str(local_dir),
            "snapshot_path": path,
            "status": "ok",
            "runtime_s": time.perf_counter() - t0,
        }
    except Exception as exc:
        result = {
            "model": model_cfg["name"],
            "repo_id": model_cfg["repo_id"],
            "local_dir": str(local_dir),
            "status": "failed",
            "error": repr(exc),
            "runtime_s": time.perf_counter() - t0,
            "requires_auth": bool(model_cfg.get("requires_auth")),
        }
    ensure_dir("logs/downloads")
    write_json(Path("logs/downloads") / f"{model_cfg['name'].replace('/', '_')}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="track1_general")
    parser.add_argument("--model", default="all", help="Model display name, config stem, or all.")
    parser.add_argument("--hf-token", default=None, help="Optional HF token. If omitted, uses cached login/env.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config(args.track)
    model_cfgs = load_track_model_configs(track_cfg)
    if args.model != "all":
        key = args.model.lower()
        model_cfgs = [
            cfg
            for cfg in model_cfgs
            if key in {cfg["name"].lower(), Path(cfg["_config_path"]).stem.lower(), cfg["repo_id"].split("/")[-1].lower()}
        ]
        if not model_cfgs:
            raise SystemExit(f"No model matched {args.model}")

    results = [download_model(cfg, hf_token=args.hf_token, resume=not args.no_resume) for cfg in model_cfgs]
    write_json(Path("logs/downloads") / f"{args.track}_summary.json", results)
    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} model downloads failed. See logs/downloads/*.json")


if __name__ == "__main__":
    main()
