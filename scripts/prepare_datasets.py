#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data.hf_loader import load_raw_track1_dataset, normalize_track1_rows
from src.utils.config import load_track_config, resolve_path
from src.utils.io import ensure_dir, write_json, write_jsonl


def prepare_track1_dataset(dataset_name: str, limit: int | None = None) -> dict:
    track_cfg = load_track_config("track1_general")
    dataset_cfg = dict(track_cfg["datasets"][dataset_name])
    cache_dir = str(resolve_path("data/hf_cache"))
    t0 = time.perf_counter()
    raw = load_raw_track1_dataset(dataset_cfg, cache_dir=cache_dir)
    rows = normalize_track1_rows(dataset_name, raw)
    if limit is not None:
        rows = rows[:limit]
    out_dir = ensure_dir(resolve_path(Path("data/processed/track1_general") / dataset_name))
    write_jsonl(out_dir / "samples.jsonl", rows)
    metadata = {
        "dataset": dataset_name,
        "repo_id": dataset_cfg["repo_id"],
        "subset": dataset_cfg.get("subset"),
        "split": dataset_cfg["split"],
        "num_rows": len(rows),
        "runtime_s": time.perf_counter() - t0,
    }
    write_json(out_dir / "metadata.json", metadata)
    print(metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="track1_general")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.track != "track1_general":
        raise SystemExit("Only track1_general is implemented in prepare_datasets.py")
    track_cfg = load_track_config(args.track)
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    results = [prepare_track1_dataset(name, limit=args.limit) for name in dataset_names]
    write_json(resolve_path("logs/track1_general/dataset_prepare_summary.json"), results)


if __name__ == "__main__":
    main()
