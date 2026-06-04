#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data.track3_editing_loader import load_raw_track3_dataset, normalize_track3_rows
from src.utils.config import load_track_config, resolve_path
from src.utils.io import ensure_dir, write_json, write_jsonl


def prepare_track3_dataset(dataset_name: str, limit: int | None = None) -> dict:
    track_cfg = load_track_config("track3_editing")
    dataset_cfg = dict(track_cfg["datasets"][dataset_name])
    cache_dir = str(resolve_path("data/hf_cache"))
    t0 = time.perf_counter()
    raw = load_raw_track3_dataset(dataset_name, dataset_cfg, cache_dir=cache_dir)
    rows = normalize_track3_rows(dataset_name, raw, dataset_cfg)
    if limit is not None:
        rows = rows[:limit]
    out_dir = ensure_dir(resolve_path(Path("data/processed/track3_editing") / dataset_name))
    write_jsonl(out_dir / "samples.jsonl", rows)
    metadata = {
        "track": "track3_editing",
        "dataset": dataset_name,
        "display_name": dataset_cfg.get("display_name"),
        "task_type": dataset_cfg.get("task_type"),
        "num_rows": len(rows),
        "limit": limit,
        "runtime_s": time.perf_counter() - t0,
        "source": "synthetic" if dataset_name == "synthetic_contradiction_repair" else "local_or_hf",
    }
    write_json(out_dir / "metadata.json", metadata)
    print(metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track3_editing")
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    results = []
    errors = []
    for name in dataset_names:
        try:
            results.append(prepare_track3_dataset(name, limit=args.limit))
        except Exception as exc:
            errors.append({"dataset": name, "error": repr(exc)})
            if not args.allow_missing:
                raise
            print({"dataset": name, "status": "missing_or_failed", "error": repr(exc)})
    summary = {"results": results, "errors": errors}
    write_json(resolve_path("logs/track3_editing/dataset_prepare_summary.json"), summary)
    if errors and not args.allow_missing:
        raise SystemExit("Track 3 dataset preparation failed")


if __name__ == "__main__":
    main()
