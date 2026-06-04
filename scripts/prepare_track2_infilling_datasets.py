#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data.track2_infilling_loader import load_raw_track2_dataset, normalize_track2_rows
from src.utils.config import load_track_config, resolve_path
from src.utils.io import ensure_dir, write_json, write_jsonl


def prepare_track2_dataset(
    dataset_name: str,
    limit: int | None = None,
    max_source_rows: int | None = None,
) -> dict:
    track_cfg = load_track_config("track2_infilling")
    dataset_cfg = dict(track_cfg["datasets"][dataset_name])
    cache_dir = str(resolve_path("data/hf_cache"))
    t0 = time.perf_counter()
    raw = load_raw_track2_dataset(dataset_cfg, cache_dir=cache_dir)
    rows = normalize_track2_rows(
        dataset_name,
        raw,
        dataset_cfg,
        limit=limit,
        max_source_rows=max_source_rows,
    )
    out_dir = ensure_dir(resolve_path(Path("data/processed/track2_infilling") / dataset_name))
    write_jsonl(out_dir / "samples.jsonl", rows)
    metadata = {
        "track": "track2_infilling",
        "dataset": dataset_name,
        "repo_id": dataset_cfg["repo_id"],
        "subset": dataset_cfg.get("subset"),
        "split": dataset_cfg["split"],
        "num_rows": len(rows),
        "limit": limit,
        "max_source_rows": max_source_rows,
        "runtime_s": time.perf_counter() - t0,
        "schema": {
            "required": [
                "sample_id",
                "prefix",
                "suffix",
                "reference",
                "span_length",
                "context_density_bin",
                "span_length_bin",
            ]
        },
    }
    write_json(out_dir / "metadata.json", metadata)
    print(metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-source-rows", type=int, default=50000)
    args = parser.parse_args()

    track_cfg = load_track_config("track2_infilling")
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    results = [
        prepare_track2_dataset(name, limit=args.limit, max_source_rows=args.max_source_rows)
        for name in dataset_names
    ]
    write_json(resolve_path("logs/track2_infilling/dataset_prepare_summary.json"), results)


if __name__ == "__main__":
    main()
