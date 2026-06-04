#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data.track6_efficiency_loader import load_and_normalize_track6_dataset, select_track6_samples
from src.utils.config import get_dataset_config, load_track_config, resolve_path
from src.utils.io import ensure_dir, write_json, write_jsonl


TRACK = "track6_efficiency"


def prepare_track6_dataset(
    dataset_name: str,
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    allow_synthetic_fallback: bool = False,
) -> dict:
    track_cfg = load_track_config(TRACK)
    dataset_cfg = get_dataset_config(track_cfg, dataset_name)
    t0 = time.perf_counter()
    rows, source_info = load_and_normalize_track6_dataset(
        dataset_name,
        dataset_cfg,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    selected, selection = select_track6_samples(
        rows,
        dataset_cfg,
        sample_size=sample_size,
        sample_seed=sample_seed,
        limit=limit,
    )
    out_dir = ensure_dir(resolve_path(Path("data/processed/track6_efficiency") / dataset_name))
    write_jsonl(out_dir / "samples.jsonl", selected)
    metadata = {
        "track": TRACK,
        "dataset": dataset_name,
        "task_type": dataset_cfg["task_type"],
        "num_rows": len(selected),
        "selection": selection,
        "source_info": source_info,
        "runtime_s": time.perf_counter() - t0,
        "result_status": "synthetic_fallback"
        if source_info.get("used_synthetic_fallback")
        else "ok",
    }
    write_json(out_dir / "metadata.json", metadata)
    print(metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config(TRACK)
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    results = [
        prepare_track6_dataset(
            name,
            limit=args.limit,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            allow_synthetic_fallback=args.allow_synthetic_fallback,
        )
        for name in dataset_names
    ]
    write_json(resolve_path("logs/track6_efficiency/dataset_prepare_summary.json"), results)


if __name__ == "__main__":
    main()
