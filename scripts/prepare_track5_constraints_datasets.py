#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data.track5_constraints_loader import (
    build_mt_bench_fixtures,
    build_spider_fixtures,
    load_hf_dataset_with_fallbacks,
    normalize_track5_rows,
    select_samples,
)
from src.utils.config import load_track_config, resolve_path
from src.utils.io import ensure_dir, write_json, write_jsonl


def prepare_track5_dataset(
    dataset_name: str,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    allow_fixtures: bool = False,
) -> dict:
    track_cfg = load_track_config("track5_constraints")
    dataset_cfg = dict(track_cfg["datasets"][dataset_name])
    t0 = time.perf_counter()
    source = "local"
    fixture = False

    if dataset_name == "json_schema":
        rows = normalize_track5_rows(dataset_name, [])
    else:
        try:
            raw = load_hf_dataset_with_fallbacks(dataset_cfg, cache_dir=str(resolve_path("data/hf_cache")))
            rows = normalize_track5_rows(dataset_name, raw)
            source = dataset_cfg.get("repo_id", "hf")
        except Exception as exc:
            if not allow_fixtures or dataset_name not in {"mt_bench", "spider"}:
                raise
            if dataset_name == "mt_bench":
                rows = build_mt_bench_fixtures()
            elif dataset_name == "spider":
                rows = build_spider_fixtures()
            else:
                raise
            source = f"fixture_after_load_error:{exc!r}"
            fixture = True

    if sample_size is None:
        sample_size = dataset_cfg.get("sample_size")
    if sample_seed is None:
        sample_seed = int(dataset_cfg.get("sample_seed", 42))
    rows, selection = select_samples(
        rows,
        int(sample_size) if sample_size is not None else None,
        sample_seed,
        limit=limit,
    )

    out_dir = ensure_dir(resolve_path(Path("data/processed/track5_constraints") / dataset_name))
    write_jsonl(out_dir / "samples.jsonl", rows)
    metadata = {
        "track": "track5_constraints",
        "dataset": dataset_name,
        "source": source,
        "fixture": fixture,
        "num_rows": len(rows),
        "sample_selection": selection,
        "runtime_s": time.perf_counter() - t0,
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
    parser.add_argument("--allow-fixtures", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track5_constraints")
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    results = [
        prepare_track5_dataset(
            name,
            limit=args.limit,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            allow_fixtures=args.allow_fixtures,
        )
        for name in dataset_names
    ]
    write_json(resolve_path("logs/track5_constraints/dataset_prepare_summary.json"), results)


if __name__ == "__main__":
    main()
