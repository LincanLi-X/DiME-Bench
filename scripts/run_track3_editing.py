#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.tracks.track3_editing import load_track3_model_configs, run_track3_model_dataset
from src.utils.config import load_track_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Track 3 model name, or 'all'.")
    parser.add_argument("--dataset", required=True, help="Track 3 dataset name, or 'all'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--target-marked", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track3_editing")
    model_names = (
        [cfg["name"] for cfg in load_track3_model_configs(track_cfg)]
        if args.model == "all"
        else [args.model]
    )
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]

    metrics = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            metrics.append(
                run_track3_model_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    limit=args.limit,
                    sample_size=args.sample_size,
                    sample_seed=args.sample_seed,
                    require_gpu=not args.no_require_cuda,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    mock_model=args.mock_model,
                    target_marked=args.target_marked,
                )
            )
    print(metrics)


if __name__ == "__main__":
    main()
