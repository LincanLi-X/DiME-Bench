#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.tracks.track1_general import run_track1_model_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="track1_general")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()

    if args.track != "track1_general":
        raise SystemExit("Only track1_general is implemented in scripts/run_track.py")
    metric = run_track1_model_dataset(
        model_name=args.model,
        dataset_name=args.dataset,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        require_gpu=not args.no_require_cuda,
        overwrite=args.overwrite,
    )
    print(metric)


if __name__ == "__main__":
    main()
