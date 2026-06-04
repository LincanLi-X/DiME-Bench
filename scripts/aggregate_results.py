#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils.config import load_track_config, load_track_model_configs, resolve_path
from src.utils.io import ensure_dir, read_json, write_json


def aggregate_track1() -> list[dict]:
    track_cfg = load_track_config("track1_general")
    rows = []
    for model_cfg in load_track_model_configs(track_cfg):
        safe_model = model_cfg["name"].replace("/", "_").replace(" ", "_")
        for dataset_name in track_cfg["datasets"]:
            metric_path = resolve_path(Path(track_cfg["reporting"]["metric_dir"]) / safe_model / f"{dataset_name}.json")
            if not metric_path.exists():
                rows.append({"model": model_cfg["name"], "dataset": dataset_name, "status": "missing"})
                continue
            metric = read_json(metric_path)
            score_keys = [
                "accuracy",
                "exact_match",
                "pass_at_1",
                "instruction_following_rate_fallback",
            ]
            score = next((metric[k] for k in score_keys if k in metric), None)
            rows.append(
                {
                    "model": model_cfg["name"],
                    "dataset": dataset_name,
                    "score": score,
                    "num_samples": metric.get("num_samples"),
                    "num_errors": metric.get("num_errors"),
                    "status": metric.get("status", "ok"),
                }
            )
    out_dir = ensure_dir(resolve_path("metrics/track1_general"))
    with (out_dir / "by_task_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "score", "num_samples", "num_errors", "status"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "by_task_table.json", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="track1_general")
    args = parser.parse_args()
    if args.track != "track1_general":
        raise SystemExit("Only track1_general aggregation is implemented.")
    rows = aggregate_track1()
    print(f"Wrote {len(rows)} rows to metrics/track1_general/by_task_table.csv")


if __name__ == "__main__":
    main()
