from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.metrics.track4_reasoning import aggregate_by_reasoning_type, score_track4_sample
from src.models.hf_local import HFLocalGenerator
from src.utils.config import get_dataset_config, load_model_config, load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text, require_cuda
from src.utils.io import append_jsonl, ensure_dir, read_json, read_jsonl, write_json
from src.utils.seed import set_seed


TRACK = "track4_reasoning"


def build_prompt(sample: dict[str, Any], dataset_name: str, dataset_cfg: dict[str, Any] | None = None) -> str:
    dataset_cfg = dataset_cfg or get_dataset_config(load_track_config(TRACK), dataset_name)
    template_path = resolve_path(dataset_cfg["prompt_template"])
    template = template_path.read_text(encoding="utf-8")
    if dataset_name in {"gsm8k", "math_500"}:
        return template.format(question=sample["question"])
    if dataset_name == "winogrande":
        return template.format(
            sentence=sample["sentence"],
            option_a=sample["options"][0],
            option_b=sample["options"][1],
        )
    if dataset_name == "path_star":
        return template.format(
            graph_text=sample["graph_text"],
            start=sample["start"],
            target=sample["target"],
        )
    raise ValueError(f"Unsupported Track 4 dataset: {dataset_name}")


def load_processed_samples(dataset_name: str) -> list[dict[str, Any]]:
    path = resolve_path(Path("data/processed/track4_reasoning") / dataset_name / "samples.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Run scripts/prepare_track4_reasoning_datasets.py first."
        )
    return read_jsonl(path)


def run_track4_model_dataset(
    model_name: str,
    dataset_name: str,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    require_gpu: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    mock_model: bool = False,
) -> dict[str, Any]:
    if require_gpu and not (dry_run or mock_model):
        require_cuda()
    set_seed(42)
    track_cfg = load_track_config(TRACK)
    model_cfg = get_track4_model_config(track_cfg, model_name)
    dataset_cfg = get_dataset_config(track_cfg, dataset_name)
    _validate_model_dataset_pair(model_cfg["name"], dataset_name, dataset_cfg)

    samples = load_processed_samples(dataset_name)
    if sample_size is None and dataset_cfg.get("sample_size"):
        sample_size = int(dataset_cfg["sample_size"])
    if sample_seed is None:
        sample_seed = int(dataset_cfg.get("sample_seed", 42))
    samples, sample_selection = select_samples(samples, sample_size, sample_seed, limit)

    safe_model = safe_name(model_cfg["name"])
    output_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["output_dir"]) / safe_model))
    metric_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["metric_dir"]) / safe_model))
    log_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["log_dir"]) / safe_model))
    output_path = output_dir / f"{dataset_name}.jsonl"
    metric_path = metric_dir / f"{dataset_name}.json"
    log_path = log_dir / f"{dataset_name}.jsonl"
    for path in [output_path, log_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to rerun.")
        if path.exists() and overwrite:
            path.unlink()

    generation_cfg = dict(track_cfg["generation"])
    max_new_tokens = int(dataset_cfg.get("max_new_tokens") or generation_cfg["default_max_new_tokens"])
    if model_cfg.get("family") == "dllm":
        generation_cfg["steps"] = int(generation_cfg.get("dllm_default_steps", 64))

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    generator = None if (dry_run or mock_model) else HFLocalGenerator(model_cfg, generation_cfg)
    try:
        iterator = tqdm(samples, desc=f"{model_cfg['name']}:{dataset_name}")
        for sample in iterator:
            prompt = build_prompt(sample, dataset_name, dataset_cfg)
            sample_t0 = time.perf_counter()
            try:
                if dry_run:
                    output = ""
                    score = {"prediction": None, "gold": sample.get("reference"), "is_correct": False, "parser_failed": True}
                    error = None
                elif mock_model:
                    output = mock_output(sample, dataset_name)
                    score = score_track4_sample(sample, dataset_name, output)
                    error = None
                else:
                    assert generator is not None
                    output = generator.generate([prompt], max_new_tokens=max_new_tokens)[0]
                    score = score_track4_sample(sample, dataset_name, output)
                    error = None
            except Exception as exc:
                output = ""
                score = {
                    "prediction": None,
                    "gold": sample.get("reference", sample.get("answer")),
                    "is_correct": False,
                    "parser_failed": True,
                }
                error = repr(exc)

            row = {
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "reasoning_type": sample["reasoning_type"],
                "model": model_cfg["name"],
                "prompt": prompt,
                "model_output": output,
                "parsed_output": score.get("prediction"),
                "reference": sample.get("reference", sample.get("answer")),
                "decoding_config": {
                    "max_new_tokens": max_new_tokens,
                    "scoring_mode": dataset_cfg.get("scoring_mode"),
                    "generation": generation_cfg,
                    "dry_run": dry_run,
                    "mock_model": mock_model,
                    "sample_selection": {
                        key: value for key, value in sample_selection.items() if key != "selected_indices"
                    },
                },
                "score": score,
                "error": error,
            }
            log_row = {
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "model": model_cfg["name"],
                "latency_s": time.perf_counter() - sample_t0,
                "error": error,
                "parser_failed": score.get("parser_failed"),
                "is_correct": score.get("is_correct"),
            }
            rows.append(row)
            append_jsonl(output_path, row)
            append_jsonl(log_path, log_row)
    finally:
        if generator is not None:
            generator.close()

    runtime_s = time.perf_counter() - t0
    metric = summarize_rows(rows, dataset_name)
    metric.update(
        {
            "track": TRACK,
            "dataset": dataset_name,
            "reasoning_type": dataset_cfg["reasoning_type"],
            "model": model_cfg["name"],
            "num_samples": len(rows),
            "runtime_s": runtime_s,
            "sample_selection": sample_selection,
            "dry_run": dry_run,
            "mock_model": mock_model,
            "cuda": cuda_summary(),
            "nvidia_smi": nvidia_smi_text(),
        }
    )
    if dry_run:
        metric["result_status"] = "dry_run"
    write_json(metric_path, metric)
    update_aggregate_tables(track_cfg)
    return metric


def summarize_rows(rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    from src.metrics.track4_reasoning import summarize_track4_rows

    return summarize_track4_rows(rows, dataset_name)


def select_samples(
    samples: list[dict[str, Any]],
    sample_size: int | None,
    sample_seed: int | None,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_selection: dict[str, Any] = {
        "source_num_samples": len(samples),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "mode": "full",
    }
    if sample_size is not None and len(samples) > sample_size:
        rng = random.Random(sample_seed)
        selected_indices = sorted(rng.sample(range(len(samples)), sample_size))
        samples = [samples[idx] for idx in selected_indices]
        sample_selection.update({"mode": "random_sample", "selected_indices": selected_indices})
    if limit is not None:
        samples = samples[:limit]
        sample_selection["limit"] = limit
        sample_selection["mode"] = sample_selection["mode"] + "+limit"
    return samples, sample_selection


def mock_output(sample: dict[str, Any], dataset_name: str) -> str:
    reference = sample.get("reference", sample.get("answer"))
    if dataset_name in {"gsm8k", "math_500"}:
        return f"Final answer: {reference}"
    if dataset_name == "winogrande":
        return str(reference)
    if dataset_name == "path_star":
        return " -> ".join(reference)
    return str(reference)


def load_track4_model_configs(track_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    configs = []
    for spec in track_cfg["models"]:
        if isinstance(spec, str):
            configs.append(load_model_config(spec))
        elif isinstance(spec, dict):
            cfg = dict(spec)
            cfg["_config_path"] = "<track4_reasoning:inline>"
            configs.append(cfg)
        else:
            raise TypeError(f"Unsupported model config entry: {spec!r}")
    return configs


def get_track4_model_config(track_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_name_norm = model_name.lower()
    for cfg in load_track4_model_configs(track_cfg):
        aliases = {
            cfg["name"].lower(),
            Path(str(cfg["_config_path"])).stem.lower(),
            str(cfg["repo_id"]).split("/")[-1].lower(),
        }
        if model_name_norm in aliases:
            return cfg
    available = ", ".join(cfg["name"] for cfg in load_track4_model_configs(track_cfg))
    raise KeyError(f"Unknown Track 4 model '{model_name}'. Available models: {available}")


def _validate_model_dataset_pair(model_name: str, dataset_name: str, dataset_cfg: dict[str, Any]) -> None:
    second = dataset_cfg.get("second_ar_baseline")
    if model_name == "GPT-2-XL" and second != "GPT-2-XL":
        raise ValueError(f"GPT-2-XL is not the configured second AR baseline for {dataset_name}.")
    if model_name == "gpt-oss-20b" and second != "gpt-oss-20b":
        raise ValueError(f"gpt-oss-20b is not the configured second AR baseline for {dataset_name}.")


def update_aggregate_tables(track_cfg: dict[str, Any]) -> None:
    metric_dir = ensure_dir(resolve_path(track_cfg["reporting"]["metric_dir"]))
    metric_rows = []
    for model_cfg in load_track4_model_configs(track_cfg):
        model_name = model_cfg["name"]
        safe_model = safe_name(model_name)
        for dataset_name, dataset_cfg in track_cfg["datasets"].items():
            path = metric_dir / safe_model / f"{dataset_name}.json"
            if not path.exists():
                continue
            metric = read_json(path)
            metric_rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "reasoning_type": dataset_cfg["reasoning_type"],
                    "metric_name": metric.get("metric_name"),
                    "metric_value": metric.get("metric_value"),
                    "num_samples": metric.get("num_samples"),
                    "parser_failure_rate": metric.get("parser_failure_rate"),
                    "result_status": metric.get("result_status"),
                }
            )

    _write_csv(metric_dir / "reasoning_main_table.csv", metric_rows)
    by_type_rows = []
    gap_rows = []
    for model_name in sorted({row["model"] for row in metric_rows}):
        rows = [row for row in metric_rows if row["model"] == model_name]
        for reasoning_type in ["chaining", "planning"]:
            values = [
                float(row["metric_value"])
                for row in rows
                if row["reasoning_type"] == reasoning_type and row.get("metric_value") is not None
            ]
            by_type_rows.append(
                {
                    "model": model_name,
                    "reasoning_type": reasoning_type,
                    "num_datasets": len(values),
                    "average": sum(values) / len(values) if values else None,
                }
            )
        gap = aggregate_by_reasoning_type(rows)
        gap_rows.append({"model": model_name, **gap})
    _write_csv(metric_dir / "by_reasoning_type.csv", by_type_rows)
    _write_csv(metric_dir / "planning_chaining_gap.csv", gap_rows)
    write_pathstar_error_analysis(track_cfg, metric_dir)


def write_pathstar_error_analysis(track_cfg: dict[str, Any], metric_dir: Path) -> None:
    output_root = resolve_path(track_cfg["reporting"]["output_dir"])
    rows = []
    for model_cfg in load_track4_model_configs(track_cfg):
        model_name = model_cfg["name"]
        path = output_root / safe_name(model_name) / "path_star.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            score = row.get("score") or {}
            rows.append(
                {
                    "model": model_name,
                    "sample_id": row.get("sample_id"),
                    "is_correct": score.get("is_correct"),
                    "valid_path": score.get("valid_path"),
                    "starts_correct": score.get("starts_correct"),
                    "reaches_target": score.get("reaches_target"),
                    "valid_edges": score.get("valid_edges"),
                    "parser_failed": score.get("parser_failed"),
                }
            )
    _write_csv(metric_dir / "pathstar_error_analysis.csv", rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name from Track 4 config, or 'all'.")
    parser.add_argument("--dataset", required=True, help="Dataset name from Track 4 config, or 'all'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config(TRACK)
    model_names = [cfg["name"] for cfg in load_track4_model_configs(track_cfg)] if args.model == "all" else [args.model]
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]

    metrics = []
    for current_model in model_names:
        for current_dataset in dataset_names:
            try:
                metrics.append(
                    run_track4_model_dataset(
                        model_name=current_model,
                        dataset_name=current_dataset,
                        limit=args.limit,
                        sample_size=args.sample_size,
                        sample_seed=args.sample_seed,
                        require_gpu=not args.no_require_cuda,
                        overwrite=args.overwrite,
                        dry_run=args.dry_run,
                        mock_model=args.mock_model,
                    )
                )
            except ValueError as exc:
                if args.model == "all":
                    print(f"Skipping {current_model}:{current_dataset}: {exc}")
                    continue
                raise
    print(metrics)


if __name__ == "__main__":
    main()
