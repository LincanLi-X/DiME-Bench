from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.track5_constraints_loader import dump_schema_json, select_samples
from src.metrics.track5_constraints import (
    extract_json_object,
    extract_sql,
    score_track5_sample,
    summarize_track5_rows,
)
from src.models.hf_local import HFLocalGenerator
from src.utils.config import load_model_config, load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text, require_cuda
from src.utils.io import append_jsonl, ensure_dir, read_jsonl, write_json
from src.utils.seed import set_seed


def build_prompt(sample: dict[str, Any], dataset_name: str, track_cfg: dict[str, Any]) -> str:
    template_path = track_cfg["datasets"][dataset_name].get("prompt_template")
    template = resolve_path(template_path).read_text(encoding="utf-8") if template_path else "{prompt}"
    if dataset_name == "mt_bench":
        return template.format(question=sample["question"])
    if dataset_name == "ifeval":
        return template.format(prompt=sample["prompt"])
    if dataset_name == "json_schema":
        return template.format(
            instruction=sample["instruction"],
            schema_json=dump_schema_json(sample["reference"]["schema"]),
        )
    if dataset_name == "spider":
        return template.format(
            db_id=sample.get("db_id") or "",
            schema=sample.get("schema") or "",
            question=sample.get("question") or "",
        )
    raise ValueError(f"Unsupported Track 5 dataset: {dataset_name}")


def load_processed_samples(dataset_name: str) -> list[dict[str, Any]]:
    path = resolve_path(Path("data/processed/track5_constraints") / dataset_name / "samples.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Run scripts/prepare_track5_constraints_datasets.py first."
        )
    return read_jsonl(path)


def run_track5_model_dataset(
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
    track_cfg = load_track_config("track5_constraints")
    model_cfg = get_track5_model_config(track_cfg, model_name)
    dataset_cfg = dict(track_cfg["datasets"][dataset_name])
    samples = load_processed_samples(dataset_name)

    if sample_size is None:
        sample_size = dataset_cfg.get("sample_size")
    if sample_seed is None:
        sample_seed = int(dataset_cfg.get("sample_seed", 42))
    samples, sample_selection = select_samples(
        samples,
        int(sample_size) if sample_size is not None else None,
        sample_seed,
        limit=limit,
    )

    safe_model = safe_name(model_cfg["name"])
    output_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["output_dir"]) / safe_model))
    metric_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["metric_dir"]) / safe_model))
    log_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["log_dir"]) / safe_model))
    trajectory_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["trajectory_dir"]) / safe_model))
    output_path = output_dir / f"{dataset_name}.jsonl"
    metric_path = metric_dir / f"{dataset_name}.json"
    log_path = log_dir / f"{dataset_name}.jsonl"
    trajectory_path = trajectory_dir / f"{dataset_name}.jsonl"

    for path in [output_path, log_path, trajectory_path]:
        if path.exists() and overwrite:
            path.unlink()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to rerun.")

    generation_cfg = dict(track_cfg["generation"])
    max_new_tokens = int(dataset_cfg.get("max_new_tokens") or generation_cfg["default_max_new_tokens"])
    if model_cfg.get("family") == "dllm":
        generation_cfg["steps"] = int(generation_cfg.get("dllm_default_steps", 64))

    prompts = [build_prompt(sample, dataset_name, track_cfg) for sample in samples]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    generator = None
    if not (dry_run or mock_model):
        generator = HFLocalGenerator(model_cfg, generation_cfg)
    try:
        for sample, prompt in tqdm(list(zip(samples, prompts)), desc=f"{model_cfg['name']}:{dataset_name}"):
            sample_t0 = time.perf_counter()
            error = None
            try:
                if dry_run:
                    output = ""
                    error = None
                elif mock_model:
                    output = mock_track5_output(sample, dataset_name)
                else:
                    assert generator is not None
                    output = generator.generate([prompt], max_new_tokens=max_new_tokens)[0]
                score = score_track5_sample(sample, dataset_name, output)
                if dry_run:
                    score["result_status"] = "dry_run"
                parsed_output = parsed_from_score(dataset_name, output, score)
            except Exception as exc:
                output = ""
                parsed_output = None
                score = {"is_correct": False, "result_status": "error"}
                error = repr(exc)

            latency_s = time.perf_counter() - sample_t0
            row = {
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "model": model_cfg["name"],
                "prompt": prompt,
                "model_output": output,
                "parsed_output": parsed_output,
                "reference": sample.get("reference"),
                "decoding_config": {
                    "max_new_tokens": max_new_tokens,
                    "generation": generation_cfg,
                    "sample_selection": {
                        key: value for key, value in sample_selection.items() if key != "selected_indices"
                    },
                    "primary_decoding": track_cfg.get("evaluation", {}).get("primary_decoding"),
                    "dry_run": dry_run,
                    "mock_model": mock_model,
                },
                "score": score,
                "error": error,
            }
            rows.append(row)
            append_jsonl(output_path, row)
            append_jsonl(
                log_path,
                {
                    "sample_id": sample["sample_id"],
                    "latency_s": latency_s,
                    "model": model_cfg["name"],
                    "dataset": dataset_name,
                    "seed": 42,
                    "mock_model": mock_model,
                    "dry_run": dry_run,
                    "error": error,
                },
            )
    finally:
        if generator is not None:
            generator.close()

    runtime_s = time.perf_counter() - t0
    metric = summarize_track5_rows(rows, dataset_name, model_cfg["name"], runtime_s=runtime_s)
    if mock_model:
        metric["result_status"] = "mock_model"
    if dry_run:
        metric["result_status"] = "dry_run"
    metric.update(
        {
            "sample_selection": sample_selection,
            "cuda": cuda_summary(),
            "nvidia_smi": nvidia_smi_text(),
            "model_config": {
                key: value
                for key, value in model_cfg.items()
                if key not in {"token", "api_key"}
            },
        }
    )
    write_json(metric_path, metric)
    return metric


def parsed_from_score(dataset_name: str, output: str, score: dict[str, Any]) -> Any:
    if dataset_name == "json_schema":
        return score.get("prediction") if score.get("prediction") is not None else extract_json_object(output)
    if dataset_name == "spider":
        return score.get("prediction") or extract_sql(output)
    return score.get("prediction", output)


def mock_track5_output(sample: dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "json_schema":
        return json.dumps(sample["reference"]["expected"], ensure_ascii=False)
    if dataset_name == "spider":
        return sample.get("reference", {}).get("gold_sql", "")
    if dataset_name == "ifeval":
        return "This is a concise response that follows the requested constraints where supported."
    if dataset_name == "mt_bench":
        return "A fixed seed makes benchmark sampling reproducible across models and reruns."
    return ""


def get_track5_model_config(track_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_name_norm = model_name.lower()
    configs = load_track5_model_configs(track_cfg)
    for cfg in configs:
        aliases = {
            cfg["name"].lower(),
            str(Path(str(cfg.get("_config_path", cfg["name"]))).stem).lower(),
            str(cfg.get("repo_id", "")).split("/")[-1].lower(),
        }
        if model_name_norm in aliases:
            return cfg
    available = ", ".join(cfg["name"] for cfg in configs)
    raise KeyError(f"Unknown Track 5 model '{model_name}'. Available models: {available}")


def load_track5_model_configs(track_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for item in track_cfg["models"]:
        if isinstance(item, str):
            configs.append(load_model_config(item))
        elif isinstance(item, dict):
            cfg = dict(item)
            cfg["_config_path"] = "configs/tracks/track5_constraints.yaml"
            configs.append(cfg)
        else:
            raise TypeError(f"Unsupported model config entry: {item!r}")
    return configs


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Track 5 model name, or 'all'.")
    parser.add_argument("--dataset", required=True, help="Track 5 dataset name, or 'all'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track5_constraints")
    model_names = (
        [cfg["name"] for cfg in load_track5_model_configs(track_cfg)] if args.model == "all" else [args.model]
    )
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]
    metrics = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            metrics.append(
                run_track5_model_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    limit=args.limit,
                    sample_size=args.sample_size,
                    sample_seed=args.sample_seed,
                    require_gpu=not args.no_require_cuda,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    mock_model=args.mock_model,
                )
            )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
