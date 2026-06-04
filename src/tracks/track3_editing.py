from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.track3_editing_loader import load_processed_track3_samples, select_track3_samples
from src.metrics.track3_editing import (
    parse_edited_text,
    score_track3_sample,
    summarize_track3_rows,
    write_track3_analysis_tables,
)
from src.models.hf_local import HFLocalGenerator
from src.utils.config import get_dataset_config, load_model_config, load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text, require_cuda
from src.utils.io import append_jsonl, ensure_dir, write_json
from src.utils.seed import set_seed


TRACK = "track3_editing"


def load_track3_model_configs(track_cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    track_cfg = track_cfg or load_track_config(TRACK)
    configs = [load_model_config(path) for path in track_cfg.get("models", [])]
    for inline_cfg in track_cfg.get("additional_models", []):
        cfg = dict(inline_cfg)
        cfg["_config_path"] = f"{TRACK}:additional_models:{cfg['name']}"
        configs.append(cfg)
    return configs


def get_track3_model_config(track_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_name_norm = model_name.lower()
    for cfg in load_track3_model_configs(track_cfg):
        aliases = {
            cfg["name"].lower(),
            Path(str(cfg.get("_config_path", ""))).stem.lower(),
            str(cfg.get("repo_id", "")).split("/")[-1].lower(),
        }
        if model_name_norm in aliases:
            return cfg
    available = ", ".join(cfg["name"] for cfg in load_track3_model_configs(track_cfg))
    raise KeyError(f"Unknown Track 3 model '{model_name}'. Available models: {available}")


def build_prompt(sample: dict[str, Any], dataset_name: str, target_marked: bool = False) -> str:
    track_cfg = load_track_config(TRACK)
    dataset_cfg = get_dataset_config(track_cfg, dataset_name)
    template_key = "target_marked_prompt_template" if target_marked else "prompt_template"
    template_path = dataset_cfg.get(template_key) or dataset_cfg["prompt_template"]
    template = resolve_path(template_path).read_text(encoding="utf-8")
    values = defaultdict(str, sample)
    return template.format_map(values)


def run_track3_model_dataset(
    model_name: str,
    dataset_name: str,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    require_gpu: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    mock_model: bool = False,
    target_marked: bool = False,
) -> dict[str, Any]:
    if require_gpu and not (dry_run or mock_model):
        require_cuda()
    set_seed(42)
    track_cfg = load_track_config(TRACK)
    model_cfg = get_track3_model_config(track_cfg, model_name)
    dataset_cfg = get_dataset_config(track_cfg, dataset_name)
    samples = load_processed_track3_samples(dataset_name)
    if sample_size is None and dataset_cfg.get("sample_size"):
        sample_size = int(dataset_cfg["sample_size"])
    if sample_seed is None:
        sample_seed = int(dataset_cfg.get("sample_seed", 42))
    samples, sample_selection = select_track3_samples(samples, sample_size, int(sample_seed), limit=limit)

    prompts = [build_prompt(sample, dataset_name, target_marked=target_marked) for sample in samples]
    generation_cfg = dict(track_cfg["generation"])
    max_new_tokens = int(dataset_cfg.get("max_new_tokens") or generation_cfg["default_max_new_tokens"])
    if model_cfg.get("family") == "dllm":
        generation_cfg["steps"] = int(generation_cfg.get("dllm_default_steps", 64))

    safe_model = _safe_name(model_cfg["name"])
    output_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["output_dir"]) / safe_model))
    metric_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["metric_dir"]) / safe_model))
    log_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["log_dir"]) / safe_model))
    trajectory_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["trajectory_dir"]) / safe_model))
    output_path = output_dir / f"{dataset_name}.jsonl"
    metric_path = metric_dir / f"{dataset_name}.json"
    log_path = log_dir / f"{dataset_name}.jsonl"
    trajectory_path = trajectory_dir / f"{dataset_name}.jsonl"

    if dry_run:
        return {
            "track": TRACK,
            "dataset": dataset_name,
            "model": model_cfg["name"],
            "num_samples": len(samples),
            "metric_name": "edit_success",
            "metric_value": None,
            "parser_failure_rate": None,
            "runtime_s": 0.0,
            "result_status": "dry_run",
            "first_prompt": prompts[0] if prompts else None,
            "sample_selection": sample_selection,
        }

    for path in (output_path, log_path, metric_path, trajectory_path):
        if path.exists() and overwrite:
            path.unlink()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to rerun.")

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    generator: HFLocalGenerator | None = None
    if not mock_model:
        generator = HFLocalGenerator(model_cfg, generation_cfg)
    try:
        iterator = zip(samples, prompts)
        for sample, prompt in tqdm(list(iterator), desc=f"{model_cfg['name']}:{dataset_name}"):
            sample_t0 = time.perf_counter()
            error = None
            try:
                if mock_model:
                    output = _mock_edit(sample)
                else:
                    assert generator is not None
                    output = generator.generate([prompt], max_new_tokens=max_new_tokens)[0]
                parsed = parse_edited_text(output)
                score = score_track3_sample(sample, parsed)
            except Exception as exc:
                output = ""
                parsed = ""
                score = {
                    "prediction": "",
                    "gold": sample.get("reference"),
                    "edit_success": False,
                    "preservation_score": 0.0,
                    "over_edit": False,
                    "changed_token_ratio": None,
                    "parser_failed": True,
                }
                error = repr(exc)

            decoding_config = {
                "max_new_tokens": max_new_tokens,
                "generation": generation_cfg,
                "target_marked": target_marked,
                "mock_model": mock_model,
                "sample_selection": sample_selection,
            }
            row = {
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "model": model_cfg["name"],
                "prompt": prompt,
                "model_output": output,
                "parsed_output": parsed,
                "reference": sample.get("reference"),
                "decoding_config": decoding_config,
                "score": score,
                "error": error,
            }
            log_row = {
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "model": model_cfg["name"],
                "latency_s": time.perf_counter() - sample_t0,
                "seed": 42,
                "mock_model": mock_model,
                "error": error,
                "output_path": str(output_path),
            }
            rows.append(row)
            append_jsonl(output_path, row)
            append_jsonl(log_path, log_row)
    finally:
        if generator is not None:
            generator.close()

    runtime_s = time.perf_counter() - t0
    metric = summarize_track3_rows(rows, dataset_name)
    metric.update(
        {
            "model": model_cfg["name"],
            "runtime_s": runtime_s,
            "sample_selection": sample_selection,
            "cuda": cuda_summary(),
            "nvidia_smi": nvidia_smi_text(),
            "trajectory_path": str(trajectory_path),
            "trajectory_available": False,
        }
    )
    write_json(metric_path, metric)
    write_track3_analysis_tables(track_cfg["reporting"]["metric_dir"])
    return metric


def _mock_edit(sample: dict[str, Any]) -> str:
    return str(sample.get("reference") or sample.get("source_text") or "")


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")
