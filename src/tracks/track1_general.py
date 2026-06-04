from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.hf_loader import normalize_track1_rows
from src.metrics.accuracy import accuracy
from src.metrics.code_eval import humaneval_passes
from src.metrics.ifeval_simple import evaluate_ifeval_fallback
from src.models.hf_local import HFLocalGenerator
from src.parsing.answer_extractors import exact_numeric_match, extract_choice, extract_number
from src.parsing.code_extractors import extract_python_completion
from src.utils.config import get_dataset_config, get_model_config, load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text, require_cuda
from src.utils.io import append_jsonl, ensure_dir, read_jsonl, write_json, write_jsonl
from src.utils.seed import set_seed


def build_prompt(sample: dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "mmlu_pro":
        options = "\n".join(
            f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(sample["options"])
        )
        return (
            "Answer the following multiple-choice question. "
            "Return only the letter of the correct option.\n\n"
            f"Question: {sample['question']}\n{options}\n\nAnswer:"
        )
    if dataset_name == "hellaswag":
        options = "\n".join(
            f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(sample["options"])
        )
        return (
            "Choose the most plausible continuation. "
            "Return only the letter of the correct option.\n\n"
            f"Context: {sample['question']}\n{options}\n\nAnswer:"
        )
    if dataset_name == "gsm8k":
        return (
            "Solve the grade-school math problem. Show concise reasoning and end with "
            "'Final answer: <number>'.\n\n"
            f"Problem: {sample['question']}\n\nSolution:"
        )
    if dataset_name == "humaneval":
        return (
            "Complete the following Python function. Return only valid Python code for the "
            "function body or continuation, with no markdown.\n\n"
            f"{sample['prompt']}"
        )
    if dataset_name == "ifeval":
        return sample["prompt"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def score_sample(sample: dict[str, Any], dataset_name: str, output: str, timeout_s: int = 5) -> dict:
    if dataset_name in {"mmlu_pro", "hellaswag"}:
        valid_letters = [chr(ord("A") + i) for i in range(len(sample["options"]))]
        pred = extract_choice(output, valid_letters)
        return {"prediction": pred, "gold": sample["answer"], "is_correct": pred == sample["answer"]}
    if dataset_name == "gsm8k":
        pred = extract_number(output)
        return {
            "prediction": pred,
            "gold": sample["answer"],
            "is_correct": exact_numeric_match(pred, sample["answer"]),
        }
    if dataset_name == "humaneval":
        completion = extract_python_completion(sample["prompt"], output)
        result = humaneval_passes(
            prompt=sample["prompt"],
            completion=completion,
            test=sample["test"],
            entry_point=sample["entry_point"],
            timeout_s=timeout_s,
        )
        return {
            "prediction": completion,
            "gold": sample.get("canonical_solution"),
            "is_correct": bool(result["passed"]),
            "execution_error": result.get("error"),
        }
    if dataset_name == "ifeval":
        result = evaluate_ifeval_fallback(sample, output)
        return {
            "prediction": output,
            "gold": sample.get("instruction_id_list"),
            "is_correct": bool(result["strict_pass"]),
            "ifeval_fallback": result,
        }
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_processed_samples(dataset_name: str) -> list[dict[str, Any]]:
    path = resolve_path(Path("data/processed/track1_general") / dataset_name / "samples.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run scripts/prepare_datasets.py --track track1_general first."
        )
    return read_jsonl(path)


def run_track1_model_dataset(
    model_name: str,
    dataset_name: str,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    require_gpu: bool = True,
    overwrite: bool = False,
) -> dict:
    if require_gpu:
        require_cuda()
    set_seed(42)
    track_cfg = load_track_config("track1_general")
    model_cfg = get_model_config(track_cfg, model_name)
    dataset_cfg = get_dataset_config(track_cfg, dataset_name)
    samples = load_processed_samples(dataset_name)
    if sample_size is None and dataset_cfg.get("sample_size"):
        sample_size = int(dataset_cfg["sample_size"])
    if sample_seed is None:
        sample_seed = int(dataset_cfg.get("sample_seed", 42))
    sample_selection = {
        "source_num_samples": len(samples),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "mode": "full",
    }
    if sample_size is not None and len(samples) > sample_size:
        rng = random.Random(sample_seed)
        selected_indices = sorted(rng.sample(range(len(samples)), sample_size))
        samples = [samples[idx] for idx in selected_indices]
        sample_selection.update(
            {
                "mode": "random_sample",
                "selected_indices": selected_indices,
            }
        )
    if limit is not None:
        samples = samples[:limit]
        sample_selection["limit"] = limit
        sample_selection["mode"] = sample_selection["mode"] + "+limit"

    safe_model = model_cfg["name"].replace("/", "_").replace(" ", "_")
    output_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["output_dir"]) / safe_model))
    metric_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["metric_dir"]) / safe_model))
    log_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["log_dir"]) / safe_model))
    output_path = output_dir / f"{dataset_name}.jsonl"
    metric_path = metric_dir / f"{dataset_name}.json"
    log_path = log_dir / f"{dataset_name}.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to rerun.")
    if output_path.exists() and overwrite:
        output_path.unlink()

    generation_cfg = dict(track_cfg["generation"])
    max_new_tokens = int(dataset_cfg.get("max_new_tokens") or generation_cfg["default_max_new_tokens"])
    if model_cfg.get("family") == "dllm":
        generation_cfg["steps"] = int(generation_cfg.get("dllm_default_steps", 64))

    prompts = [build_prompt(sample, dataset_name) for sample in samples]
    rows = []
    t0 = time.perf_counter()
    generator = HFLocalGenerator(model_cfg, generation_cfg)
    try:
        for sample, prompt in tqdm(list(zip(samples, prompts)), desc=f"{model_cfg['name']}:{dataset_name}"):
            sample_t0 = time.perf_counter()
            try:
                output = generator.generate([prompt], max_new_tokens=max_new_tokens)[0]
                score = score_sample(
                    sample,
                    dataset_name,
                    output,
                    timeout_s=int(dataset_cfg.get("timeout_s", 5)),
                )
                error = None
            except Exception as exc:
                output = ""
                score = {"prediction": None, "gold": sample.get("answer"), "is_correct": False}
                error = repr(exc)
            row = {
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "model": model_cfg["name"],
                "prompt": prompt,
                "model_output": output,
                "score": score,
                "error": error,
                "latency_s": time.perf_counter() - sample_t0,
                "decoding_config": {
                    "max_new_tokens": max_new_tokens,
                    "scoring_mode": dataset_cfg.get("scoring_mode"),
                    "generation": generation_cfg,
                    "sample_selection": {
                        key: value for key, value in sample_selection.items() if key != "selected_indices"
                    },
                },
            }
            rows.append(row)
            append_jsonl(output_path, row)
    finally:
        generator.close()

    # The jsonl file is written incrementally above so long runs keep partial
    # results if interrupted.
    metric = summarize_rows(rows, dataset_name)
    metric.update(
        {
            "model": model_cfg["name"],
            "dataset": dataset_name,
            "num_samples": len(rows),
            "sample_selection": sample_selection,
            "total_runtime_s": time.perf_counter() - t0,
            "cuda": cuda_summary(),
            "nvidia_smi": nvidia_smi_text(),
        }
    )
    write_json(metric_path, metric)
    write_json(log_path, {"model_config": model_cfg, "dataset_config": dataset_cfg, "metric": metric})
    return metric


def summarize_rows(rows: list[dict], dataset_name: str) -> dict:
    flat = []
    errors = 0
    for row in rows:
        if row.get("error"):
            errors += 1
        score = row.get("score") or {}
        flat.append({"is_correct": bool(score.get("is_correct"))})
    metric_name = {
        "mmlu_pro": "accuracy",
        "hellaswag": "accuracy",
        "gsm8k": "exact_match",
        "humaneval": "pass_at_1",
        "ifeval": "instruction_following_rate_fallback",
    }[dataset_name]
    out = {metric_name: accuracy(flat), "num_errors": errors}
    if dataset_name == "ifeval":
        supported = []
        unsupported = []
        for row in rows:
            fb = ((row.get("score") or {}).get("ifeval_fallback") or {})
            supported.append(int(fb.get("supported_count", 0)))
            unsupported.append(int(fb.get("unsupported_count", 0)))
        out["ifeval_supported_instructions"] = sum(supported)
        out["ifeval_unsupported_instructions"] = sum(unsupported)
        out["warning"] = "IFEval score uses fallback checker; run official IFEval for paper numbers."
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name from Track 1 config, or 'all'.")
    parser.add_argument("--dataset", required=True, help="Dataset name from Track 1 config, or 'all'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track1_general")
    model_names = [cfg["name"] for cfg in track_cfg_models(track_cfg)] if args.model == "all" else [args.model]
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]

    metrics = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            metrics.append(
                run_track1_model_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    limit=args.limit,
                    sample_size=args.sample_size,
                    sample_seed=args.sample_seed,
                    require_gpu=not args.no_require_cuda,
                    overwrite=args.overwrite,
                )
            )
    print(metrics)


def track_cfg_models(track_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from src.utils.config import load_track_model_configs

    return load_track_model_configs(track_cfg)


if __name__ == "__main__":
    main()
