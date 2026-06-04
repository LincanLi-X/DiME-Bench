from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.metrics.track2_infilling import (
    add_bertscore,
    score_infilling_sample,
    summarize_infilling_rows,
    write_breakdowns,
)
from src.models.hf_local import HFLocalGenerator
from src.utils.config import get_dataset_config, load_model_config, load_track_config, resolve_path
from src.utils.gpu import cuda_summary, nvidia_smi_text, require_cuda
from src.utils.io import append_jsonl, ensure_dir, read_jsonl, write_json
from src.utils.seed import set_seed


WRAPPER_VERSION = "track2_infilling_v1"


def load_track2_model_configs(track_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    configs = []
    for entry in track_cfg["models"]:
        if isinstance(entry, str):
            configs.append(load_model_config(entry))
        elif isinstance(entry, dict):
            cfg = dict(entry)
            cfg["_config_path"] = "<inline:track2_infilling.yaml>"
            configs.append(cfg)
        else:
            raise TypeError(f"Unsupported model entry: {entry!r}")
    return configs


def get_track2_model_config(track_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_name_norm = model_name.lower()
    for cfg in load_track2_model_configs(track_cfg):
        aliases = {
            str(cfg["name"]).lower(),
            Path(str(cfg.get("_config_path", cfg["name"]))).stem.lower(),
            str(cfg["repo_id"]).split("/")[-1].lower(),
        }
        if model_name_norm in aliases:
            return cfg
    available = ", ".join(cfg["name"] for cfg in load_track2_model_configs(track_cfg))
    raise KeyError(f"Unknown Track 2 model '{model_name}'. Available models: {available}")


def build_prompt(sample: dict[str, Any], model_cfg: dict[str, Any], track_cfg: dict[str, Any]) -> str:
    template_key = "ar_instruction"
    if model_cfg.get("family") == "ar_fim":
        template_key = "fim_starcoder2"
    elif model_cfg.get("family") == "dllm":
        template_key = "dllm_masked"
    template_path = resolve_path(track_cfg["prompt_templates"][template_key])
    template = template_path.read_text(encoding="utf-8")
    mask_count = int(sample.get("span_length") or 32)
    mask_block = " ".join(["[MASK]"] * min(mask_count, 256))
    return template.format(prefix=sample["prefix"], suffix=sample["suffix"], mask_block=mask_block)


def clean_infilling_output(text: str, track_cfg: dict[str, Any]) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(
        r"^(here is|the missing middle span is|missing middle span:|middle span:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    for marker in track_cfg.get("generation", {}).get("truncate_at_end_markers", []):
        if marker and marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    return cleaned


def load_processed_samples(dataset_name: str) -> list[dict[str, Any]]:
    path = resolve_path(Path("data/processed/track2_infilling") / dataset_name / "samples.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Run scripts/prepare_track2_infilling_datasets.py first."
        )
    return read_jsonl(path)


def run_track2_model_dataset(
    model_name: str,
    dataset_name: str,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    require_gpu: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    mock_model: bool = False,
    compute_bertscore: bool = False,
) -> dict[str, Any]:
    if require_gpu and not (dry_run or mock_model):
        require_cuda()
    set_seed(42)
    track_cfg = load_track_config("track2_infilling")
    model_cfg = get_track2_model_config(track_cfg, model_name)
    dataset_cfg = get_dataset_config(track_cfg, dataset_name)
    samples = load_processed_samples(dataset_name)
    if sample_size is None and dataset_cfg.get("sample_size"):
        sample_size = int(dataset_cfg["sample_size"])
    if sample_seed is None:
        sample_seed = int(dataset_cfg.get("sample_seed", 42))
    samples, sample_selection = _select_samples(samples, sample_size, sample_seed, limit)

    safe_model = safe_model_name(model_cfg["name"])
    output_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["output_dir"]) / safe_model))
    metric_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["metric_dir"]) / safe_model))
    log_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["log_dir"]) / safe_model))
    traj_dir = ensure_dir(resolve_path(Path(track_cfg["reporting"]["trajectory_dir"]) / safe_model))
    output_path = output_dir / f"{dataset_name}.jsonl"
    metric_path = metric_dir / f"{dataset_name}.json"
    log_path = log_dir / f"{dataset_name}.jsonl"
    trajectory_path = traj_dir / f"{dataset_name}.jsonl"
    for path in [output_path, log_path, trajectory_path]:
        if path.exists() and overwrite:
            path.unlink()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to rerun.")

    generation_cfg = dict(track_cfg["generation"])
    max_new_tokens = int(dataset_cfg.get("max_new_tokens") or generation_cfg["default_max_new_tokens"])
    if model_cfg.get("family") == "dllm":
        generation_cfg["steps"] = int(generation_cfg.get("dllm_default_steps", 64))

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    generator: HFLocalGenerator | None = None
    if not dry_run and not mock_model:
        generator = HFLocalGenerator(model_cfg, generation_cfg)
    try:
        for sample in tqdm(samples, desc=f"{model_cfg['name']}:{dataset_name}"):
            sample_t0 = time.perf_counter()
            prompt = build_prompt(sample, model_cfg, track_cfg)
            trajectory = None
            error = None
            try:
                if dry_run:
                    model_output = ""
                    error = "dry_run"
                elif mock_model:
                    model_output = sample["reference"]
                elif generator is not None and _can_native_dllm_infilling(model_cfg, generator, generation_cfg):
                    model_output, trajectory = _generate_native_dllm_infilling(
                        generator=generator,
                        sample=sample,
                        max_new_tokens=max_new_tokens,
                        steps=int(generation_cfg.get("steps", 64)),
                    )
                elif generator is not None:
                    model_output = generator.generate([prompt], max_new_tokens=max_new_tokens)[0]
                else:
                    raise RuntimeError("Generator was not initialized.")
            except Exception as exc:
                model_output = ""
                error = repr(exc)

            parsed_output = clean_infilling_output(model_output, track_cfg)
            score = score_infilling_sample(sample, parsed_output)
            row = {
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "model": model_cfg["name"],
                "prompt": prompt,
                "model_output": model_output,
                "parsed_output": parsed_output,
                "reference": sample["reference"],
                "decoding_config": {
                    "max_new_tokens": max_new_tokens,
                    "generation": generation_cfg,
                    "sample_selection": {
                        key: value for key, value in sample_selection.items() if key != "selected_indices"
                    },
                    "dry_run": dry_run,
                    "mock_model": mock_model,
                    "native_dllm_infilling": bool(trajectory),
                    "wrapper_version": WRAPPER_VERSION,
                },
                "score": score,
                "error": error,
            }
            rows.append({**row, "sample": sample})
            append_jsonl(output_path, row)
            append_jsonl(
                log_path,
                {
                    "sample_id": sample["sample_id"],
                    "latency_s": time.perf_counter() - sample_t0,
                    "tokens": {
                        "prediction_tokens": score["prediction_length"],
                        "reference_tokens": score["reference_length"],
                        "span_length": sample.get("span_length"),
                    },
                    "batch_size": int(generation_cfg.get("batch_size", 1)),
                    "seed": 42,
                    "wrapper_version": WRAPPER_VERSION,
                    "cuda": cuda_summary(),
                    "error": error,
                },
            )
            if trajectory is not None:
                append_jsonl(trajectory_path, {"sample_id": sample["sample_id"], **trajectory})
    finally:
        if generator is not None:
            generator.close()

    bertscore_info = {"available": False, "note": "BERTScore not requested."}
    if compute_bertscore and rows:
        bertscore_info = add_bertscore(rows)
        if bertscore_info.get("available"):
            output_path.unlink()
            for row in rows:
                row_to_write = {key: value for key, value in row.items() if key != "sample"}
                append_jsonl(output_path, row_to_write)

    runtime_s = time.perf_counter() - t0
    metric = summarize_infilling_rows(rows, dataset_name, model_cfg["name"], runtime_s)
    if dry_run:
        metric["result_status"] = "dry_run"
    if mock_model:
        metric["result_status"] = "mock_model"
    metric.update(
        {
            "sample_selection": sample_selection,
            "cuda": cuda_summary(),
            "nvidia_smi": nvidia_smi_text(),
            "bertscore": bertscore_info,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "trajectory_path": str(trajectory_path) if trajectory_path.exists() else None,
        }
    )
    breakdown_paths = write_breakdowns(rows, metric_dir / dataset_name)
    metric["breakdowns"] = breakdown_paths
    write_json(metric_path, metric)
    return metric


def safe_model_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _select_samples(
    samples: list[dict[str, Any]],
    sample_size: int | None,
    sample_seed: int,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection: dict[str, Any] = {
        "source_num_samples": len(samples),
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "mode": "full",
    }
    if sample_size is not None and len(samples) > sample_size:
        rng = random.Random(sample_seed)
        selected_indices = sorted(rng.sample(range(len(samples)), sample_size))
        samples = [samples[idx] for idx in selected_indices]
        selection.update({"mode": "random_sample", "selected_indices": selected_indices})
    if limit is not None:
        samples = samples[:limit]
        selection["limit"] = limit
        selection["mode"] = selection["mode"] + "+limit"
    return samples, selection


def _can_native_dllm_infilling(
    model_cfg: dict[str, Any],
    generator: HFLocalGenerator,
    generation_cfg: dict[str, Any],
) -> bool:
    return (
        bool(generation_cfg.get("native_dllm_infilling", True))
        and model_cfg.get("family") == "dllm"
        and getattr(generator.model.config, "mask_token_id", None) is not None
    )


def _generate_native_dllm_infilling(
    generator: HFLocalGenerator,
    sample: dict[str, Any],
    max_new_tokens: int,
    steps: int,
) -> tuple[str, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    tokenizer = generator.tokenizer
    model = generator.model
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    mask_token_id = int(getattr(model.config, "mask_token_id"))
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or eos_token_id

    left = (
        "Fill the masked middle span using both the prefix and suffix. "
        "Return only text that belongs in the masked span.\n\n"
        f"Prefix:\n{sample['prefix']}\n\nMasked middle span:\n"
    )
    right = f"\n\nSuffix:\n{sample['suffix']}"
    left_ids = tokenizer(left, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    right_ids = tokenizer(right, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    answer_len = min(max_new_tokens, max(8, int(sample.get("span_length") or max_new_tokens)))
    mask_ids = torch.full((answer_len,), mask_token_id, dtype=left_ids.dtype)
    input_ids = torch.cat([left_ids, mask_ids, right_ids], dim=0).unsqueeze(0).to(model_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
    mutable_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=model_device)
    start = left_ids.numel()
    end = start + answer_len
    mutable_mask[:, start:end] = True

    trajectory_steps: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in range(max(1, steps)):
            current_mask = (input_ids == mask_token_id) & mutable_mask
            mask_count = int(current_mask.sum().item())
            if mask_count == 0:
                break
            model_type = str(getattr(model.config, "model_type", "")).lower()
            forward_attention_mask = None if model_type == "dream" else attention_mask
            model_out = model(input_ids=input_ids, attention_mask=forward_attention_mask, use_cache=False)
            logits = model_out.logits
            if model_type == "dream":
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            probs = F.softmax(logits.float(), dim=-1)
            confidence, candidates = probs.max(dim=-1)
            candidates = candidates.masked_fill(candidates == mask_token_id, eos_token_id or pad_token_id)
            remaining_steps = max(1, steps - step)
            positions = torch.nonzero(current_mask[0], as_tuple=False).flatten()
            num_transfer = max(1, int(torch.ceil(torch.tensor(mask_count / remaining_steps)).item()))
            num_transfer = min(num_transfer, positions.numel())
            row_conf = confidence[0, positions]
            selected = positions[torch.topk(row_conf, k=num_transfer).indices]
            input_ids[0, selected] = candidates[0, selected]
            rel_positions = [int(pos.item() - start) for pos in selected]
            trajectory_steps.append(
                {
                    "step": step,
                    "mask_count": mask_count,
                    "finalized_positions": rel_positions,
                    "mean_confidence": float(row_conf.mean().item()) if row_conf.numel() else None,
                }
            )

    answer_ids = input_ids[0, start:end].tolist()
    cleaned_ids = []
    stop_ids = {mask_token_id}
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)
    if pad_token_id is not None:
        stop_ids.add(pad_token_id)
    for token_id in answer_ids:
        if token_id in stop_ids:
            break
        cleaned_ids.append(token_id)
    output = tokenizer.decode(cleaned_ids, skip_special_tokens=True).strip()
    trajectory = {
        "available": True,
        "method": "generic_confidence_unmasking_middle",
        "answer_region_tokens": answer_len,
        "steps": trajectory_steps,
    }
    return output, trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name from Track 2 config, or 'all'.")
    parser.add_argument("--dataset", required=True, help="Dataset name from Track 2 config, or 'all'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    args = parser.parse_args()

    track_cfg = load_track_config("track2_infilling")
    model_names = [cfg["name"] for cfg in load_track2_model_configs(track_cfg)] if args.model == "all" else [args.model]
    dataset_names = list(track_cfg["datasets"].keys()) if args.dataset == "all" else [args.dataset]

    metrics = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            metrics.append(
                run_track2_model_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    limit=args.limit,
                    sample_size=args.sample_size,
                    sample_seed=args.sample_seed,
                    require_gpu=not args.no_require_cuda,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    mock_model=args.mock_model,
                    compute_bertscore=args.bertscore,
                )
            )
    print(metrics)


if __name__ == "__main__":
    main()
