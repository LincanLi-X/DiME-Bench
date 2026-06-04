from __future__ import annotations

import re
from statistics import median
from typing import Any, Iterable

from datasets import Dataset, load_dataset


WORD_RE = re.compile(r"\b\w+(?:'\w+)?\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def load_raw_track2_dataset(dataset_cfg: dict[str, Any], cache_dir: str) -> Dataset:
    repo_id = dataset_cfg["repo_id"]
    subset = dataset_cfg.get("subset")
    split = dataset_cfg["split"]
    if subset:
        return load_dataset(repo_id, subset, split=split, cache_dir=cache_dir)
    return load_dataset(repo_id, split=split, cache_dir=cache_dir)


def normalize_track2_rows(
    dataset_name: str,
    raw: Iterable[dict[str, Any]],
    dataset_cfg: dict[str, Any],
    limit: int | None = None,
    max_source_rows: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw):
        if max_source_rows is not None and idx >= max_source_rows:
            break
        sample = _normalize_one(dataset_name, idx, row, dataset_cfg)
        if sample is None:
            continue
        rows.append(sample)
        if limit is not None and len(rows) >= limit:
            break
    _assign_density_bins(rows)
    return rows


def _normalize_one(
    dataset_name: str,
    idx: int,
    row: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if dataset_name == "wikitext103_sentence":
        return _normalize_wikitext(idx, row, dataset_cfg)
    if dataset_name == "cnn_dailymail_paragraph":
        return _normalize_cnn(idx, row, dataset_cfg)
    if dataset_name == "arxiv_abstract":
        return _normalize_arxiv(idx, row, dataset_cfg)
    raise ValueError(f"Unsupported Track 2 dataset: {dataset_name}")


def _normalize_wikitext(idx: int, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    text = _clean_text(str(row.get("text") or ""))
    if len(_tokens(text)) < 80 or text.startswith("="):
        return None
    sentences = _sentences(text)
    if len(sentences) < 3:
        return None
    min_len = int(cfg.get("min_span_tokens", 10))
    max_len = int(cfg.get("max_span_tokens", 40))
    candidate_idx = _pick_sentence(sentences, min_len, max_len)
    if candidate_idx is None or candidate_idx == 0 or candidate_idx >= len(sentences) - 1:
        return None
    reference = sentences[candidate_idx]
    prefix = " ".join(sentences[:candidate_idx])
    suffix = " ".join(sentences[candidate_idx + 1 :])
    return _build_sample("wikitext103_sentence", idx, prefix, suffix, reference, cfg, row)


def _normalize_cnn(idx: int, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    article = _clean_text(str(row.get("article") or row.get("text") or row.get("document") or ""))
    if len(_tokens(article)) < 260:
        return None
    paragraphs = [_clean_text(part) for part in re.split(r"\n{2,}|\r\n{2,}", article) if _tokens(part)]
    min_len = int(cfg.get("min_span_tokens", 80))
    max_len = int(cfg.get("max_span_tokens", 200))
    if len(paragraphs) >= 3:
        for para_idx in range(1, len(paragraphs) - 1):
            if min_len <= len(_tokens(paragraphs[para_idx])) <= max_len:
                prefix = " ".join(paragraphs[:para_idx])
                suffix = " ".join(paragraphs[para_idx + 1 :])
                return _build_sample("cnn_dailymail_paragraph", idx, prefix, suffix, paragraphs[para_idx], cfg, row)
    return _sentence_group_sample("cnn_dailymail_paragraph", idx, article, cfg, row)


def _normalize_arxiv(idx: int, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    abstract = str(row.get("abstract") or row.get("summary") or "")
    abstract = _clean_text(abstract.replace("Abstract:", ""))
    if len(_tokens(abstract)) < 180:
        return None
    return _sentence_group_sample("arxiv_abstract", idx, abstract, cfg, row)


def _sentence_group_sample(
    dataset_name: str,
    idx: int,
    text: str,
    cfg: dict[str, Any],
    raw_row: dict[str, Any],
) -> dict[str, Any] | None:
    sentences = _sentences(text)
    if len(sentences) < 5:
        return None
    min_len = int(cfg.get("min_span_tokens", 80))
    max_len = int(cfg.get("max_span_tokens", 150))
    best: tuple[int, int] | None = None
    center = len(sentences) / 2
    for start in range(1, len(sentences) - 2):
        length = 0
        for end in range(start + 1, len(sentences)):
            length += len(_tokens(sentences[end - 1]))
            if length > max_len:
                break
            if min_len <= length <= max_len and end < len(sentences):
                candidate = (start, end)
                if best is None:
                    best = candidate
                else:
                    best_center = sum(best) / 2
                    cand_center = sum(candidate) / 2
                    if abs(cand_center - center) < abs(best_center - center):
                        best = candidate
                break
    if best is None:
        return None
    start, end = best
    reference = " ".join(sentences[start:end])
    prefix = " ".join(sentences[:start])
    suffix = " ".join(sentences[end:])
    return _build_sample(dataset_name, idx, prefix, suffix, reference, cfg, raw_row)


def _build_sample(
    dataset_name: str,
    idx: int,
    prefix: str,
    suffix: str,
    reference: str,
    cfg: dict[str, Any],
    raw_row: dict[str, Any],
) -> dict[str, Any] | None:
    prefix = _clip_words(_clean_text(prefix), int(cfg.get("prefix_token_budget", 256)), from_left=False)
    suffix = _clip_words(_clean_text(suffix), int(cfg.get("suffix_token_budget", 256)), from_left=True)
    reference = _clean_text(reference)
    if not prefix or not suffix or not reference:
        return None
    span_length = len(_tokens(reference))
    density = context_density(prefix, suffix)
    return {
        "sample_id": f"{dataset_name}-{idx}",
        "dataset": dataset_name,
        "domain": cfg.get("domain"),
        "prefix": prefix,
        "suffix": suffix,
        "reference": reference,
        "span_length": span_length,
        "prefix_length": len(_tokens(prefix)),
        "suffix_length": len(_tokens(suffix)),
        "span_length_bin": span_length_bin(span_length),
        "context_density": density,
        "context_density_bin": "unknown",
        "source_metadata": _metadata(raw_row),
    }


def context_density(prefix: str, suffix: str, window_tokens: int = 64) -> float:
    boundary = _tokens(prefix)[-window_tokens:] + _tokens(suffix)[:window_tokens]
    if not boundary:
        return 0.0
    content = [tok for tok in boundary if len(tok) >= 5 and not tok.isdigit()]
    unique_content = len(set(content))
    return unique_content / len(boundary)


def span_length_bin(span_length: int) -> str:
    if span_length <= 40:
        return "short"
    if span_length <= 120:
        return "medium"
    return "long"


def _assign_density_bins(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    threshold = median(float(row["context_density"]) for row in rows)
    for row in rows:
        row["context_density_bin"] = "high" if float(row["context_density"]) >= threshold else "low"


def _pick_sentence(sentences: list[str], min_len: int, max_len: int) -> int | None:
    center = len(sentences) / 2
    candidates = [
        idx
        for idx, sentence in enumerate(sentences)
        if min_len <= len(_tokens(sentence)) <= max_len
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs(idx - center))


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(_tokens(part)) < 4:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [part for part in merged if len(_tokens(part)) >= 4]


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def _clip_words(text: str, budget: int, from_left: bool) -> str:
    words = str(text).split()
    if len(words) <= budget:
        return " ".join(words)
    clipped = words[:budget] if from_left else words[-budget:]
    return " ".join(clipped)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).replace("@-@", "-")).strip()
    return text


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    keep = ["id", "article_id", "url", "title", "section_names"]
    return {key: row.get(key) for key in keep if key in row}
