from __future__ import annotations

import re
from typing import Any


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    return len([x for x in re.split(r"[.!?]+", text) if x.strip()])


def _count_paragraphs(text: str) -> int:
    return len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])


def _count_bullets(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text))


def _kwargs_value(kwargs: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = kwargs.get(name)
        if value is not None:
            return value
    return None


def check_instruction(instruction_id: str, kwargs: dict[str, Any], response: str) -> tuple[bool | None, str]:
    """Best-effort deterministic IFEval subset.

    Returns (None, reason) for instructions that should be evaluated by the official
    google-research IFEval implementation. The Track 1 runner records unsupported
    counts separately so these fallback scores are not mistaken for official IFEval.
    """

    text = response.strip()
    iid = instruction_id.lower()

    if iid == "punctuation:no_comma":
        return ("," not in text, "no comma")

    if "number_words" in iid:
        target = _kwargs_value(kwargs, "num_words")
        relation = _kwargs_value(kwargs, "relation") or "at least"
        if target is None:
            return None, "missing num_words"
        count = _count_words(text)
        return _compare_count(count, int(target), str(relation)), f"word count {count}"

    if "number_sentences" in iid:
        target = _kwargs_value(kwargs, "num_sentences")
        relation = _kwargs_value(kwargs, "relation") or "at least"
        if target is None:
            return None, "missing num_sentences"
        count = _count_sentences(text)
        return _compare_count(count, int(target), str(relation)), f"sentence count {count}"

    if "number_paragraphs" in iid:
        target = _kwargs_value(kwargs, "num_paragraphs")
        relation = _kwargs_value(kwargs, "relation") or "at least"
        if target is None:
            return None, "missing num_paragraphs"
        count = _count_paragraphs(text)
        return _compare_count(count, int(target), str(relation)), f"paragraph count {count}"

    if "number_bullet_lists" in iid or "number_bullets" in iid:
        target = _kwargs_value(kwargs, "num_bullets")
        if target is None:
            return None, "missing num_bullets"
        count = _count_bullets(text)
        return count >= int(target), f"bullet count {count}"

    if "number_placeholders" in iid:
        target = _kwargs_value(kwargs, "num_placeholders")
        if target is None:
            return None, "missing num_placeholders"
        count = len(re.findall(r"\[[^\[\]]+\]", text))
        return count >= int(target), f"placeholder count {count}"

    if "keywords" in iid:
        keywords = _kwargs_value(kwargs, "keywords")
        if not keywords:
            return None, "missing keywords"
        if isinstance(keywords, str):
            keywords = [keywords]
        return all(str(keyword).lower() in text.lower() for keyword in keywords), "keyword inclusion"

    if "forbidden_words" in iid:
        words = _kwargs_value(kwargs, "forbidden_words")
        if not words:
            return None, "missing forbidden words"
        if isinstance(words, str):
            words = [words]
        return not any(str(word).lower() in text.lower() for word in words), "forbidden words"

    if "json" in iid:
        try:
            import json

            json.loads(text)
            return True, "json parse"
        except Exception:
            return False, "json parse"

    return None, "unsupported by fallback checker"


def _compare_count(count: int, target: int, relation: str) -> bool:
    relation = relation.lower().replace("_", " ")
    if "less" in relation or "fewer" in relation or "below" in relation:
        return count < target
    if "more" in relation or "above" in relation or "at least" in relation:
        return count >= target
    if "equal" in relation or "exact" in relation:
        return count == target
    return count >= target


def evaluate_ifeval_fallback(sample: dict, response: str) -> dict:
    instruction_ids = sample.get("instruction_id_list") or []
    kwargs_list = sample.get("kwargs") or []
    results = []
    for idx, instruction_id in enumerate(instruction_ids):
        kwargs = kwargs_list[idx] if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict) else {}
        passed, reason = check_instruction(instruction_id, kwargs, response)
        results.append({"instruction_id": instruction_id, "passed": passed, "reason": reason})
    supported = [r for r in results if r["passed"] is not None]
    unsupported = [r for r in results if r["passed"] is None]
    strict_pass = bool(supported) and all(r["passed"] for r in supported) and not unsupported
    loose_supported_pass = bool(supported) and all(r["passed"] for r in supported)
    return {
        "strict_pass": strict_pass,
        "loose_supported_pass": loose_supported_pass,
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "instruction_results": results,
        "warning": "Fallback checker is not official IFEval; use official evaluator for paper numbers.",
    }
