from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


OPTION_RE = re.compile(
    r"(?:answer|option|choice|final answer)\s*(?:is|:)?\s*[\(\[]?\s*([A-J])\s*[\)\].:]?",
    flags=re.IGNORECASE,
)


def extract_choice(text: str, valid_letters: list[str] | None = None) -> str | None:
    valid = set(valid_letters or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    matches = OPTION_RE.findall(text)
    for match in matches:
        letter = match.upper()
        if letter in valid:
            return letter
    # Prefer standalone capital letters near the end.
    tail = text.strip()[-120:]
    for token in re.findall(r"\b([A-J])\b", tail):
        letter = token.upper()
        if letter in valid:
            return letter
    return None


def extract_number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    final = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, re.I)
    if final:
        return normalize_number(final[-1])
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return normalize_number(nums[-1])
    return None


def normalize_number(text: str) -> str:
    text = str(text).strip().replace(",", "")
    text = re.sub(r"[^0-9+\-./]", "", text)
    if "/" in text and text.count("/") == 1:
        num, den = text.split("/")
        try:
            return str(Decimal(num) / Decimal(den)).rstrip("0").rstrip(".")
        except (InvalidOperation, ZeroDivisionError):
            return text
    try:
        dec = Decimal(text)
        return format(dec.normalize(), "f").rstrip("0").rstrip(".") or "0"
    except InvalidOperation:
        return text


def exact_numeric_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return normalize_number(pred) == normalize_number(gold)
