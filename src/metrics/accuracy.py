from __future__ import annotations


def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("is_correct")) / len(rows)
