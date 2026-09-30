"""Frozen deterministic source generator for the synthetic-repair dataset."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

GENERATOR_VERSION = "1.0.0"

# Each pair is deliberately local: only the factual clause changes.  This makes
# target/non-target spans derivable and keeps preservation diagnostics auditable.
_FACTS = (
    ("water freezes at 10 degrees Celsius", "water freezes at 0 degrees Celsius"),
    ("Earth has two natural moons", "Earth has one natural moon"),
    ("a week contains eight days", "a week contains seven days"),
    ("a triangle has four sides", "a triangle has three sides"),
    ("the chemical symbol for gold is Ag", "the chemical symbol for gold is Au"),
    ("the chemical symbol for silver is Au", "the chemical symbol for silver is Ag"),
    ("the human heart has three chambers", "the human heart has four chambers"),
    (
        "Mars is the closest planet to the Sun",
        "Mercury is the closest planet to the Sun",
    ),
    ("Jupiter is the smallest planet", "Mercury is the smallest planet"),
    ("the Pacific is the smallest ocean", "the Pacific is the largest ocean"),
    (
        "light travels slower than sound in air",
        "light travels faster than sound in air",
    ),
    (
        "pure water boils at 50 degrees Celsius at sea level",
        "pure water boils at 100 degrees Celsius at sea level",
    ),
    ("an adult human normally has 12 teeth", "an adult human normally has 32 teeth"),
    ("a leap year has 364 days", "a leap year has 366 days"),
    (
        "the binary representation of decimal two is 11",
        "the binary representation of decimal two is 10",
    ),
    ("a right angle measures 45 degrees", "a right angle measures 90 degrees"),
    ("the square root of 81 is 8", "the square root of 81 is 9"),
    (
        "the sum of the interior angles of a triangle is 360 degrees",
        "the sum of the interior angles of a triangle is 180 degrees",
    ),
    ("DNA stands for dynamic nuclear acid", "DNA stands for deoxyribonucleic acid"),
    (
        "plants absorb oxygen for photosynthesis",
        "plants absorb carbon dioxide for photosynthesis",
    ),
    (
        "the SI unit of electric current is the volt",
        "the SI unit of electric current is the ampere",
    ),
    ("the SI unit of force is the watt", "the SI unit of force is the newton"),
    (
        "the speed of light in vacuum is about 300 kilometers per second",
        "the speed of light in vacuum is about 300,000 kilometers per second",
    ),
    ("the atomic number of carbon is 8", "the atomic number of carbon is 6"),
    ("the atomic number of oxygen is 6", "the atomic number of oxygen is 8"),
    (
        "the freezing point of water on the Fahrenheit scale is 0 degrees",
        "the freezing point of water on the Fahrenheit scale is 32 degrees",
    ),
    ("one kilometer equals 100 meters", "one kilometer equals 1,000 meters"),
    ("one hour contains 100 minutes", "one hour contains 60 minutes"),
    (
        "the Earth completes one rotation in about 365 days",
        "the Earth completes one rotation in about 24 hours",
    ),
    (
        "the Earth completes one orbit around the Sun in about 24 hours",
        "the Earth completes one orbit around the Sun in about 365 days",
    ),
    (
        "sound can travel through a perfect vacuum",
        "sound cannot travel through a perfect vacuum",
    ),
    (
        "the largest organ of the human body is the liver",
        "the largest organ of the human body is the skin",
    ),
)


def generate_synthetic_repair_rows() -> Iterator[dict[str, Any]]:
    """Yield the frozen v1 contradiction-repair rows in stable order."""
    for index, (incorrect, correct) in enumerate(_FACTS, start=1):
        source = f"The draft report states that {incorrect}."
        reference = f"The draft report states that {correct}."
        yield {
            "id": f"synthetic-repair-{index:04d}",
            "source_text": source,
            "instruction": (
                "Correct the factual contradiction using the evidence. "
                "Preserve every unrelated part of the draft."
            ),
            "reference_edit": reference,
            "edit_type": "contradiction_repair",
            "evidence": [f"The verified evidence states that {correct}."],
            "metadata": {
                "generator": "dimebench.synthetic_repair",
                "generator_version": GENERATOR_VERSION,
                "case_index": index,
            },
        }


__all__ = ["GENERATOR_VERSION", "generate_synthetic_repair_rows"]
