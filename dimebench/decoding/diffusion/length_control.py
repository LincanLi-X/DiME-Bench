"""Fixed-canvas construction metadata for diffusion generation and infilling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanvasLayout:
    """Token offsets for a fixed masked span inside conditioned context."""

    left_length: int
    generated_length: int
    right_length: int = 0

    def __post_init__(self) -> None:
        if self.left_length < 0 or self.right_length < 0:
            raise ValueError("context lengths cannot be negative")
        if self.generated_length <= 0:
            raise ValueError("generated_length must be positive")

    @property
    def generated_start(self) -> int:
        return self.left_length

    @property
    def generated_end(self) -> int:
        return self.left_length + self.generated_length

    @property
    def total_length(self) -> int:
        return self.left_length + self.generated_length + self.right_length

    @property
    def generated_slice(self) -> slice:
        return slice(self.generated_start, self.generated_end)
