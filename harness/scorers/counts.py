"""Solved-problem counts, and the rule for reading a difference between two of them."""

from __future__ import annotations

from dataclasses import dataclass

# research-integrity.md: with 20 problems per set, a difference of one or two problems is noise.
NOISE_PROBLEMS = 2


@dataclass(frozen=True)
class SolvedCount:
    solved: int
    n: int
    errors: int = 0  # requests that failed (counted as unsolved)

    def __post_init__(self) -> None:
        if not 0 <= self.solved <= self.n or self.errors < 0 or self.errors > self.n:
            raise ValueError(
                f"impossible count: solved={self.solved} n={self.n} errors={self.errors}"
            )

    def __str__(self) -> str:
        return f"{self.solved}/{self.n}"


def difference_is_noise(a: SolvedCount, b: SolvedCount) -> bool:
    """True when two counts out of the same number of problems differ by no more than two."""
    if a.n != b.n:
        raise ValueError("counts are over different numbers of problems")
    return abs(a.solved - b.solved) <= NOISE_PROBLEMS


def describe_difference(a: SolvedCount, b: SolvedCount) -> str:
    """Text for a report, so a small difference is never presented as a result."""
    d = a.solved - b.solved
    if difference_is_noise(a, b):
        return f"{a} vs {b}: within noise ({'no difference' if d == 0 else f'{abs(d)} problem(s)'})"
    return f"{a} vs {b}: {abs(d)} problems apart"
