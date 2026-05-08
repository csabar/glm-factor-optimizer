"""Reusable objective penalties for factor binning optimization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

PenaltyContext = dict[str, Any]
PenaltyFunction = Callable[[PenaltyContext], float]
PenaltyInput = PenaltyFunction | Sequence[PenaltyFunction] | Mapping[str, PenaltyFunction]


def bin_count_penalty(
    per_bin: float = 0.001,
    max_bins: int | None = None,
    excess_penalty: float | None = None,
) -> PenaltyFunction:
    """Return a penalty for model complexity from the number of bins."""

    def calculate(context: PenaltyContext) -> float:
        bin_count = int(context["bin_count"])
        value = bin_count * per_bin
        if max_bins is not None:
            value += max(bin_count - max_bins, 0) * (excess_penalty if excess_penalty is not None else per_bin)
        return float(value)

    return calculate


def small_bin_size_penalty(
    min_size: float,
    penalty: float = 0.01,
    size_column: str = "bin_size",
) -> PenaltyFunction:
    """Return a penalty for bins with small exposure, weight, or row count."""

    def calculate(context: PenaltyContext) -> float:
        table = context["bin_table"]
        return float((table[size_column] < min_size).sum() * penalty)

    return calculate


def small_count_penalty(
    min_count: float,
    penalty: float = 0.01,
    count_column: str = "actual",
) -> PenaltyFunction:
    """Return a penalty for bins with a small observed target total."""

    def calculate(context: PenaltyContext) -> float:
        table = context["bin_table"]
        return float((table[count_column] < min_count).sum() * penalty)

    return calculate


def train_validation_gap_penalty(
    tolerance: float = 0.0,
    scale: float = 1.0,
) -> PenaltyFunction:
    """Return a penalty for validation deviance worse than train deviance."""

    def calculate(context: PenaltyContext) -> float:
        gap = context["validation_deviance"] - context["train_deviance"] - tolerance
        return float(max(gap, 0.0) * scale)

    return calculate


def unstable_relativity_penalty(
    max_ratio: float = 3.0,
    penalty: float = 0.01,
) -> PenaltyFunction:
    """Return a penalty for large adjacent observed-level jumps across bins."""

    if max_ratio <= 1.0:
        raise ValueError("max_ratio must be greater than 1.0.")

    def calculate(context: PenaltyContext) -> float:
        table = context["bin_table"]
        if len(table) < 2:
            return 0.0
        measure = np.asarray(table["bin_size"], dtype=float)
        actual = np.asarray(table["actual"], dtype=float)
        level = np.clip((actual + 1e-9) / np.clip(measure, 1e-9, None), 1e-9, None)
        jumps = np.abs(np.diff(np.log(level)))
        excess = np.maximum(jumps - np.log(max_ratio), 0.0)
        return float(excess.sum() * penalty)

    return calculate


small_target_penalty = small_count_penalty

