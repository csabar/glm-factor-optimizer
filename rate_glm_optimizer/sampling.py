"""Generic sampling helpers for representative modeling datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def stratified_sample(
    df: pd.DataFrame,
    strata: str | Sequence[str],
    *,
    size: int | None = None,
    frac: float | None = None,
    min_per_group: int = 1,
    seed: int = 42,
    weight_col: str | None = None,
    add_sample_weight: bool = False,
    sample_weight_col: str = "sample_weight",
) -> pd.DataFrame:
    """Return a stratified sample with optional weighted row selection."""

    if (size is None) == (frac is None):
        raise ValueError("Provide exactly one of size or frac.")
    if min_per_group < 0:
        raise ValueError("min_per_group must be non-negative.")
    if frac is not None and not 0.0 < frac <= 1.0:
        raise ValueError("frac must be in (0, 1].")
    if size is not None and size <= 0:
        raise ValueError("size must be positive.")
    if df.empty:
        raise ValueError("Cannot sample an empty dataframe.")

    strata_cols = _as_list(strata)
    for column in strata_cols:
        if column not in df.columns:
            raise KeyError(f"Column {column!r} is missing.")
    if weight_col is not None and weight_col not in df.columns:
        raise KeyError(f"Column {weight_col!r} is missing.")

    counts = df.groupby(strata_cols, dropna=False, sort=False).size().rename("rows").reset_index()
    targets = _target_sizes(counts["rows"].to_numpy(), size=size, frac=frac, min_per_group=min_per_group)
    counts["sample_rows"] = targets

    if size is not None and int(counts["sample_rows"].sum()) > size:
        raise ValueError("size is too small for the requested min_per_group.")

    sampled: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    grouped = df.groupby(strata_cols, dropna=False, sort=False)
    for index, (_, group) in enumerate(grouped):
        n_rows = int(counts.iloc[index]["sample_rows"])
        if n_rows <= 0:
            continue
        weights = None
        if weight_col is not None:
            weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            weights = weights if float(weights.sum()) > 0.0 else None
        sample = group.sample(
            n=min(n_rows, len(group)),
            replace=False,
            weights=weights,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        if add_sample_weight:
            sample = sample.copy()
            fraction = max(len(sample) / len(group), 1e-9)
            sample[sample_weight_col] = 1.0 / fraction
        sampled.append(sample)

    if not sampled:
        return df.iloc[0:0].copy()
    result = pd.concat(sampled, axis=0)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def missing_strata(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    strata: str | Sequence[str],
) -> pd.DataFrame:
    """Return strata combinations present in population but absent in sample."""

    strata_cols = _as_list(strata)
    population_keys = population[strata_cols].drop_duplicates()
    sample_keys = sample[strata_cols].drop_duplicates()
    merged = population_keys.merge(sample_keys, how="left", indicator=True, on=strata_cols)
    return merged.loc[merged["_merge"] == "left_only", strata_cols].reset_index(drop=True)


def _target_sizes(
    counts: np.ndarray,
    *,
    size: int | None,
    frac: float | None,
    min_per_group: int,
) -> np.ndarray:
    if frac is not None:
        targets = np.ceil(counts * frac).astype(int)
        targets = np.maximum(targets, min_per_group)
        return np.minimum(targets, counts)

    assert size is not None
    if size > int(counts.sum()):
        raise ValueError("size cannot exceed the dataframe row count.")
    minimum = np.minimum(counts, min_per_group).astype(int)
    if int(minimum.sum()) > size:
        raise ValueError("size is too small for the requested min_per_group.")
    remaining = size - int(minimum.sum())
    available = counts - minimum
    if remaining <= 0 or int(available.sum()) == 0:
        return minimum
    raw = available / available.sum() * remaining
    add = np.floor(raw).astype(int)
    remainder = remaining - int(add.sum())
    if remainder > 0:
        order = np.argsort(-(raw - add))
        for index in order[:remainder]:
            add[index] += 1
    return np.minimum(minimum + add, counts).astype(int)


def _as_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)
