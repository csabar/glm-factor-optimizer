"""Small train/validation/holdout split helper."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._backend import is_spark_dataframe

_UNSET = object()


def split(
    df: pd.DataFrame,
    train: float | object = _UNSET,
    validation: float | object = _UNSET,
    holdout: float | object = _UNSET,
    *,
    train_fraction: float | None = None,
    validation_fraction: float | None = None,
    holdout_fraction: float | None = None,
    seed: int | None = 42,
    time: str | None = None,
    time_split: str = "exact",
) -> tuple[Any, Any, Any]:
    """Return train, validation, and holdout dataframes.

    Parameters
    ----------
    df:
        Pandas or Spark dataframe to split.
    train:
        Fraction assigned to the training sample.
    validation:
        Fraction assigned to the validation sample.
    holdout:
        Fraction assigned to the holdout sample.
    train_fraction:
        Alias for ``train``.
    validation_fraction:
        Alias for ``validation``.
    holdout_fraction:
        Alias for ``holdout``.
    seed:
        Random seed used when ``time`` is not supplied.
    time:
        Optional time-like column used to split ordered rows instead of random
        rows.
    time_split:
        Time split strategy. Pandas supports only ``"exact"``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Train, validation, and holdout samples.
    """

    train_value = _resolve_fraction("train", train, train_fraction, 0.6)
    validation_value = _resolve_fraction("validation", validation, validation_fraction, 0.2)
    holdout_value = _resolve_fraction("holdout", holdout, holdout_fraction, 0.2)
    time_split = _resolve_time_split(time_split)

    if is_spark_dataframe(df):
        from .spark.split import split as spark_split

        return spark_split(
            df,
            train=train_value,
            validation=validation_value,
            holdout=holdout_value,
            seed=seed,
            time=time,
            time_split=time_split,
        )

    if time_split != "exact":
        raise ValueError("Pandas split supports only time_split='exact'.")

    total = train_value + validation_value + holdout_value
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train + validation + holdout must equal 1.0.")
    if len(df) == 0:
        raise ValueError("Cannot split an empty dataframe.")

    ordered = df.sort_values(time).reset_index(drop=True) if time else df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)
    train_end = int(round(len(ordered) * train_value))
    validation_end = train_end + int(round(len(ordered) * validation_value))
    validation_end = min(validation_end, len(ordered))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def _resolve_fraction(
    name: str,
    value: float | object,
    alias: float | None,
    default: float,
) -> float:
    if value is not _UNSET and alias is not None:
        raise ValueError(f"Use either {name} or {name}_fraction, not both.")
    if alias is not None:
        return float(alias)
    if value is not _UNSET:
        return float(value)
    return default


def _resolve_time_split(value: str) -> str:
    normalized = str(value).lower().strip()
    if normalized not in {"exact", "approximate"}:
        raise ValueError("time_split must be 'exact' or 'approximate'.")
    return normalized
