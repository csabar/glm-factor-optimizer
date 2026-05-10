"""Spark train/validation/holdout split helper."""

from __future__ import annotations

from typing import Any

from ._deps import require_pyspark

_UNSET = object()


def split(
    df: Any,
    train: float | object = _UNSET,
    validation: float | object = _UNSET,
    holdout: float | object = _UNSET,
    *,
    train_fraction: float | None = None,
    validation_fraction: float | None = None,
    holdout_fraction: float | None = None,
    seed: int | None = 42,
    time: str | None = None,
) -> tuple[Any, Any, Any]:
    """Return train, validation, and holdout Spark dataframes.

    Parameters
    ----------
    df:
        Spark dataframe to split.
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

    Returns
    -------
    tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]
        Train, validation, and holdout Spark dataframes.
    """

    train_value = _resolve_fraction("train", train, train_fraction, 0.6)
    validation_value = _resolve_fraction("validation", validation, validation_fraction, 0.2)
    holdout_value = _resolve_fraction("holdout", holdout, holdout_fraction, 0.2)

    total = train_value + validation_value + holdout_value
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train + validation + holdout must equal 1.0.")
    if time is None:
        train_df, validation_df, holdout_df = df.randomSplit(
            [train_value, validation_value, holdout_value],
            seed=seed,
        )
        return train_df, validation_df, holdout_df

    spark = require_pyspark()
    F = spark.functions
    W = spark.Window
    ordered_window = W.orderBy(time)
    all_rows_window = ordered_window.rowsBetween(W.unboundedPreceding, W.unboundedFollowing)
    ordered = (
        df.withColumn("__rn", F.row_number().over(ordered_window))
        .withColumn("__n", F.count(F.lit(1)).over(all_rows_window))
        .withColumn("__p", F.col("__rn") / F.col("__n"))
    )
    train_df = ordered.where(F.col("__p") <= train_value).drop("__rn", "__n", "__p")
    validation_df = ordered.where((F.col("__p") > train_value) & (F.col("__p") <= train_value + validation_value)).drop(
        "__rn",
        "__n",
        "__p",
    )
    holdout_df = ordered.where(F.col("__p") > train_value + validation_value).drop("__rn", "__n", "__p")
    return train_df, validation_df, holdout_df


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
