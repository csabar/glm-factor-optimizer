"""Spark train/validation/holdout split helper."""

from __future__ import annotations

from typing import Any

from ._deps import require_pyspark


def split(
    df: Any,
    train: float = 0.6,
    validation: float = 0.2,
    holdout: float = 0.2,
    *,
    seed: int = 42,
    time: str | None = None,
) -> tuple[Any, Any, Any]:
    """Return train, validation, and holdout Spark dataframes."""

    total = train + validation + holdout
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train + validation + holdout must equal 1.0.")
    if time is None:
        train_df, validation_df, holdout_df = df.randomSplit([train, validation, holdout], seed=seed)
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
    train_df = ordered.where(F.col("__p") <= train).drop("__rn", "__n", "__p")
    validation_df = ordered.where((F.col("__p") > train) & (F.col("__p") <= train + validation)).drop(
        "__rn",
        "__n",
        "__p",
    )
    holdout_df = ordered.where(F.col("__p") > train + validation).drop("__rn", "__n", "__p")
    return train_df, validation_df, holdout_df
