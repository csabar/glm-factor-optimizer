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
    time_split: str = "exact",
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
    time_split:
        ``"exact"`` keeps ordered row-number semantics. ``"approximate"``
        uses approximate time quantile thresholds and avoids a global window.

    Returns
    -------
    tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]
        Train, validation, and holdout Spark dataframes.
    """

    train_value = _resolve_fraction("train", train, train_fraction, 0.6)
    validation_value = _resolve_fraction("validation", validation, validation_fraction, 0.2)
    holdout_value = _resolve_fraction("holdout", holdout, holdout_fraction, 0.2)
    time_split = _resolve_time_split(time_split)

    total = train_value + validation_value + holdout_value
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train + validation + holdout must equal 1.0.")
    if time is None:
        train_df, validation_df, holdout_df = df.randomSplit(
            [train_value, validation_value, holdout_value],
            seed=seed,
        )
        return train_df, validation_df, holdout_df

    if time_split == "approximate":
        return _approximate_time_split(df, time, train_value, validation_value)

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


def _approximate_time_split(
    df: Any,
    time: str,
    train: float,
    validation: float,
) -> tuple[Any, Any, Any]:
    spark = require_pyspark()
    F = spark.functions
    empty = df.where(F.lit(False))
    if train <= 0 and validation <= 0:
        return empty, empty, df
    if train >= 1:
        return df, empty, empty

    value = _time_value(df, time)
    work = df.withColumn("__split_time_value", value)
    train_cut, validation_cut = _time_cutpoints(work, "__split_time_value", train, train + validation)
    time_col = F.col("__split_time_value")

    if train <= 0:
        train_df = work.where(F.lit(False))
    else:
        train_df = work.where(time_col.isNull() | (time_col <= F.lit(train_cut)))

    if validation <= 0:
        validation_df = work.where(F.lit(False))
    else:
        lower = F.lit(True) if train <= 0 else time_col > F.lit(train_cut)
        validation_df = work.where(time_col.isNotNull() & lower & (time_col <= F.lit(validation_cut)))

    holdout_df = work.where(time_col.isNotNull() & (time_col > F.lit(validation_cut)))
    return (
        train_df.drop("__split_time_value"),
        validation_df.drop("__split_time_value"),
        holdout_df.drop("__split_time_value"),
    )


def _time_value(df: Any, time: str) -> Any:
    spark = require_pyspark()
    F = spark.functions
    simple_type = df.schema[time].dataType.simpleString().split("(")[0]
    if simple_type in {"byte", "short", "integer", "long", "float", "double", "decimal"}:
        return F.col(time).cast("double")
    if simple_type in {"date", "timestamp", "timestamp_ntz"}:
        return F.unix_timestamp(F.col(time).cast("timestamp")).cast("double")
    raise ValueError("Spark approximate time split requires numeric, date, or timestamp time columns.")


def _time_cutpoints(df: Any, column: str, train: float, validation_end: float) -> tuple[float, float]:
    probabilities = sorted({float(min(max(value, 0.0), 1.0)) for value in [train, validation_end]})
    values = [float(value) for value in df.approxQuantile(column, probabilities, 0.01)]
    if not values:
        raise ValueError("Cannot split an empty or all-null time column.")
    by_probability = dict(zip(probabilities, values, strict=False))
    train_cut = by_probability[float(min(max(train, 0.0), 1.0))]
    validation_cut = by_probability[float(min(max(validation_end, 0.0), 1.0))]
    return train_cut, validation_cut


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
