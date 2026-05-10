"""Spark interaction diagnostics with bounded pandas output."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from ._deps import require_pyspark


def pair_diagnostics(
    df: object,
    *,
    factors: Sequence[str],
    target: str,
    prediction: str,
    exposure: str | None = None,
    weight: str | None = None,
    min_bin_size: float = 1.0,
) -> pd.DataFrame:
    """Score observed-vs-predicted deviations for all Spark factor pairs."""

    rows: list[dict[str, float | int | str]] = []
    for factor_a, factor_b in combinations(factors, 2):
        table = _pair_table(
            df,
            factor_a=factor_a,
            factor_b=factor_b,
            target=target,
            prediction=prediction,
            exposure=exposure,
            weight=weight,
        )
        table = table.loc[table["bin_size"] >= min_bin_size].copy()
        if table.empty:
            continue
        abs_deviation = table["actual_to_predicted"].sub(1.0).abs()
        weights = table["bin_size"].clip(lower=1e-9)
        rows.append(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "pair": f"{factor_a}:{factor_b}",
                "cells": int(len(table)),
                "total_bin_size": float(table["bin_size"].sum()),
                "mean_abs_deviation": float(np.average(abs_deviation, weights=weights)),
                "max_abs_deviation": float(abs_deviation.max()),
                "rmse_deviation": float(np.sqrt(np.average(np.square(abs_deviation), weights=weights))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "factor_a",
                "factor_b",
                "pair",
                "cells",
                "total_bin_size",
                "mean_abs_deviation",
                "max_abs_deviation",
                "rmse_deviation",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["mean_abs_deviation", "max_abs_deviation", "pair"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def find_interactions(
    train_df: object,
    validation_df: object,
    *,
    factors: Sequence[str],
    target: str,
    prediction: str,
    exposure: str | None = None,
    weight: str | None = None,
    min_bin_size: float = 1.0,
) -> pd.DataFrame:
    """Return Spark pair interaction candidates present in train and validation."""

    train = pair_diagnostics(
        train_df,
        factors=factors,
        target=target,
        prediction=prediction,
        exposure=exposure,
        weight=weight,
        min_bin_size=min_bin_size,
    )
    validation = pair_diagnostics(
        validation_df,
        factors=factors,
        target=target,
        prediction=prediction,
        exposure=exposure,
        weight=weight,
        min_bin_size=min_bin_size,
    )
    if train.empty or validation.empty:
        return pd.DataFrame(
            columns=[
                "factor_a",
                "factor_b",
                "pair",
                "train_mean_abs_deviation",
                "validation_mean_abs_deviation",
                "score",
            ]
        )
    merged = train.merge(
        validation,
        on=["factor_a", "factor_b", "pair"],
        suffixes=("_train", "_validation"),
    )
    merged["score"] = np.minimum(
        merged["mean_abs_deviation_train"],
        merged["mean_abs_deviation_validation"],
    )
    return (
        merged.rename(
            columns={
                "mean_abs_deviation_train": "train_mean_abs_deviation",
                "mean_abs_deviation_validation": "validation_mean_abs_deviation",
                "max_abs_deviation_train": "train_max_abs_deviation",
                "max_abs_deviation_validation": "validation_max_abs_deviation",
                "rmse_deviation_train": "train_rmse_deviation",
                "rmse_deviation_validation": "validation_rmse_deviation",
            }
        )
        .sort_values(["score", "validation_mean_abs_deviation", "pair"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _pair_table(
    df: object,
    *,
    factor_a: str,
    factor_b: str,
    target: str,
    prediction: str,
    exposure: str | None,
    weight: str | None,
) -> pd.DataFrame:
    spark = require_pyspark()
    F = spark.functions
    left = F.coalesce(F.col(factor_a).cast("string"), F.lit("missing")).alias(factor_a)
    right = F.coalesce(F.col(factor_b).cast("string"), F.lit("missing")).alias(factor_b)
    work = df.select(
        left,
        right,
        F.col(target).cast("double").alias("__target"),
        F.col(prediction).cast("double").alias("__prediction"),
        *([F.col(exposure).cast("double").alias("__exposure")] if exposure is not None else []),
        *([F.col(weight).cast("double").alias("__weight")] if weight is not None else []),
    )
    group_cols = [factor_a, factor_b]
    if exposure is not None:
        table = work.groupBy(*group_cols).agg(
            F.sum("__exposure").alias("bin_size"),
            F.sum("__target").alias("actual"),
            F.sum("__prediction").alias("predicted"),
            F.count(F.lit(1)).alias("rows"),
        )
    elif weight is not None:
        table = work.groupBy(*group_cols).agg(
            F.sum("__weight").alias("bin_size"),
            F.sum(F.col("__target") * F.col("__weight")).alias("actual"),
            F.sum(F.col("__prediction") * F.col("__weight")).alias("predicted"),
            F.count(F.lit(1)).alias("rows"),
        )
    else:
        table = work.groupBy(*group_cols).agg(
            F.count(F.lit(1)).cast("double").alias("bin_size"),
            F.sum("__target").alias("actual"),
            F.sum("__prediction").alias("predicted"),
            F.count(F.lit(1)).alias("rows"),
        )
    table = table.withColumn("actual_to_predicted", F.col("actual") / F.greatest(F.col("predicted"), F.lit(1e-9)))
    return table.toPandas()
