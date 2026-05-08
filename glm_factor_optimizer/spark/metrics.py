"""Spark-native validation metrics."""

from __future__ import annotations

from typing import Any

from ._deps import require_pyspark


def model_deviance(
    df: Any,
    target: str,
    prediction: str,
    *,
    family: str,
    weight: str | None = None,
) -> float:
    """Return family-specific mean deviance as a driver float."""

    spark = require_pyspark()
    F = spark.functions
    y = F.col(target).cast("double")
    mu = F.greatest(F.col(prediction).cast("double"), F.lit(1e-9))
    family = family.lower().strip()

    if family == "poisson":
        dev = F.when(y > 0, 2.0 * (y * F.log(y / mu) - (y - mu))).otherwise(2.0 * mu)
    elif family == "gamma":
        safe_y = F.greatest(y, F.lit(1e-9))
        dev = 2.0 * (((safe_y - mu) / mu) - F.log(safe_y / mu))
    elif family == "gaussian":
        dev = (y - mu) * (y - mu)
    else:
        raise ValueError("family must be 'poisson', 'gamma', or 'gaussian'.")

    if weight is None:
        row = df.select(F.avg(dev).alias("deviance")).first()
    else:
        w = F.col(weight).cast("double")
        row = df.select((F.sum(dev * w) / F.sum(w)).alias("deviance")).first()
    return float(row["deviance"])


def summary(
    df: Any,
    target: str,
    prediction: str,
    exposure: str | None = None,
    family: str = "poisson",
    weight: str | None = None,
) -> Any:
    """Return a one-row Spark dataframe with actual/predicted summary."""

    spark = require_pyspark()
    F = spark.functions
    aggs = [
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.col(target).cast("double")).alias("actual"),
        F.sum(F.col(prediction).cast("double")).alias("predicted"),
    ]
    if exposure is not None:
        aggs.append(F.sum(F.col(exposure).cast("double")).alias("exposure"))
    result = df.agg(*aggs)
    result = result.withColumn("actual_to_predicted", F.col("actual") / F.greatest(F.col("predicted"), F.lit(1e-9)))
    if exposure is not None:
        result = result.withColumn("actual_rate", F.col("actual") / F.greatest(F.col("exposure"), F.lit(1e-9)))
        result = result.withColumn("predicted_rate", F.col("predicted") / F.greatest(F.col("exposure"), F.lit(1e-9)))
    return result.withColumn("deviance", F.lit(model_deviance(df, target, prediction, family=family, weight=weight)))


def calibration(
    df: Any,
    target: str,
    prediction: str,
    exposure: str | None = None,
    bins: int = 10,
) -> Any:
    """Return a Spark dataframe grouped by predicted level quantile."""

    spark = require_pyspark()
    F = spark.functions
    W = spark.Window
    rank_value = (
        F.col(prediction).cast("double") / F.greatest(F.col(exposure).cast("double"), F.lit(1e-9))
        if exposure is not None
        else F.col(prediction).cast("double")
    )
    work = df.withColumn("__rank_value", rank_value)
    work = work.withColumn("bin", F.ntile(bins).over(W.orderBy("__rank_value")))
    aggs = [
        F.sum(F.col(target).cast("double")).alias("actual"),
        F.sum(F.col(prediction).cast("double")).alias("predicted"),
        F.count(F.lit(1)).alias("rows"),
    ]
    if exposure is not None:
        aggs.append(F.sum(F.col(exposure).cast("double")).alias("exposure"))
    table = work.groupBy("bin").agg(*aggs).orderBy("bin")
    table = table.withColumn("actual_to_predicted", F.col("actual") / F.greatest(F.col("predicted"), F.lit(1e-9)))
    if exposure is not None:
        table = table.withColumn("actual_rate", F.col("actual") / F.greatest(F.col("exposure"), F.lit(1e-9)))
        table = table.withColumn("predicted_rate", F.col("predicted") / F.greatest(F.col("exposure"), F.lit(1e-9)))
    else:
        table = table.withColumn("actual_mean", F.col("actual") / F.greatest(F.col("rows"), F.lit(1)))
        table = table.withColumn("predicted_mean", F.col("predicted") / F.greatest(F.col("rows"), F.lit(1)))
    return table
