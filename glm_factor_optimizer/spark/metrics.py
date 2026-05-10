"""Spark-native validation metrics."""

from __future__ import annotations

import math
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
    """Return family-specific mean deviance as a driver float.

    Parameters
    ----------
    df:
        Scored Spark dataframe.
    target:
        Observed outcome column.
    prediction:
        Predicted outcome column.
    family:
        GLM family name.
    weight:
        Optional row-weight column.

    Returns
    -------
    float
        Weighted or unweighted mean deviance collected to the driver.
    """

    spark = require_pyspark()
    F = spark.functions
    dev = deviance_expression(target, prediction, family=family)
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
    """Return a one-row Spark dataframe with observed/predicted summary.

    Parameters
    ----------
    df:
        Scored Spark dataframe.
    target:
        Observed outcome column.
    prediction:
        Predicted outcome column.
    exposure:
        Optional exposure column for exposure-adjusted summary columns.
    family:
        GLM family used for deviance scoring.
    weight:
        Optional row-weight column.

    Returns
    -------
    pyspark.sql.DataFrame
        One-row Spark dataframe with totals, ratio, and deviance.
    """

    return summary_frame(
        df,
        target=target,
        prediction=prediction,
        exposure=exposure,
        family=family,
        weight=weight,
        include_errors=False,
    )


def summary_frame(
    df: Any,
    *,
    target: str,
    prediction: str,
    exposure: str | None = None,
    family: str = "poisson",
    weight: str | None = None,
    include_errors: bool = False,
) -> Any:
    """Return one-row Spark metrics, optionally including MAE and RMSE."""

    spark = require_pyspark()
    F = spark.functions
    y = F.col(target).cast("double")
    mu = F.col(prediction).cast("double")
    dev = deviance_expression(target, prediction, family=family)
    error = y - mu
    aggs = [
        F.count(F.lit(1)).alias("rows"),
        F.sum(y).alias("actual"),
        F.sum(mu).alias("predicted"),
    ]
    if exposure is not None:
        aggs.append(F.sum(F.col(exposure).cast("double")).alias("exposure"))
    if weight is None:
        aggs.append(F.avg(dev).alias("deviance"))
        if include_errors:
            aggs.extend([
                F.avg(F.abs(error)).alias("mae"),
                F.sqrt(F.avg(error * error)).alias("rmse"),
            ])
    else:
        w = F.col(weight).cast("double")
        aggs.append((F.sum(dev * w) / F.sum(w)).alias("deviance"))
        if include_errors:
            aggs.extend([
                (F.sum(F.abs(error) * w) / F.sum(w)).alias("mae"),
                F.sqrt(F.sum(error * error * w) / F.sum(w)).alias("rmse"),
            ])

    result = df.agg(*aggs)
    result = result.withColumn("actual_to_predicted", F.col("actual") / F.greatest(F.col("predicted"), F.lit(1e-9)))
    if exposure is not None:
        result = result.withColumn("actual_rate", F.col("actual") / F.greatest(F.col("exposure"), F.lit(1e-9)))
        result = result.withColumn("predicted_rate", F.col("predicted") / F.greatest(F.col("exposure"), F.lit(1e-9)))
    else:
        result = result.withColumn("actual_mean", F.col("actual") / F.greatest(F.col("rows"), F.lit(1)))
        result = result.withColumn("predicted_mean", F.col("predicted") / F.greatest(F.col("rows"), F.lit(1)))
    return result


def deviance_expression(target: str, prediction: str, *, family: str) -> Any:
    """Return a Spark column expression for family-specific deviance."""

    spark = require_pyspark()
    F = spark.functions
    y = F.col(target).cast("double")
    mu = F.greatest(F.col(prediction).cast("double"), F.lit(1e-9))
    normalized = family.lower().strip()
    if normalized == "poisson":
        return F.when(y > 0, 2.0 * (y * F.log(y / mu) - (y - mu))).otherwise(2.0 * mu)
    if normalized == "gamma":
        safe_y = F.greatest(y, F.lit(1e-9))
        return 2.0 * (((safe_y - mu) / mu) - F.log(safe_y / mu))
    if normalized == "gaussian":
        return (y - mu) * (y - mu)
    raise ValueError("family must be 'poisson', 'gamma', or 'gaussian'.")


def calibration(
    df: Any,
    target: str,
    prediction: str,
    exposure: str | None = None,
    bins: int = 10,
) -> Any:
    """Return a Spark dataframe grouped by predicted level quantile.

    Parameters
    ----------
    df:
        Scored Spark dataframe.
    target:
        Observed outcome column.
    prediction:
        Predicted outcome column.
    exposure:
        Optional exposure column used to rank by prediction divided by
        exposure.
    bins:
        Number of quantile bands to create.

    Returns
    -------
    pyspark.sql.DataFrame
        Calibration table by prediction band.
    """

    spark = require_pyspark()
    F = spark.functions
    rank_value = (
        F.col(prediction).cast("double") / F.greatest(F.col(exposure).cast("double"), F.lit(1e-9))
        if exposure is not None
        else F.col(prediction).cast("double")
    )
    work = df.withColumn("__rank_value", rank_value)
    work = _with_approx_quantile_bin(work, "__rank_value", bins)
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


def _with_approx_quantile_bin(df: Any, column: str, bins: int) -> Any:
    spark = require_pyspark()
    F = spark.functions
    bins = max(int(bins), 1)
    if bins == 1:
        return df.withColumn("bin", F.lit(1))

    probabilities = [index / bins for index in range(bins + 1)]
    edges = [
        float(value)
        for value in df.approxQuantile(column, probabilities, 0.01)
        if value is not None and math.isfinite(float(value))
    ]
    if len(edges) < 2:
        return df.withColumn("bin", F.lit(1))

    minimum = min(edges)
    maximum = max(edges)
    thresholds = sorted({edge for edge in edges[1:-1] if minimum < edge < maximum})
    bin_expr = F.when(F.col(column).isNull(), F.lit(1))
    for index, threshold in enumerate(thresholds, start=1):
        bin_expr = bin_expr.when(F.col(column) <= F.lit(threshold), F.lit(index))
    return df.withColumn("bin", bin_expr.otherwise(F.lit(len(thresholds) + 1)))
