"""Spark-backed validation reports collected as small notebook tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ._deps import require_pyspark
from .metrics import calibration, model_deviance, summary_frame


def scored_summary(
    df: Any,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """Return a one-row Spark validation summary as a pandas table."""

    table = summary_frame(
        df,
        target=target,
        prediction=prediction,
        exposure=exposure,
        weight=weight,
        family=family,
        include_errors=True,
    ).toPandas()
    if label is not None:
        table.insert(0, "sample", label)
    return table


def by_factor_report(
    df: Any,
    *,
    factor: str,
    target: str,
    prediction: str,
    exposure: str | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Return observed-vs-predicted diagnostics by one Spark factor."""

    spark = require_pyspark()
    F = spark.functions
    group_col = F.coalesce(F.col(factor).cast("string"), F.lit("missing")).alias(factor)
    work = df.select(
        group_col,
        F.col(target).cast("double").alias("__target"),
        F.col(prediction).cast("double").alias("__prediction"),
        *([F.col(exposure).cast("double").alias("__exposure")] if exposure is not None else []),
        *([F.col(weight).cast("double").alias("__weight")] if weight is not None else []),
    )

    if exposure is not None:
        table = work.groupBy(factor).agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum("__exposure").alias("bin_size"),
            F.sum("__target").alias("actual"),
            F.sum("__prediction").alias("predicted"),
        )
        table = table.withColumn("actual_rate", F.col("actual") / F.greatest(F.col("bin_size"), F.lit(1e-9)))
        table = table.withColumn("predicted_rate", F.col("predicted") / F.greatest(F.col("bin_size"), F.lit(1e-9)))
    elif weight is not None:
        table = work.groupBy(factor).agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum("__weight").alias("bin_size"),
            F.sum(F.col("__target") * F.col("__weight")).alias("actual"),
            F.sum(F.col("__prediction") * F.col("__weight")).alias("predicted"),
        )
        table = table.withColumn("actual_mean", F.col("actual") / F.greatest(F.col("bin_size"), F.lit(1e-9)))
        table = table.withColumn("predicted_mean", F.col("predicted") / F.greatest(F.col("bin_size"), F.lit(1e-9)))
    else:
        table = work.groupBy(factor).agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum("__target").alias("actual"),
            F.sum("__prediction").alias("predicted"),
        )
        table = table.withColumn("bin_size", F.col("rows").cast("double"))
        table = table.withColumn("actual_mean", F.col("actual") / F.greatest(F.col("rows"), F.lit(1)))
        table = table.withColumn("predicted_mean", F.col("predicted") / F.greatest(F.col("rows"), F.lit(1)))

    table = table.withColumn("actual_to_predicted", F.col("actual") / F.greatest(F.col("predicted"), F.lit(1e-9)))
    result = table.orderBy(factor).toPandas()
    return result.reset_index(drop=True)


def train_validation_comparison(
    train_df: Any,
    validation_df: Any,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Return train-vs-validation scored summaries for Spark frames."""

    train = scored_summary(
        train_df,
        target=target,
        prediction=prediction,
        family=family,
        exposure=exposure,
        weight=weight,
        label="train",
    )
    validation = scored_summary(
        validation_df,
        target=target,
        prediction=prediction,
        family=family,
        exposure=exposure,
        weight=weight,
        label="validation",
    )
    return pd.concat([train, validation], ignore_index=True)


def validation_report(
    validation_df: Any,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    factors: Sequence[str] | None = None,
    bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Return Spark validation reports as bounded pandas metadata tables."""

    calibration_table = calibration(
        validation_df,
        target=target,
        prediction=prediction,
        exposure=exposure,
        bins=bins,
    ).toPandas()
    report = {
        "summary": scored_summary(
            validation_df,
            target=target,
            prediction=prediction,
            family=family,
            exposure=exposure,
            weight=weight,
        ),
        "calibration": calibration_table,
        "lift": _lift_from_calibration(calibration_table, exposure=exposure),
    }
    for factor in factors or []:
        report[f"by_{factor}"] = by_factor_report(
            validation_df,
            factor=factor,
            target=target,
            prediction=prediction,
            exposure=exposure,
            weight=weight,
        )
    return report


def holdout_final_report(
    holdout_df: Any,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    factors: Sequence[str] | None = None,
    bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Return final holdout reports for a scored Spark dataset."""

    return validation_report(
        holdout_df,
        target=target,
        prediction=prediction,
        family=family,
        exposure=exposure,
        weight=weight,
        factors=factors,
        bins=bins,
    )


def deviance_score(
    df: Any,
    *,
    target: str,
    prediction: str,
    family: str,
    weight: str | None = None,
) -> float:
    """Return model deviance for an already scored Spark dataframe."""

    return model_deviance(df, target, prediction, family=family, weight=weight)


def scored_metrics(
    df: Any,
    *,
    target: str,
    prediction: str,
    family: str,
    weight: str | None = None,
) -> dict[str, float]:
    """Return scalar deviance, MAE, and RMSE for scored Spark data."""

    row = summary_frame(
        df,
        target=target,
        prediction=prediction,
        family=family,
        weight=weight,
        include_errors=True,
    ).first()
    return {
        "deviance": float(row["deviance"]),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
    }


def _lift_from_calibration(
    calibration_table: pd.DataFrame,
    *,
    exposure: str | None,
) -> pd.DataFrame:
    table = calibration_table.copy()
    if exposure is not None:
        overall = float(table["actual"].sum()) / max(float(table["exposure"].sum()), 1e-9)
        table["lift"] = table["actual_rate"] / max(overall, 1e-9)
        return table
    overall = float(table["actual"].sum()) / max(float(table["rows"].sum()), 1e-9)
    table["lift"] = table["actual_mean"] / max(overall, 1e-9)
    return table
