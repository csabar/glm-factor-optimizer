"""Validation reports for iterative GLM model design."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import calibration, lift_table, model_deviance, summary, weighted_mae, weighted_rmse


def scored_summary(
    df: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """Return one-row validation metrics for scored data."""

    table = summary(
        df,
        target=target,
        prediction=prediction,
        exposure=exposure,
        weight=weight,
        family=family,
    )
    sample_weight = df[weight] if weight else None
    table["mae"] = weighted_mae(df[target], df[prediction], sample_weight)
    table["rmse"] = weighted_rmse(df[target], df[prediction], sample_weight)
    if label is not None:
        table.insert(0, "sample", label)
    return table


def by_factor_report(
    df: pd.DataFrame,
    *,
    factor: str,
    target: str,
    prediction: str,
    exposure: str | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Return actual-vs-predicted diagnostics by one factor."""

    if exposure is not None:
        table = (
            df.groupby(factor, dropna=False)
            .agg(
                rows=(target, "size"),
                bin_size=(exposure, "sum"),
                actual=(target, "sum"),
                predicted=(prediction, "sum"),
            )
            .reset_index()
        )
        table["actual_rate"] = table["actual"] / table["bin_size"].clip(lower=1e-9)
        table["predicted_rate"] = table["predicted"] / table["bin_size"].clip(lower=1e-9)
    elif weight is not None:
        work = df.copy()
        work["_weighted_actual"] = work[target] * work[weight]
        work["_weighted_predicted"] = work[prediction] * work[weight]
        table = (
            work.groupby(factor, dropna=False)
            .agg(
                rows=(target, "size"),
                bin_size=(weight, "sum"),
                actual=("_weighted_actual", "sum"),
                predicted=("_weighted_predicted", "sum"),
            )
            .reset_index()
        )
        table["actual_mean"] = table["actual"] / table["bin_size"].clip(lower=1e-9)
        table["predicted_mean"] = table["predicted"] / table["bin_size"].clip(lower=1e-9)
    else:
        table = (
            df.groupby(factor, dropna=False)
            .agg(
                rows=(target, "size"),
                actual=(target, "sum"),
                predicted=(prediction, "sum"),
            )
            .reset_index()
        )
        table["bin_size"] = table["rows"].astype(float)
        table["actual_mean"] = table["actual"] / table["rows"].clip(lower=1)
        table["predicted_mean"] = table["predicted"] / table["rows"].clip(lower=1)

    table["actual_to_predicted"] = table["actual"] / table["predicted"].clip(lower=1e-9)
    return table.sort_values(factor).reset_index(drop=True)


def train_validation_comparison(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Return train-vs-validation scored summaries."""

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
    validation_df: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    factors: Sequence[str] | None = None,
    bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Return a rich validation report for scored data."""

    report = {
        "summary": scored_summary(
            validation_df,
            target=target,
            prediction=prediction,
            family=family,
            exposure=exposure,
            weight=weight,
        ),
        "calibration": calibration(
            validation_df,
            target=target,
            prediction=prediction,
            exposure=exposure,
            bins=bins,
        ),
        "lift": lift_table(
            validation_df,
            target=target,
            prediction=prediction,
            exposure=exposure,
            bins=bins,
        ),
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
    holdout_df: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    factors: Sequence[str] | None = None,
    bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Return final holdout reports for a scored holdout dataset."""

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


def model_version_comparison(versions: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Return a compact dataframe from stored model-version metadata."""

    rows = []
    for version in versions:
        rows.append(
            {
                "version": version.get("version"),
                "factors": ",".join(version.get("factors", [])),
                "train_deviance": version.get("train_deviance", np.nan),
                "validation_deviance": version.get("validation_deviance", np.nan),
                "validation_mae": version.get("validation_mae", np.nan),
                "validation_rmse": version.get("validation_rmse", np.nan),
            }
        )
    return pd.DataFrame(rows)


def deviance_score(
    df: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    family: str,
    weight: str | None = None,
) -> float:
    """Return model deviance for an already scored dataframe."""

    return model_deviance(
        df[target],
        df[prediction],
        family=family,
        weight=df[weight] if weight else None,
    )

