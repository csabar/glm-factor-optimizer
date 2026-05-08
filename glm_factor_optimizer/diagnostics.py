"""Diagnostics for spotting candidate factor interactions."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd


def pair_diagnostics(
    df: pd.DataFrame,
    *,
    factors: Sequence[str],
    target: str,
    prediction: str,
    exposure: str | None = None,
    weight: str | None = None,
    min_bin_size: float = 1.0,
) -> pd.DataFrame:
    """Score observed-vs-predicted deviations for all factor pairs.

    Parameters
    ----------
    df:
        Scored data containing factor, target, and prediction columns.
    factors:
        Transformed factor columns to pair.
    target:
        Observed outcome column.
    prediction:
        Predicted outcome column.
    exposure:
        Optional exposure column used as cell size.
    weight:
        Optional weight column used as cell size when exposure is absent.
    min_bin_size:
        Minimum cell size required for a pair cell to contribute.

    Returns
    -------
    pandas.DataFrame
        Ranked pair table with weighted deviation diagnostics.
    """

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
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    factors: Sequence[str],
    target: str,
    prediction: str,
    exposure: str | None = None,
    weight: str | None = None,
    min_bin_size: float = 1.0,
) -> pd.DataFrame:
    """Return pair interaction candidates that appear in train and validation.

    Parameters
    ----------
    train_df:
        Scored training data.
    validation_df:
        Scored validation data.
    factors:
        Transformed factor columns to pair.
    target:
        Observed outcome column.
    prediction:
        Predicted outcome column.
    exposure:
        Optional exposure column used as cell size.
    weight:
        Optional weight column used as cell size when exposure is absent.
    min_bin_size:
        Minimum cell size required for a pair cell to contribute.

    Returns
    -------
    pandas.DataFrame
        Ranked interaction-candidate table combining train and validation
        diagnostics.
    """

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
    df: pd.DataFrame,
    *,
    factor_a: str,
    factor_b: str,
    target: str,
    prediction: str,
    exposure: str | None,
    weight: str | None,
) -> pd.DataFrame:
    group_cols = [factor_a, factor_b]
    if exposure is not None:
        table = (
            df.groupby(group_cols, dropna=False)
            .agg(
                bin_size=(exposure, "sum"),
                actual=(target, "sum"),
                predicted=(prediction, "sum"),
                rows=(target, "size"),
            )
            .reset_index()
        )
    elif weight is not None:
        work = df.copy()
        work["_weighted_actual"] = work[target] * work[weight]
        work["_weighted_predicted"] = work[prediction] * work[weight]
        table = (
            work.groupby(group_cols, dropna=False)
            .agg(
                bin_size=(weight, "sum"),
                actual=("_weighted_actual", "sum"),
                predicted=("_weighted_predicted", "sum"),
                rows=(target, "size"),
            )
            .reset_index()
        )
    else:
        table = (
            df.groupby(group_cols, dropna=False)
            .agg(
                bin_size=(target, "size"),
                actual=(target, "sum"),
                predicted=(prediction, "sum"),
                rows=(target, "size"),
            )
            .reset_index()
        )
    table["actual_to_predicted"] = table["actual"] / table["predicted"].clip(lower=1e-9)
    return table
