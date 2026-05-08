"""Metrics and validation tables for GLMs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _array(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def poisson_deviance(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    weight: pd.Series | np.ndarray | list[float] | None = None,
) -> float:
    """Return mean Poisson deviance.

    Parameters
    ----------
    actual:
        Observed nonnegative outcome values.
    predicted:
        Predicted mean values.
    weight:
        Optional observation weights.

    Returns
    -------
    float
        Weighted or unweighted mean Poisson deviance.
    """

    y = np.clip(_array(actual), a_min=0.0, a_max=None)
    mu = np.clip(_array(predicted), a_min=1e-9, a_max=None)
    positive = y > 0
    ratio = np.ones_like(y)
    ratio[positive] = y[positive] / mu[positive]
    values = 2.0 * np.where(positive, y * np.log(ratio) - (y - mu), mu)
    return _weighted_mean(values, weight)


def gamma_deviance(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    weight: pd.Series | np.ndarray | list[float] | None = None,
) -> float:
    """Return mean Gamma deviance for positive continuous targets.

    Parameters
    ----------
    actual:
        Observed positive outcome values.
    predicted:
        Predicted positive mean values.
    weight:
        Optional observation weights.

    Returns
    -------
    float
        Weighted or unweighted mean Gamma deviance.
    """

    y = np.clip(_array(actual), a_min=1e-9, a_max=None)
    mu = np.clip(_array(predicted), a_min=1e-9, a_max=None)
    values = 2.0 * ((y - mu) / mu - np.log(y / mu))
    return _weighted_mean(values, weight)


def gaussian_deviance(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    weight: pd.Series | np.ndarray | list[float] | None = None,
) -> float:
    """Return mean Gaussian deviance, equivalent to mean squared error.

    Parameters
    ----------
    actual:
        Observed outcome values.
    predicted:
        Predicted mean values.
    weight:
        Optional observation weights.

    Returns
    -------
    float
        Weighted or unweighted mean squared error.
    """

    values = np.square(_array(actual) - _array(predicted))
    return _weighted_mean(values, weight)


def weighted_mae(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    weight: pd.Series | np.ndarray | list[float] | None = None,
) -> float:
    """Return weighted mean absolute error.

    Parameters
    ----------
    actual:
        Observed outcome values.
    predicted:
        Predicted mean values.
    weight:
        Optional observation weights.

    Returns
    -------
    float
        Weighted or unweighted mean absolute error.
    """

    values = np.abs(_array(actual) - _array(predicted))
    return _weighted_mean(values, weight)


def weighted_rmse(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    weight: pd.Series | np.ndarray | list[float] | None = None,
) -> float:
    """Return weighted root mean squared error.

    Parameters
    ----------
    actual:
        Observed outcome values.
    predicted:
        Predicted mean values.
    weight:
        Optional observation weights.

    Returns
    -------
    float
        Weighted or unweighted root mean squared error.
    """

    values = np.square(_array(actual) - _array(predicted))
    return float(np.sqrt(_weighted_mean(values, weight)))


def model_deviance(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    *,
    family: str,
    weight: pd.Series | np.ndarray | list[float] | None = None,
) -> float:
    """Return family-specific mean deviance.

    Parameters
    ----------
    actual:
        Observed outcome values.
    predicted:
        Predicted mean values.
    family:
        GLM family name, such as ``"poisson"``, ``"gamma"``, or
        ``"gaussian"``.
    weight:
        Optional observation weights.

    Returns
    -------
    float
        Weighted or unweighted mean deviance for the requested family.
    """

    normalized = family.lower().strip()
    if normalized == "poisson":
        return poisson_deviance(actual, predicted, weight)
    if normalized == "gamma":
        return gamma_deviance(actual, predicted, weight)
    if normalized == "gaussian":
        return gaussian_deviance(actual, predicted, weight)
    raise ValueError("family must be 'poisson', 'gamma', or 'gaussian'.")


def _weighted_mean(values: np.ndarray, weight: pd.Series | np.ndarray | list[float] | None) -> float:
    if weight is None:
        return float(np.mean(values))
    return float(np.average(values, weights=_array(weight)))


def summary(
    df: pd.DataFrame,
    target: str,
    prediction: str,
    exposure: str | None = None,
    weight: str | None = None,
    family: str = "poisson",
) -> pd.DataFrame:
    """Return one-row actual-vs-predicted summary.

    Parameters
    ----------
    df:
        Scored data.
    target:
        Observed outcome column.
    prediction:
        Predicted outcome column.
    exposure:
        Optional exposure column for rate-style summary columns.
    weight:
        Optional row-weight column for deviance scoring.
    family:
        GLM family used for deviance scoring.

    Returns
    -------
    pandas.DataFrame
        One-row summary with totals, ratio, deviance, and mean or exposure
        columns.
    """

    actual = float(df[target].sum())
    predicted = float(df[prediction].sum())
    sample_weight = df[weight] if weight else None
    row = {
        "rows": int(len(df)),
        "actual": actual,
        "predicted": predicted,
        "actual_to_predicted": actual / predicted if predicted else np.nan,
        "deviance": model_deviance(df[target], df[prediction], family=family, weight=sample_weight),
    }
    if exposure is not None:
        total_exposure = float(df[exposure].sum())
        row["exposure"] = total_exposure
        row["actual_rate"] = actual / total_exposure if total_exposure else np.nan
        row["predicted_rate"] = predicted / total_exposure if total_exposure else np.nan
    else:
        row["actual_mean"] = float(df[target].mean())
        row["predicted_mean"] = float(df[prediction].mean())
    return pd.DataFrame([row])


def calibration(
    df: pd.DataFrame,
    target: str,
    prediction: str,
    exposure: str | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Group rows by predicted value or exposure-adjusted prediction quantile.

    Parameters
    ----------
    df:
        Scored data.
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
    pandas.DataFrame
        Calibration table by prediction band.
    """

    columns = [target, prediction] + ([exposure] if exposure is not None else [])
    work = df[columns].copy()
    if exposure is not None:
        rank_value = work[prediction] / work[exposure].clip(lower=1e-9)
    else:
        rank_value = work[prediction]
    rank = rank_value.rank(method="first")
    work["bin"] = pd.qcut(rank, q=min(bins, len(work)), labels=False, duplicates="drop")
    work["bin"] = work["bin"].astype(int) + 1
    agg_spec = {"actual": (target, "sum"), "predicted": (prediction, "sum")}
    if exposure is not None:
        agg_spec["exposure"] = (exposure, "sum")
    table = work.groupby("bin", dropna=False).agg(**agg_spec).reset_index()
    if exposure is not None:
        table["actual_rate"] = table["actual"] / table["exposure"].clip(lower=1e-9)
        table["predicted_rate"] = table["predicted"] / table["exposure"].clip(lower=1e-9)
    else:
        counts = work.groupby("bin", dropna=False).size().to_numpy()
        table["actual_mean"] = table["actual"] / counts
        table["predicted_mean"] = table["predicted"] / counts
    table["actual_to_predicted"] = table["actual"] / table["predicted"].clip(lower=1e-9)
    return table


def lift_table(
    df: pd.DataFrame,
    target: str,
    prediction: str,
    exposure: str | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Return calibration table with lift versus the overall observed level.

    Parameters
    ----------
    df:
        Scored data.
    target:
        Observed outcome column.
    prediction:
        Predicted outcome column.
    exposure:
        Optional exposure column used for rate-style lift.
    bins:
        Number of quantile bands to create.

    Returns
    -------
    pandas.DataFrame
        Calibration table with an added ``lift`` column.
    """

    table = calibration(df, target, prediction, exposure, bins=bins)
    if exposure is not None:
        overall = float(df[target].sum() / df[exposure].sum())
        table["lift"] = table["actual_rate"] / max(overall, 1e-9)
    else:
        overall = float(df[target].mean())
        table["lift"] = table["actual_mean"] / max(overall, 1e-9)
    return table
