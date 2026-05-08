"""Candidate-factor screening before detailed Optuna optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .bins import apply_spec, category_target_order, make_categorical_groups, make_numeric_bins
from .metrics import model_deviance
from .model import fit_glm

JsonDict = dict[str, Any]

try:
    from scipy.stats import chi2
except ModuleNotFoundError:  # pragma: no cover - statsmodels normally brings scipy.
    chi2 = None


def rank_factors(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    target: str,
    factors: Sequence[str],
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    factor_kinds: Mapping[str, str] | None = None,
    bins: int = 5,
    max_groups: int = 6,
    prediction: str = "predicted",
    min_bin_size: float = 1.0,
) -> pd.DataFrame:
    """Rank candidate factors by validation deviance improvement.

    Parameters
    ----------
    train_df:
        Training data used to derive simple factor specs and fit screening
        models.
    validation_df:
        Validation data used to compare candidate factors.
    target:
        Observed outcome column.
    factors:
        Raw candidate factor columns to screen.
    family:
        GLM family name, such as ``"poisson"``, ``"gamma"``, or
        ``"gaussian"``.
    exposure:
        Optional exposure column used as a log offset.
    weight:
        Optional row-weight column.
    factor_kinds:
        Optional mapping from factor name to ``"numeric"`` or
        ``"categorical"``.
    bins:
        Number of numeric bins used by the simple screening spec.
    max_groups:
        Maximum number of ordered categorical groups used by the simple
        screening spec.
    prediction:
        Prediction column name used internally by screening models.
    min_bin_size:
        Threshold used to count small bins in screening diagnostics.

    Returns
    -------
    pandas.DataFrame
        Ranked table with deviance improvement, significance-style statistics,
        missingness, coverage, bin-size diagnostics, specs, and errors.

    Notes
    -----
    The screening pass intentionally uses simple train-derived specs. It is a
    quick ranking tool; final binning should still be optimized factor by
    factor with ``optimize_factor`` or ``GLMWorkflow``.
    """

    required = [target, *factors]
    if exposure is not None:
        required.append(exposure)
    if weight is not None:
        required.append(weight)
    for column in required:
        if column not in train_df.columns or column not in validation_df.columns:
            raise KeyError(f"Column {column!r} must exist in train and validation data.")

    baseline = fit_glm(
        train_df,
        target=target,
        factors=[],
        family=family,
        exposure=exposure,
        weight=weight,
        prediction=prediction,
    )
    baseline_train_prediction = baseline.predict(train_df)
    baseline_validation_prediction = baseline.predict(validation_df)
    baseline_train_deviance = model_deviance(
        train_df[target],
        baseline_train_prediction,
        family=family,
        weight=train_df[weight] if weight else None,
    )
    baseline_validation_deviance = model_deviance(
        validation_df[target],
        baseline_validation_prediction,
        family=family,
        weight=validation_df[weight] if weight else None,
    )

    rows: list[dict[str, Any]] = []
    for factor in factors:
        kind = (factor_kinds or {}).get(factor) or _infer_kind(train_df[factor])
        try:
            spec = _screening_spec(
                train_df,
                factor=factor,
                kind=kind,
                target=target,
                exposure=exposure,
                weight=weight,
                bins=bins,
                max_groups=max_groups,
            )
            train_transformed = apply_spec(train_df, spec)
            validation_transformed = apply_spec(validation_df, spec)
            model = fit_glm(
                train_transformed,
                target=target,
                factors=[str(spec["output"])],
                family=family,
                exposure=exposure,
                weight=weight,
                prediction=prediction,
            )
            train_prediction = model.predict(train_transformed)
            validation_prediction = model.predict(validation_transformed)
            train_deviance = model_deviance(
                train_transformed[target],
                train_prediction,
                family=family,
                weight=train_transformed[weight] if weight else None,
            )
            validation_deviance = model_deviance(
                validation_transformed[target],
                validation_prediction,
                family=family,
                weight=validation_transformed[weight] if weight else None,
            )
            bin_table = _bin_table(
                train_transformed,
                output=str(spec["output"]),
                target=target,
                exposure=exposure,
                weight=weight,
                labels=list(spec["labels"]),
            )
            improvement = float(baseline_validation_deviance - validation_deviance)
            effective_n = _effective_n(validation_transformed, weight)
            degrees_of_freedom = max(int(len(bin_table) - 1), 1)
            likelihood_ratio_stat = max(improvement * effective_n, 0.0)
            rows.append(
                {
                    "factor": factor,
                    "kind": kind,
                    "output": str(spec["output"]),
                    "baseline_train_deviance": float(baseline_train_deviance),
                    "baseline_validation_deviance": float(baseline_validation_deviance),
                    "train_deviance": float(train_deviance),
                    "validation_deviance": float(validation_deviance),
                    "deviance_improvement": improvement,
                    "relative_improvement": improvement / max(float(baseline_validation_deviance), 1e-9),
                    "likelihood_ratio_stat": float(likelihood_ratio_stat),
                    "degrees_of_freedom": degrees_of_freedom,
                    "p_value": _chi_square_p_value(likelihood_ratio_stat, degrees_of_freedom),
                    "train_missing_rate": _missing_rate(train_df[factor]),
                    "validation_missing_rate": _missing_rate(validation_df[factor]),
                    "train_measure_coverage": _measure_coverage(train_df, factor, exposure, weight),
                    "validation_measure_coverage": _measure_coverage(validation_df, factor, exposure, weight),
                    "bins": int(len(bin_table)),
                    "min_bin_size": float(bin_table["bin_size"].min()),
                    "small_bins": int((bin_table["bin_size"] < min_bin_size).sum()),
                    "spec": spec,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "factor": factor,
                    "kind": kind,
                    "output": None,
                    "baseline_train_deviance": float(baseline_train_deviance),
                    "baseline_validation_deviance": float(baseline_validation_deviance),
                    "train_deviance": np.nan,
                    "validation_deviance": np.inf,
                    "deviance_improvement": -np.inf,
                    "relative_improvement": -np.inf,
                    "likelihood_ratio_stat": 0.0,
                    "degrees_of_freedom": 0,
                    "p_value": np.nan,
                    "train_missing_rate": _missing_rate(train_df[factor]) if factor in train_df else np.nan,
                    "validation_missing_rate": _missing_rate(validation_df[factor])
                    if factor in validation_df
                    else np.nan,
                    "train_measure_coverage": _measure_coverage(train_df, factor, exposure, weight)
                    if factor in train_df
                    else np.nan,
                    "validation_measure_coverage": _measure_coverage(validation_df, factor, exposure, weight)
                    if factor in validation_df
                    else np.nan,
                    "bins": 0,
                    "min_bin_size": 0.0,
                    "small_bins": 0,
                    "spec": None,
                    "error": repr(exc),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["deviance_improvement", "factor"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _screening_spec(
    train_df: pd.DataFrame,
    *,
    factor: str,
    kind: str,
    target: str,
    exposure: str | None,
    weight: str | None,
    bins: int,
    max_groups: int,
) -> JsonDict:
    kind = kind.lower().strip()
    if kind == "numeric":
        return make_numeric_bins(train_df[factor], bins=bins, column=factor)
    if kind == "categorical":
        target_order = category_target_order(train_df, factor, target, exposure=exposure, weight=weight)
        category_count = len(target_order)
        group_count = max(1, min(max_groups, category_count))
        cutpoints = _even_cutpoints(category_count, group_count)
        return make_categorical_groups(
            train_df,
            factor,
            target,
            exposure=exposure,
            weight=weight,
            cutpoints=cutpoints,
        )
    raise ValueError("kind must be 'numeric' or 'categorical'.")


def _even_cutpoints(category_count: int, group_count: int) -> list[int]:
    if category_count <= 1 or group_count <= 1:
        return []
    raw = np.linspace(0, category_count, group_count + 1)[1:-1]
    return sorted({int(round(value)) for value in raw if 0 < int(round(value)) < category_count})


def _bin_table(
    df: pd.DataFrame,
    *,
    output: str,
    target: str,
    exposure: str | None,
    weight: str | None,
    labels: list[str],
) -> pd.DataFrame:
    observed = list(labels)
    for extra in ["missing", "other"]:
        if (df[output] == extra).any() and extra not in observed:
            observed.append(extra)
    if exposure is not None:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(exposure, "sum"), actual=(target, "sum"), rows=(target, "size"))
            .reindex(observed, fill_value=0.0)
        )
    elif weight is not None:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(weight, "sum"), actual=(target, "sum"), rows=(target, "size"))
            .reindex(observed, fill_value=0.0)
        )
    else:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(target, "size"), actual=(target, "sum"), rows=(target, "size"))
            .reindex(observed, fill_value=0.0)
        )
    return table.reset_index().rename(columns={output: "bin"})


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.isna().mean())


def _measure_coverage(
    df: pd.DataFrame,
    factor: str,
    exposure: str | None,
    weight: str | None,
) -> float:
    nonmissing = df[factor].notna()
    if exposure is not None:
        total = float(df[exposure].sum())
        return float(df.loc[nonmissing, exposure].sum() / total) if total else np.nan
    if weight is not None:
        total = float(df[weight].sum())
        return float(df.loc[nonmissing, weight].sum() / total) if total else np.nan
    return float(nonmissing.mean()) if len(df) else np.nan


def _effective_n(df: pd.DataFrame, weight: str | None) -> float:
    if weight is not None:
        return float(df[weight].sum())
    return float(len(df))


def _chi_square_p_value(statistic: float, degrees_of_freedom: int) -> float:
    if chi2 is None or degrees_of_freedom <= 0:
        return np.nan
    return float(chi2.sf(statistic, degrees_of_freedom))
