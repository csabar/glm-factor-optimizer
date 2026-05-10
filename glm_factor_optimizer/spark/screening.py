"""Spark candidate-factor screening with bounded metadata output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._deps import require_pyspark
from .bins import apply_spec, categorical_group_spec_from_order, category_target_order, make_numeric_bins
from .metrics import model_deviance
from .model import fit_glm

JsonDict = dict[str, Any]

try:
    from scipy.stats import chi2
except ModuleNotFoundError:  # pragma: no cover - statsmodels normally brings scipy.
    chi2 = None


def rank_factors(
    train_df: Any,
    validation_df: Any,
    *,
    target: str,
    factors: Sequence[str],
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    factor_kinds: Mapping[str, str] | None = None,
    bins: int = 5,
    max_groups: int = 6,
    prediction: str = "prediction",
    min_bin_size: float = 1.0,
) -> pd.DataFrame:
    """Rank Spark candidate factors and return a small pandas table."""

    required = [target, *factors]
    if exposure is not None:
        required.append(exposure)
    if weight is not None:
        required.append(weight)
    _check_columns(train_df, validation_df, required)
    train_metadata = _factor_metadata(train_df, factors, exposure, weight)
    validation_metadata = _factor_metadata(validation_df, factors, exposure, weight)
    effective_n = _effective_n(validation_df, weight)

    baseline = fit_glm(
        train_df,
        target=target,
        factors=[],
        family=family,
        exposure=exposure,
        weight=weight,
        prediction=prediction,
    )
    baseline_train_scored = baseline.transform(train_df)
    baseline_validation_scored = baseline.transform(validation_df)
    baseline_train_deviance = model_deviance(
        baseline_train_scored,
        target,
        prediction,
        family=family,
        weight=weight,
    )
    baseline_validation_deviance = model_deviance(
        baseline_validation_scored,
        target,
        prediction,
        family=family,
        weight=weight,
    )
    baseline.release()

    rows: list[dict[str, Any]] = []
    for factor in factors:
        kind = (factor_kinds or {}).get(factor) or infer_kind(train_df, factor)
        model = None
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
            train_scored = model.transform(train_transformed)
            validation_scored = model.transform(validation_transformed)
            train_deviance = model_deviance(
                train_scored,
                target,
                prediction,
                family=family,
                weight=weight,
            )
            validation_deviance = model_deviance(
                validation_scored,
                target,
                prediction,
                family=family,
                weight=weight,
            )
            bin_table = bin_table_for_spec(
                train_transformed,
                output=str(spec["output"]),
                target=target,
                exposure=exposure,
                weight=weight,
                labels=list(spec["labels"]),
            )
            improvement = float(baseline_validation_deviance - validation_deviance)
            degrees_of_freedom = max(int(len(bin_table) - 1), 1)
            likelihood_ratio_stat = max(improvement * effective_n, 0.0)
            train_factor_metadata = train_metadata[factor]
            validation_factor_metadata = validation_metadata[factor]
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
                    "train_missing_rate": train_factor_metadata["missing_rate"],
                    "validation_missing_rate": validation_factor_metadata["missing_rate"],
                    "train_measure_coverage": train_factor_metadata["measure_coverage"],
                    "validation_measure_coverage": validation_factor_metadata["measure_coverage"],
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
                    "train_missing_rate": train_metadata.get(factor, {}).get("missing_rate", np.nan),
                    "validation_missing_rate": validation_metadata.get(factor, {}).get("missing_rate", np.nan),
                    "train_measure_coverage": train_metadata.get(factor, {}).get("measure_coverage", np.nan),
                    "validation_measure_coverage": validation_metadata.get(factor, {}).get("measure_coverage", np.nan),
                    "bins": 0,
                    "min_bin_size": 0.0,
                    "small_bins": 0,
                    "spec": None,
                    "error": repr(exc),
                }
            )
        finally:
            if model is not None:
                model.release()

    return (
        pd.DataFrame(rows)
        .sort_values(["deviance_improvement", "factor"], ascending=[False, True])
        .reset_index(drop=True)
    )


def infer_kind(df: Any, factor: str) -> str:
    """Infer numeric vs categorical kind from a Spark schema."""

    simple_type = df.schema[factor].dataType.simpleString().split("(")[0]
    if simple_type in {"byte", "short", "integer", "long", "float", "double", "decimal"}:
        return "numeric"
    return "categorical"


def bin_table_for_spec(
    df: Any,
    *,
    output: str,
    target: str,
    exposure: str | None,
    weight: str | None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return a bounded pandas bin table from a transformed Spark frame."""

    spark = require_pyspark()
    F = spark.functions
    if exposure is not None:
        aggs = [F.sum(F.col(exposure).cast("double")).alias("bin_size")]
    elif weight is not None:
        aggs = [F.sum(F.col(weight).cast("double")).alias("bin_size")]
    else:
        aggs = [F.count(F.lit(1)).cast("double").alias("bin_size")]
    aggs.extend(
        [
            F.sum(F.col(target).cast("double")).alias("actual"),
            F.count(F.lit(1)).alias("rows"),
        ]
    )
    table = df.groupBy(output).agg(*aggs).toPandas().rename(columns={output: "bin"})
    if labels is None:
        return table.sort_values("bin").reset_index(drop=True)
    observed = pd.DataFrame({"bin": list(labels)})
    for extra in ["missing", "other"]:
        if extra in set(table["bin"].astype(str)) and extra not in observed["bin"].tolist():
            observed = pd.concat([observed, pd.DataFrame({"bin": [extra]})], ignore_index=True)
    return observed.merge(table, on="bin", how="left").fillna({"bin_size": 0.0, "actual": 0.0, "rows": 0})


def _screening_spec(
    train_df: Any,
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
        return make_numeric_bins(train_df, factor, bins=bins)
    if kind == "categorical":
        target_order = category_target_order(train_df, factor, target, exposure=exposure, weight=weight)
        category_count = len(target_order)
        group_count = max(1, min(max_groups, category_count))
        cutpoints = even_cutpoints(category_count, group_count)
        return categorical_group_spec_from_order(factor, target_order, cutpoints=cutpoints)
    raise ValueError("kind must be 'numeric' or 'categorical'.")


def even_cutpoints(category_count: int, group_count: int) -> list[int]:
    """Return approximately even ordered-category cutpoints."""

    if category_count <= 1 or group_count <= 1:
        return []
    raw = np.linspace(0, category_count, group_count + 1)[1:-1]
    return sorted({int(round(value)) for value in raw if 0 < int(round(value)) < category_count})


def _missing_rate(df: Any, factor: str) -> float:
    spark = require_pyspark()
    F = spark.functions
    row = df.select(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col(factor).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("missing"),
    ).first()
    rows = int(row["rows"])
    return float(row["missing"] / rows) if rows else np.nan


def _factor_metadata(
    df: Any,
    factors: Sequence[str],
    exposure: str | None,
    weight: str | None,
) -> dict[str, dict[str, float]]:
    spark = require_pyspark()
    F = spark.functions
    if exposure is not None:
        measure = F.col(exposure).cast("double")
    elif weight is not None:
        measure = F.col(weight).cast("double")
    else:
        measure = F.lit(1.0)

    aggs = [
        F.count(F.lit(1)).alias("__rows"),
        F.sum(measure).alias("__measure_total"),
    ]
    for index, factor in enumerate(factors):
        nonmissing = F.col(factor).isNotNull()
        aggs.extend([
            F.sum(F.when(nonmissing, F.lit(0)).otherwise(F.lit(1))).alias(f"__missing_{index}"),
            F.sum(F.when(nonmissing, measure).otherwise(F.lit(0.0))).alias(f"__covered_{index}"),
        ])
    row = df.select(*aggs).first()
    rows = float(row["__rows"] or 0.0)
    total = float(row["__measure_total"] or 0.0)
    metadata: dict[str, dict[str, float]] = {}
    for index, factor in enumerate(factors):
        missing = float(row[f"__missing_{index}"] or 0.0)
        covered = float(row[f"__covered_{index}"] or 0.0)
        metadata[factor] = {
            "missing_rate": missing / rows if rows else np.nan,
            "measure_coverage": covered / total if total else np.nan,
        }
    return metadata


def _measure_coverage(
    df: Any,
    factor: str,
    exposure: str | None,
    weight: str | None,
) -> float:
    spark = require_pyspark()
    F = spark.functions
    nonmissing = F.col(factor).isNotNull()
    if exposure is not None:
        measure = F.col(exposure).cast("double")
    elif weight is not None:
        measure = F.col(weight).cast("double")
    else:
        measure = F.lit(1.0)
    row = df.select(
        F.sum(measure).alias("total"),
        F.sum(F.when(nonmissing, measure).otherwise(F.lit(0.0))).alias("covered"),
    ).first()
    total = float(row["total"] or 0.0)
    return float(row["covered"] / total) if total else np.nan


def _effective_n(df: Any, weight: str | None) -> float:
    spark = require_pyspark()
    F = spark.functions
    if weight is not None:
        row = df.select(F.sum(F.col(weight).cast("double")).alias("n")).first()
    else:
        row = df.select(F.count(F.lit(1)).cast("double").alias("n")).first()
    return float(row["n"])


def _chi_square_p_value(statistic: float, degrees_of_freedom: int) -> float:
    if chi2 is None or degrees_of_freedom <= 0:
        return np.nan
    return float(chi2.sf(statistic, degrees_of_freedom))


def _check_columns(train_df: Any, validation_df: Any, columns: list[str]) -> None:
    train_missing = [column for column in columns if column not in train_df.columns]
    validation_missing = [column for column in columns if column not in validation_df.columns]
    missing = sorted(set(train_missing + validation_missing))
    if missing:
        raise KeyError(f"Column {missing[0]!r} must exist in train and validation data.")
