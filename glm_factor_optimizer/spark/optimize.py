"""Optuna factor optimization for Spark dataframes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._deps import require_pyspark
from .bins import apply_spec, categorical_group_spec_from_order, category_target_order, make_numeric_bins
from .metrics import model_deviance
from .model import fit_glm

JsonDict = dict[str, Any]
PenaltyContext = dict[str, Any]
PenaltyFunction = Callable[[PenaltyContext], float]
PenaltyInput = PenaltyFunction | Sequence[PenaltyFunction] | Mapping[str, PenaltyFunction]

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SparkOptimizationResult:
    """Result of optimizing one Spark factor.

    Parameters
    ----------
    factor:
        Raw factor column that was optimized.
    kind:
        Factor kind, either ``"numeric"`` or ``"categorical"``.
    output:
        Transformed model column created by the best spec.
    spec:
        JSON-serializable best binning or grouping spec.
    score:
        Best objective value, including deviance and penalties.
    validation_deviance:
        Validation deviance for the best trial before added penalties.
    trials:
        Trial history table collected on the driver.
    fixed_factors:
        Fixed transformed columns included in the GLM during optimization.
    penalty_breakdown:
        Penalty components recorded for the best trial.
    """

    factor: str
    kind: str
    output: str
    spec: JsonDict
    score: float
    validation_deviance: float
    trials: pd.DataFrame
    fixed_factors: list[str]
    penalty_breakdown: dict[str, float]


def optimize_factor(
    train_df: Any,
    validation_df: Any,
    *,
    target: str,
    factor: str,
    family: str = "poisson",
    exposure: str | None = None,
    kind: str = "numeric",
    fixed_factors: list[str] | None = None,
    weight: str | None = None,
    prediction: str = "prediction",
    trials: int = 50,
    max_bins: int = 8,
    n_prebins: int = 10,
    min_bin_size: float = 100.0,
    bin_penalty: float = 0.001,
    small_bin_penalty: float = 0.01,
    penalties: PenaltyInput | None = None,
    seed: int | None = 42,
    cache_input: bool = True,
    cache_trials: bool = False,
) -> SparkOptimizationResult:
    """Optimize one Spark factor using validation deviance.

    Parameters
    ----------
    train_df:
        Spark training dataframe.
    validation_df:
        Spark validation dataframe.
    target:
        Observed outcome column.
    factor:
        Raw factor column to optimize.
    family:
        Spark GLR family name.
    exposure:
        Optional exposure column used as a log offset.
    kind:
        Factor kind, either ``"numeric"`` or ``"categorical"``.
    fixed_factors:
        Already transformed model columns to keep in the GLM.
    weight:
        Optional row-weight column.
    prediction:
        Prediction column written by Spark GLR.
    trials:
        Number of Optuna trials.
    max_bins:
        Maximum number of bins or groups allowed before extra penalty applies.
    n_prebins:
        Number of numeric pre-bins used to define candidate cutpoints.
    min_bin_size:
        Minimum bin size used by the small-bin penalty.
    bin_penalty:
        Penalty multiplier for additional bins.
    small_bin_penalty:
        Penalty multiplier for bins smaller than ``min_bin_size``.
    penalties:
        Optional extra penalty callable, list of callables, or mapping of names
        to callables.
    seed:
        Optional Optuna sampler seed.
    cache_input:
        Whether to cache train and validation dataframes during optimization.
    cache_trials:
        Whether to cache transformed trial dataframes.

    Returns
    -------
    SparkOptimizationResult
        Best spec, score, validation deviance, trial history, and penalty
        breakdown.

    Notes
    -----
    Optuna runs on the driver. Each trial applies a JSON spec, fits Spark ML's
    GLR model, scores validation, and returns a scalar objective to Optuna.
    """

    optuna = _require_optuna()
    require_pyspark()
    kind = kind.lower().strip()
    if kind not in {"numeric", "categorical"}:
        raise ValueError("kind must be 'numeric' or 'categorical'.")
    fixed_factors_list = list(fixed_factors) if fixed_factors is not None else []
    extra_penalties = _named_penalties(penalties)

    cached_inputs: list[Any] = []
    if cache_input:
        train_df, train_cached = _cache_if_supported(train_df)
        validation_df, validation_cached = _cache_if_supported(validation_df)
        cached_inputs.extend([frame for frame, cached in [(train_df, train_cached), (validation_df, validation_cached)] if cached])

    base_numeric = make_numeric_bins(train_df, factor, bins=n_prebins) if kind == "numeric" else None
    base_categorical = (
        category_target_order(train_df, factor, target, exposure=exposure, weight=weight)
        if kind == "categorical"
        else None
    )

    def objective(trial: Any) -> float:
        spec = (
            _numeric_spec_from_trial(trial, base_numeric)
            if kind == "numeric"
            else _categorical_spec_from_trial(trial, factor, base_categorical)
        )
        trial.set_user_attr("spec", spec)
        train_transformed = None
        validation_transformed = None
        train_trial_cached = False
        validation_trial_cached = False
        model = None
        try:
            train_transformed = apply_spec(train_df, spec)
            validation_transformed = apply_spec(validation_df, spec)
            if cache_trials:
                train_transformed, train_trial_cached = _cache_if_supported(train_transformed)
                validation_transformed, validation_trial_cached = _cache_if_supported(validation_transformed)
            else:
                train_trial_cached = False
                validation_trial_cached = False

            model = fit_glm(
                train_transformed,
                target=target,
                factors=[*fixed_factors_list, str(spec["output"])],
                family=family,
                exposure=exposure,
                weight=weight,
                prediction=prediction,
            )
            scored_train = model.transform(train_transformed)
            scored_validation = model.transform(validation_transformed)
            train_deviance = model_deviance(
                scored_train,
                target,
                prediction,
                family=family,
                weight=weight,
            )
            validation_deviance = model_deviance(
                scored_validation,
                target,
                prediction,
                family=family,
                weight=weight,
            )
            table = _bin_summary(train_transformed, str(spec["output"]), target, exposure, weight, list(spec["labels"]))
            bin_count = int(len(table))
            small_bins = int((table["bin_size"] < min_bin_size).sum())
            complexity = bin_count * bin_penalty + max(bin_count - max_bins, 0) * bin_penalty
            size_penalty = small_bins * small_bin_penalty
            context = {
                "train": train_transformed,
                "validation": validation_transformed,
                "scored_train": scored_train,
                "scored_validation": scored_validation,
                "model": model,
                "spec": spec,
                "bin_table": table,
                "target": target,
                "exposure": exposure,
                "family": family,
                "factor": factor,
                "kind": kind,
                "fixed_factors": fixed_factors_list,
                "train_deviance": float(train_deviance),
                "validation_deviance": float(validation_deviance),
                "bin_count": bin_count,
                "small_bins": small_bins,
            }
            custom_penalty, custom_breakdown = _extra_penalty(extra_penalties, context)
            score = float(validation_deviance + complexity + size_penalty + custom_penalty)
            trial.set_user_attr("train_deviance", float(train_deviance))
            trial.set_user_attr("validation_deviance", float(validation_deviance))
            trial.set_user_attr("bins", bin_count)
            trial.set_user_attr("min_bin_size", float(table["bin_size"].min()))
            trial.set_user_attr("small_bins", small_bins)
            trial.set_user_attr("fixed_factors", list(fixed_factors_list))
            trial.set_user_attr("complexity_penalty", float(complexity))
            trial.set_user_attr("small_bin_penalty", float(size_penalty))
            trial.set_user_attr("custom_penalty", float(custom_penalty))
            for penalty_name, penalty_value in custom_breakdown.items():
                trial.set_user_attr(f"penalty_{penalty_name}", float(penalty_value))
            trial.set_user_attr("score", score)
            return score
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            return float("inf")
        finally:
            if model is not None:
                model.release()
            if cache_trials:
                if train_transformed is not None and train_trial_cached:
                    train_transformed.unpersist()
                if validation_transformed is not None and validation_trial_cached:
                    validation_transformed.unpersist()

    try:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best = study.best_trial
        spec = best.user_attrs["spec"]
        return SparkOptimizationResult(
            factor=factor,
            kind=kind,
            output=str(spec["output"]),
            spec=spec,
            score=float(best.value),
            validation_deviance=float(best.user_attrs.get("validation_deviance", np.nan)),
            trials=_trial_table(study),
            fixed_factors=fixed_factors_list,
            penalty_breakdown={
                "complexity": float(best.user_attrs.get("complexity_penalty", 0.0)),
                "small_bin": float(best.user_attrs.get("small_bin_penalty", 0.0)),
                "custom": float(best.user_attrs.get("custom_penalty", 0.0)),
            },
        )
    finally:
        for cached in cached_inputs:
            cached.unpersist()


def _cache_if_supported(df: Any) -> tuple[Any, bool]:
    try:
        cached = df.cache()
    except Exception as exc:
        if _is_unsupported_cache_error(exc):
            return df, False
        raise
    try:
        cached.count()
    except Exception as exc:
        try:
            cached.unpersist()
        except Exception:
            _LOGGER.debug("Unable to unpersist failed Spark cache.", exc_info=True)
        if _is_unsupported_cache_error(exc):
            return df, False
        raise
    return cached, True


def _is_unsupported_cache_error(exc: Exception) -> bool:
    text = str(exc)
    return "NOT_SUPPORTED_WITH_SERVERLESS" in text or "PERSIST TABLE is not supported" in text


def _numeric_spec_from_trial(trial: Any, base: JsonDict) -> JsonDict:
    edges = [float(edge) for edge in base["edges"]]
    selected = [
        edge
        for index, edge in enumerate(edges[1:-1], start=1)
        if trial.suggest_categorical(f"cut_{index}", [False, True])
    ]
    final_edges = sorted({edges[0], *selected, edges[-1]})
    labels = [f"bin_{index + 1}" for index in range(len(final_edges) - 1)]
    return {
        "type": "numeric",
        "column": base["column"],
        "output": base["output"],
        "method": "optuna_quantile_subset",
        "edges": final_edges,
        "labels": labels,
        "prebin_edges": edges,
    }


def _categorical_spec_from_trial(
    trial: Any,
    factor: str,
    target_order: pd.DataFrame,
) -> JsonDict:
    base = categorical_group_spec_from_order(factor, target_order)
    cutpoints = [
        index
        for index in range(1, len(base["order"]))
        if trial.suggest_categorical(f"group_cut_{index}", [False, True])
    ]
    return categorical_group_spec_from_order(factor, target_order, cutpoints=cutpoints)


def _bin_summary(
    df: Any,
    output: str,
    target: str,
    exposure: str | None,
    weight: str | None,
    labels: list[str],
) -> pd.DataFrame:
    spark = require_pyspark()
    F = spark.functions
    if exposure is not None:
        aggs = [F.sum(F.col(exposure).cast("double")).alias("bin_size")]
    elif weight is not None:
        aggs = [F.sum(F.col(weight).cast("double")).alias("bin_size")]
    else:
        aggs = [F.count(F.lit(1)).cast("double").alias("bin_size")]
    aggs.extend([
        F.sum(F.col(target).cast("double")).alias("actual"),
        F.count(F.lit(1)).alias("rows"),
    ])
    table = df.groupBy(output).agg(*aggs).toPandas().rename(columns={output: "bin"})
    observed = pd.DataFrame({"bin": list(labels)})
    return observed.merge(table, on="bin", how="left").fillna({"bin_size": 0.0, "actual": 0.0, "rows": 0})


def _named_penalties(penalties: PenaltyInput | None) -> list[tuple[str, PenaltyFunction]]:
    if penalties is None:
        return []
    if callable(penalties):
        return [("custom", penalties)]
    if isinstance(penalties, Mapping):
        return [(str(name), function) for name, function in penalties.items()]
    return [(f"custom_{index + 1}", function) for index, function in enumerate(penalties)]


def _extra_penalty(
    penalties: list[tuple[str, PenaltyFunction]],
    context: PenaltyContext,
) -> tuple[float, dict[str, float]]:
    breakdown: dict[str, float] = {}
    total = 0.0
    for name, function in penalties:
        value = float(function(context))
        breakdown[name] = value
        total += value
    return total, breakdown


def _trial_table(study: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {"number": trial.number, "value": trial.value, "state": str(trial.state)}
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def _require_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ImportError("Install optuna to use Spark factor optimization.") from exc
    return optuna
