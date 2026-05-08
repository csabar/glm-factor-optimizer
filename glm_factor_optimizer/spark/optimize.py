"""Optuna factor optimization for Spark dataframes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._deps import require_pyspark
from .bins import apply_spec, make_categorical_groups, make_numeric_bins
from .metrics import model_deviance
from .model import fit_glm

JsonDict = dict[str, Any]
PenaltyContext = dict[str, Any]
PenaltyFunction = Callable[[PenaltyContext], float]
PenaltyInput = PenaltyFunction | Sequence[PenaltyFunction] | Mapping[str, PenaltyFunction]


@dataclass(slots=True)
class SparkOptimizationResult:
    """Result of optimizing one Spark factor."""

    factor: str
    kind: str
    output: str
    spec: JsonDict
    score: float
    validation_deviance: float
    trials: pd.DataFrame
    fixed: list[str]
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
    fixed: list[str] | None = None,
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

    Optuna runs on the driver. Each trial applies a JSON spec, fits Spark ML's
    GLR model, scores validation, and returns a scalar objective to Optuna.
    """

    optuna = _require_optuna()
    require_pyspark()
    kind = kind.lower().strip()
    if kind not in {"numeric", "categorical"}:
        raise ValueError("kind must be 'numeric' or 'categorical'.")
    fixed = list(fixed or [])
    extra_penalties = _named_penalties(penalties)

    cached_inputs: list[Any] = []
    if cache_input:
        train_df = train_df.cache()
        validation_df = validation_df.cache()
        train_df.count()
        validation_df.count()
        cached_inputs.extend([train_df, validation_df])

    base_numeric = make_numeric_bins(train_df, factor, bins=n_prebins) if kind == "numeric" else None

    def objective(trial: Any) -> float:
        spec = (
            _numeric_spec_from_trial(trial, base_numeric)
            if kind == "numeric"
            else _categorical_spec_from_trial(trial, train_df, factor, target, exposure, weight)
        )
        trial.set_user_attr("spec", spec)
        train_transformed = None
        validation_transformed = None
        try:
            train_transformed = apply_spec(train_df, spec)
            validation_transformed = apply_spec(validation_df, spec)
            if cache_trials:
                train_transformed = train_transformed.cache()
                validation_transformed = validation_transformed.cache()
                train_transformed.count()
                validation_transformed.count()

            model = fit_glm(
                train_transformed,
                target=target,
                factors=[*fixed, str(spec["output"])],
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
                "fixed": fixed,
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
            if cache_trials:
                if train_transformed is not None:
                    train_transformed.unpersist()
                if validation_transformed is not None:
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
            fixed=fixed,
            penalty_breakdown={
                "complexity": float(best.user_attrs.get("complexity_penalty", 0.0)),
                "small_bin": float(best.user_attrs.get("small_bin_penalty", 0.0)),
                "custom": float(best.user_attrs.get("custom_penalty", 0.0)),
            },
        )
    finally:
        for cached in cached_inputs:
            cached.unpersist()


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
    train_df: Any,
    factor: str,
    target: str,
    exposure: str | None,
    weight: str | None,
) -> JsonDict:
    base = make_categorical_groups(train_df, factor, target, exposure=exposure, weight=weight)
    cutpoints = [
        index
        for index in range(1, len(base["order"]))
        if trial.suggest_categorical(f"group_cut_{index}", [False, True])
    ]
    return make_categorical_groups(
        train_df,
        factor,
        target,
        exposure=exposure,
        weight=weight,
        cutpoints=cutpoints,
    )


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
