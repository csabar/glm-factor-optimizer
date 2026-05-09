"""Optuna optimization for one factor at a time."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .bins import apply_spec, make_categorical_groups, make_numeric_bins
from .metrics import model_deviance
from .model import fit_glm
from .penalties import (
    PenaltyContext,
    PenaltyFunction,
    PenaltyInput,
    small_bin_size_penalty,
    small_count_penalty,
    train_validation_gap_penalty,
)

JsonDict = dict[str, Any]


@dataclass(slots=True)
class OptimizationResult:
    """Result of optimizing one factor's bins or groups.

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
        Trial history table.
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


def _require_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ImportError("Install optuna to use optimize_factor: pip install optuna") from exc
    return optuna


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
    train_df: pd.DataFrame,
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
    df: pd.DataFrame,
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
            .reset_index()
            .rename(columns={output: "bin"})
        )
    elif weight is not None:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(weight, "sum"), actual=(target, "sum"), rows=(target, "size"))
            .reindex(observed, fill_value=0.0)
            .reset_index()
            .rename(columns={output: "bin"})
        )
    else:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(target, "size"), actual=(target, "sum"), rows=(target, "size"))
            .reindex(observed, fill_value=0.0)
            .reset_index()
            .rename(columns={output: "bin"})
        )
    return table


def _trial_table(study: Any) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        row = {"number": trial.number, "value": trial.value, "state": str(trial.state)}
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


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
    return float(total), breakdown


def optimize_factor(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    target: str,
    exposure: str | None = None,
    factor: str,
    kind: str = "numeric",
    family: str = "poisson",
    fixed_factors: list[str] | None = None,
    weight: str | None = None,
    prediction: str = "predicted_count",
    trials: int = 50,
    max_bins: int = 8,
    n_prebins: int = 10,
    min_bin_size: float = 100.0,
    bin_penalty: float = 0.001,
    small_bin_penalty: float = 0.01,
    penalties: PenaltyInput | None = None,
    seed: int | None = 42,
) -> OptimizationResult:
    """Optimize one numeric or categorical factor using validation deviance.

    Parameters
    ----------
    train_df:
        Training data used to derive candidate specs and fit candidate models.
    validation_df:
        Validation data used to score candidate specs.
    target:
        Observed outcome column.
    exposure:
        Optional exposure column used as a log offset.
    factor:
        Raw factor column to optimize.
    kind:
        Factor kind, either ``"numeric"`` or ``"categorical"``.
    family:
        GLM family name, such as ``"poisson"``, ``"gamma"``, or
        ``"gaussian"``.
    fixed_factors:
        Already transformed model columns to keep in the GLM.
    weight:
        Optional row-weight column.
    prediction:
        Prediction name used by candidate models.
    trials:
        Number of Optuna trials.
    max_bins:
        Maximum number of bins or groups allowed before extra complexity
        penalty applies.
    n_prebins:
        Number of numeric pre-bins used to define candidate cutpoints.
    min_bin_size:
        Minimum bin size threshold.
    bin_penalty:
        Penalty multiplier for each bin and excess bin.
    small_bin_penalty:
        Penalty multiplier for each bin below the minimum size threshold.
    penalties:
        Optional extra penalty callable, list of callables, or mapping of
        penalty names to callables.
    seed:
        Optional Optuna sampler seed.

    Returns
    -------
    OptimizationResult
        Best spec, score, validation deviance, trial history, and penalty
        breakdown.

    Notes
    -----
    Extra penalties can be passed as a callable, a list of callables, or a
    ``dict`` of named callables. Each function receives a context dictionary and
    returns a numeric penalty to add to the objective score.
    """

    kind = kind.lower().strip()
    if kind not in {"numeric", "categorical"}:
        raise ValueError("kind must be 'numeric' or 'categorical'.")
    fixed_factors_list = list(fixed_factors) if fixed_factors is not None else []
    required_columns = [target, factor, *fixed_factors_list] + ([exposure] if exposure is not None else [])
    for column in required_columns:
        if column not in train_df.columns or column not in validation_df.columns:
            raise KeyError(f"Column {column!r} must exist in train and validation data.")

    optuna = _require_optuna()
    extra_penalties = _named_penalties(penalties)
    base_numeric = make_numeric_bins(train_df[factor], bins=n_prebins, column=factor) if kind == "numeric" else None
    output = str(base_numeric["output"]) if base_numeric is not None else f"{factor}_group"

    def objective(trial: Any) -> float:
        if kind == "numeric":
            spec = _numeric_spec_from_trial(trial, base_numeric)
        else:
            spec = _categorical_spec_from_trial(trial, train_df, factor, target, exposure, weight)

        trial.set_user_attr("spec", spec)
        try:
            train_transformed = apply_spec(train_df, spec)
            validation_transformed = apply_spec(validation_df, spec)
            model = fit_glm(
                train_transformed,
                target=target,
                exposure=exposure,
                family=family,
                factors=[*fixed_factors_list, str(spec["output"])],
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

            table = _bin_summary(
                train_transformed,
                output=str(spec["output"]),
                target=target,
                exposure=exposure,
                weight=weight,
                labels=list(spec["labels"]),
            )
            bin_count = int(len(table))
            small_bins = int((table["bin_size"] < min_bin_size).sum())
            complexity = bin_count * bin_penalty + max(bin_count - max_bins, 0) * bin_penalty
            size_penalty = small_bins * small_bin_penalty
            context = {
                "train": train_transformed,
                "validation": validation_transformed,
                "model": model,
                "spec": spec,
                "bin_table": table,
                "train_prediction": train_prediction,
                "validation_prediction": validation_prediction,
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

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_trial
    spec = best.user_attrs["spec"]
    return OptimizationResult(
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
