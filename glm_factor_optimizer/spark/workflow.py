"""Spark sequential modeling workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import SparkGLM
from .model import FittedSparkGLM
from .optimize import PenaltyInput, SparkOptimizationResult
from .split import split


@dataclass(slots=True)
class SparkWorkflowResult:
    """Artifacts from a Spark workflow run.

    Parameters
    ----------
    train:
        Training Spark dataframe after selected specs are applied.
    validation:
        Scored validation Spark dataframe.
    holdout:
        Scored holdout Spark dataframe.
    model:
        Final fitted Spark GLM.
    selected_factors:
        Transformed model columns selected for the final GLM.
    specs:
        Mapping from raw factor name to JSON-serializable spec.
    optimizations:
        Per-factor optimization results.
    validation_report:
        Validation report Spark dataframes.
    holdout_report:
        Holdout report Spark dataframes.
    """

    train: Any
    validation: Any
    holdout: Any
    model: FittedSparkGLM
    selected_factors: list[str]
    specs: dict[str, dict]
    optimizations: list[SparkOptimizationResult]
    validation_report: dict[str, Any]
    holdout_report: dict[str, Any]


@dataclass(slots=True)
class SparkGLMWorkflow:
    """Sequential Spark workflow around ``SparkGLM``.

    Parameters
    ----------
    target:
        Observed outcome column.
    family:
        Spark GLR family name.
    exposure:
        Optional exposure column used as a log offset.
    weight:
        Optional row-weight column.
    prediction:
        Prediction column written by Spark GLR.
    link:
        Optional Spark GLR link function override.
    trials:
        Number of Optuna trials per optimized factor.
    max_bins:
        Maximum number of bins or groups allowed per factor.
    n_prebins:
        Number of numeric pre-bins used to define candidate cutpoints.
    min_bin_size:
        Default minimum bin size used by optimization.
    factor_kinds:
        Mapping from raw factor name to ``"numeric"`` or ``"categorical"``.
    penalties:
        Optional extra optimization penalties.
    seed:
        Random seed used for splitting and optimization.
    cache_input:
        Whether to cache train and validation dataframes during optimization.
    cache_trials:
        Whether to cache transformed trial dataframes.
    """

    target: str
    family: str = "poisson"
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "prediction"
    link: str | None = None
    trials: int = 50
    max_bins: int = 8
    n_prebins: int = 10
    min_bin_size: float = 100.0
    factor_kinds: dict[str, str] = field(default_factory=dict)
    penalties: PenaltyInput | None = None
    seed: int | None = 42
    cache_input: bool = True
    cache_trials: bool = False

    def fit(
        self,
        df: Any,
        factors: list[str],
        *,
        train_fraction: float = 0.6,
        validation_fraction: float = 0.2,
        holdout_fraction: float = 0.2,
        time: str | None = None,
    ) -> SparkWorkflowResult:
        """Optimize each factor sequentially, then fit a final Spark GLM.

        Parameters
        ----------
        df:
            Source Spark dataframe.
        factors:
            Raw candidate factors to optimize.
        train_fraction:
            Fraction assigned to the training sample.
        validation_fraction:
            Fraction assigned to the validation sample.
        holdout_fraction:
            Fraction assigned to the holdout sample.
        time:
            Optional time-like column for ordered splitting instead of random
            splitting.

        Returns
        -------
        SparkWorkflowResult
            Final Spark model, scored samples, selected specs, optimizations,
            and reports.
        """

        train, validation, holdout = split(
            df,
            train=train_fraction,
            validation=validation_fraction,
            holdout=holdout_fraction,
            seed=self.seed or 42,
            time=time,
        )
        glm = SparkGLM(
            target=self.target,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            prediction=self.prediction,
            link=self.link,
        )
        specs: dict[str, dict] = {}
        selected: list[str] = []
        optimizations: list[SparkOptimizationResult] = []

        for factor in factors:
            result = glm.optimize(
                train,
                validation,
                factor,
                kind=self.factor_kinds.get(factor) or _infer_kind(train, factor),
                fixed=selected,
                trials=self.trials,
                max_bins=self.max_bins,
                n_prebins=self.n_prebins,
                min_bin_size=self.min_bin_size,
                penalties=self.penalties,
                seed=self.seed,
                cache_input=self.cache_input,
                cache_trials=self.cache_trials,
            )
            specs[factor] = result.spec
            optimizations.append(result)
            train = glm.apply(train, result.spec)
            validation = glm.apply(validation, result.spec)
            holdout = glm.apply(holdout, result.spec)
            selected.append(result.output)

        model = glm.fit(train, selected)
        validation_scored = glm.predict(validation, model)
        holdout_scored = glm.predict(holdout, model)
        return SparkWorkflowResult(
            train=train,
            validation=validation_scored,
            holdout=holdout_scored,
            model=model,
            selected_factors=selected,
            specs=specs,
            optimizations=optimizations,
            validation_report=glm.report(validation_scored),
            holdout_report=glm.report(holdout_scored),
        )


def _infer_kind(df: Any, factor: str) -> str:
    simple = df.schema[factor].dataType.simpleString().split("(")[0]
    if simple in {"byte", "short", "integer", "long", "float", "double", "decimal"}:
        return "numeric"
    return "categorical"
