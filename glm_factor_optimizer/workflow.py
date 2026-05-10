"""Higher-level sequential modeling workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .core import GLM
from .diagnostics import find_interactions
from .model import FittedGLM
from .optimize import OptimizationResult, PenaltyInput
from .runs import RunLogger
from .screening import rank_factors
from .split import split


@dataclass(slots=True)
class WorkflowResult:
    """Artifacts from sequential one-factor optimization and final fitting.

    Parameters
    ----------
    train:
        Scored training data after selected specs are applied.
    validation:
        Scored validation data after selected specs are applied.
    holdout:
        Scored holdout data after selected specs are applied.
    model:
        Final fitted GLM.
    selected_factors:
        Transformed model columns selected for the final GLM.
    selected_raw_factors:
        Raw input factors whose specs were selected.
    specs:
        Mapping from raw factor name to JSON-serializable spec.
    optimizations:
        Per-factor optimization results.
    ranking:
        Optional candidate-ranking table.
    validation_report:
        Validation report tables.
    holdout_report:
        Holdout report tables.
    coefficients:
        Final model coefficient table.
    diagnostics:
        Optional interaction diagnostics.
    run_dir:
        Optional filesystem run directory created by logging.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame
    model: FittedGLM
    selected_factors: list[str]
    selected_raw_factors: list[str]
    specs: dict[str, dict]
    optimizations: list[OptimizationResult]
    ranking: pd.DataFrame | None
    validation_report: dict[str, pd.DataFrame]
    holdout_report: dict[str, pd.DataFrame]
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame | None = None
    run_dir: str | None = None


@dataclass(slots=True)
class GLMWorkflow:
    """Sequential factor optimization workflow around the simple ``GLM`` API.

    Parameters
    ----------
    target:
        Observed outcome column.
    family:
        GLM family name, such as ``"poisson"``, ``"gamma"``, or
        ``"gaussian"``.
    exposure:
        Optional exposure column used as a log offset.
    weight:
        Optional row-weight column.
    prediction:
        Prediction column written to scored outputs.
    trials:
        Number of Optuna trials per optimized factor.
    max_bins:
        Maximum number of bins or groups allowed per factor.
    n_prebins:
        Number of numeric pre-bins used to define candidate cutpoints.
    min_bin_size:
        Default minimum bin size used by optimization and diagnostics.
    factor_kinds:
        Mapping from raw factor name to ``"numeric"`` or ``"categorical"``.
    penalties:
        Optional extra optimization penalties.
    seed:
        Random seed used for splitting and optimization.
    rank_candidates:
        Whether to rank factors before sequential optimization.
    top_n:
        Optional number of ranked factors to optimize.
    screening_bins:
        Number of numeric bins used during factor screening.
    screening_max_groups:
        Maximum number of ordered categorical groups used during screening.
    interaction_diagnostics:
        Whether to produce interaction diagnostics after final fitting.
    interaction_min_bin_size:
        Optional minimum cell size for interaction diagnostics.
    output_dir:
        Optional parent directory for filesystem logging.
    logger:
        Optional existing run logger to use instead of creating one.
    """

    target: str
    family: str = "poisson"
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "predicted"
    trials: int = 50
    max_bins: int = 8
    n_prebins: int = 10
    min_bin_size: float = 100.0
    factor_kinds: dict[str, str] = field(default_factory=dict)
    penalties: PenaltyInput | None = None
    seed: int | None = 42
    rank_candidates: bool = False
    top_n: int | None = None
    screening_bins: int = 5
    screening_max_groups: int = 6
    interaction_diagnostics: bool = False
    interaction_min_bin_size: float | None = None
    output_dir: str | Path | None = None
    logger: RunLogger | None = None

    def fit(
        self,
        df: pd.DataFrame,
        factors: list[str],
        *,
        train_fraction: float = 0.6,
        validation_fraction: float = 0.2,
        holdout_fraction: float = 0.2,
        time: str | None = None,
        time_split: str = "exact",
    ) -> WorkflowResult:
        """Optimize selected factors sequentially, then fit a final GLM.

        Parameters
        ----------
        df:
            Source data containing the outcome, factors, and optional exposure
            or weight columns.
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
        time_split:
            Time split strategy. Pandas supports only ``"exact"``.

        Returns
        -------
        WorkflowResult
            Final scored samples, selected specs, model artifacts, reports, and
            optional logging path.
        """

        train, validation, holdout = split(
            df,
            train=train_fraction,
            validation=validation_fraction,
            holdout=holdout_fraction,
            seed=self.seed or 42,
            time=time,
            time_split=time_split,
        )
        glm = GLM(
            target=self.target,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            prediction=self.prediction,
        )

        specs: dict[str, dict] = {}
        selected: list[str] = []
        selected_raw: list[str] = []
        optimizations: list[OptimizationResult] = []
        ranking: pd.DataFrame | None = None
        candidate_factors = list(factors)
        created_logger = self.logger is None and self.output_dir is not None
        active_logger = self.logger or (
            RunLogger(self.output_dir, name="glm_workflow") if created_logger else None
        )

        if self.rank_candidates or self.top_n is not None:
            ranking = rank_factors(
                train,
                validation,
                target=self.target,
                factors=candidate_factors,
                family=self.family,
                exposure=self.exposure,
                weight=self.weight,
                factor_kinds=self.factor_kinds,
                bins=self.screening_bins,
                max_groups=self.screening_max_groups,
                prediction=self.prediction,
                min_bin_size=self.min_bin_size,
            )
            candidate_factors = ranking.loc[ranking["error"].isna(), "factor"].tolist()
            if self.top_n is not None:
                candidate_factors = candidate_factors[: self.top_n]

        for factor in candidate_factors:
            result = glm.optimize(
                train,
                validation,
                factor,
                kind=self.factor_kinds.get(factor) or _infer_kind(train[factor]),
                fixed_factors=selected,
                trials=self.trials,
                max_bins=self.max_bins,
                n_prebins=self.n_prebins,
                min_bin_size=self.min_bin_size,
                penalties=self.penalties,
                seed=self.seed,
            )
            specs[factor] = result.spec
            optimizations.append(result)
            train = glm.apply(train, result.spec)
            validation = glm.apply(validation, result.spec)
            holdout = glm.apply(holdout, result.spec)
            selected.append(result.output)
            selected_raw.append(factor)

        model = glm.fit(train, selected)
        train_scored = glm.predict(train, model)
        validation_scored = glm.predict(validation, model)
        holdout_scored = glm.predict(holdout, model)
        validation_report = glm.report(validation_scored)
        holdout_report = glm.report(holdout_scored)
        coefficients = model.coefficients()
        diagnostics = None
        if self.interaction_diagnostics and len(selected) >= 2:
            diagnostics = find_interactions(
                train_scored,
                validation_scored,
                factors=selected,
                target=self.target,
                prediction=self.prediction,
                exposure=self.exposure,
                weight=self.weight,
                min_bin_size=self.interaction_min_bin_size
                if self.interaction_min_bin_size is not None
                else self.min_bin_size,
            )

        if active_logger is not None:
            _log_workflow(
                active_logger,
                workflow=self,
                factors=list(factors),
                selected_raw=selected_raw,
                selected=selected,
                specs=specs,
                optimizations=optimizations,
                ranking=ranking,
                validation_report=validation_report,
                holdout_report=holdout_report,
                coefficients=coefficients,
                diagnostics=diagnostics,
            )
            if created_logger:
                active_logger.close()

        return WorkflowResult(
            train=train_scored,
            validation=validation_scored,
            holdout=holdout_scored,
            model=model,
            selected_factors=selected,
            selected_raw_factors=selected_raw,
            specs=specs,
            optimizations=optimizations,
            ranking=ranking,
            validation_report=validation_report,
            holdout_report=holdout_report,
            coefficients=coefficients,
            diagnostics=diagnostics,
            run_dir=str(active_logger.path) if active_logger is not None else None,
        )


def run_workflow(
    df: pd.DataFrame,
    *,
    target: str,
    factors: list[str],
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    prediction: str = "predicted",
    factor_kinds: dict[str, str] | None = None,
    trials: int = 50,
    max_bins: int = 8,
    n_prebins: int = 10,
    min_bin_size: float = 100.0,
    penalties: PenaltyInput | None = None,
    rank_candidates: bool = False,
    top_n: int | None = None,
    screening_bins: int = 5,
    screening_max_groups: int = 6,
    interaction_diagnostics: bool = False,
    output_dir: str | Path | None = None,
    seed: int | None = 42,
) -> WorkflowResult:
    """Run the default sequential optimization workflow.

    Parameters
    ----------
    df:
        Source data.
    target:
        Observed outcome column.
    factors:
        Raw candidate factors to optimize.
    family:
        GLM family name, such as ``"poisson"``, ``"gamma"``, or
        ``"gaussian"``.
    exposure:
        Optional exposure column used as a log offset.
    weight:
        Optional row-weight column.
    prediction:
        Prediction column written to scored outputs.
    factor_kinds:
        Optional mapping from factor name to ``"numeric"`` or
        ``"categorical"``.
    trials:
        Number of Optuna trials per optimized factor.
    max_bins:
        Maximum number of bins or groups allowed per factor.
    n_prebins:
        Number of numeric pre-bins used to define candidate cutpoints.
    min_bin_size:
        Default minimum bin size used by optimization and diagnostics.
    penalties:
        Optional extra optimization penalties.
    rank_candidates:
        Whether to rank factors before sequential optimization.
    top_n:
        Optional number of ranked factors to optimize.
    screening_bins:
        Number of numeric bins used during factor screening.
    screening_max_groups:
        Maximum number of ordered categorical groups used during screening.
    interaction_diagnostics:
        Whether to produce interaction diagnostics after final fitting.
    output_dir:
        Optional parent directory for filesystem logging.
    seed:
        Random seed used for splitting and optimization.

    Returns
    -------
    WorkflowResult
        Final scored samples, selected specs, model artifacts, reports, and
        optional logging path.
    """

    workflow = GLMWorkflow(
        target=target,
        family=family,
        exposure=exposure,
        weight=weight,
        prediction=prediction,
        factor_kinds=factor_kinds or {},
        trials=trials,
        max_bins=max_bins,
        n_prebins=n_prebins,
        min_bin_size=min_bin_size,
        penalties=penalties,
        rank_candidates=rank_candidates,
        top_n=top_n,
        screening_bins=screening_bins,
        screening_max_groups=screening_max_groups,
        interaction_diagnostics=interaction_diagnostics,
        output_dir=output_dir,
        seed=seed,
    )
    return workflow.fit(df, factors)


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _log_workflow(
    logger: RunLogger,
    *,
    workflow: GLMWorkflow,
    factors: list[str],
    selected_raw: list[str],
    selected: list[str],
    specs: dict[str, dict],
    optimizations: list[OptimizationResult],
    ranking: pd.DataFrame | None,
    validation_report: dict[str, pd.DataFrame],
    holdout_report: dict[str, pd.DataFrame],
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
) -> None:
    logger.log_params(
        {
            "target": workflow.target,
            "family": workflow.family,
            "exposure": workflow.exposure,
            "weight": workflow.weight,
            "prediction": workflow.prediction,
            "trials": workflow.trials,
            "max_bins": workflow.max_bins,
            "n_prebins": workflow.n_prebins,
            "min_bin_size": workflow.min_bin_size,
            "factors": factors,
            "selected_raw_factors": selected_raw,
            "selected_factors": selected,
            "rank_candidates": workflow.rank_candidates,
            "top_n": workflow.top_n,
        }
    )
    logger.log_json("specs.json", specs)
    if ranking is not None:
        logger.log_frame("factor_ranking.csv", ranking.drop(columns=["spec"], errors="ignore"))
        logger.log_json(
            "factor_ranking_specs.json",
            {row["factor"]: row["spec"] for _, row in ranking.dropna(subset=["spec"]).iterrows()},
        )
    for result in optimizations:
        logger.log_frame(f"trials/{result.factor}_trials.csv", result.trials)
    logger.log_report("validation", validation_report)
    logger.log_report("holdout", holdout_report)
    logger.log_frame("coefficients.csv", coefficients)
    if diagnostics is not None:
        logger.log_frame("interaction_diagnostics.csv", diagnostics)
