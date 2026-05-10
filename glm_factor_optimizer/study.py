"""Interactive notebook-oriented GLM study workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._backend import is_spark_dataframe
from .bins import apply_spec
from .core import GLM
from .diagnostics import find_interactions as find_interaction_candidates
from .factor import FactorBlock
from .metrics import model_deviance, weighted_mae, weighted_rmse
from .model import FittedGLM
from .optimize import OptimizationResult, PenaltyInput, optimize_factor
from .runs import RunLogger
from .screening import rank_factors
from .split import split as split_frame
from .validation import (
    holdout_final_report,
    model_version_comparison,
    train_validation_comparison,
    validation_report as make_validation_report,
)

JsonDict = dict[str, Any]


class GLMStudy:
    """Stateful notebook workbench for iterative GLM model design."""

    def __new__(cls, df: pd.DataFrame, *args: Any, **kwargs: Any) -> GLMStudy:
        if cls is GLMStudy and is_spark_dataframe(df):
            from .spark.study import SparkGLMStudy

            return SparkGLMStudy(df, *args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target: str,
        family: str = "poisson",
        exposure: str | None = None,
        weight: str | None = None,
        prediction: str = "predicted",
        factor_kinds: dict[str, str] | None = None,
        min_bin_size: float = 100.0,
        seed: int | None = 42,
    ) -> None:
        """Create an interactive study for GLM factor design.

        Parameters
        ----------
        df:
            Source data containing the outcome, candidate factor columns, and
            optional exposure or weight columns.
        target:
            Name of the observed outcome column.
        family:
            GLM family name. Supported values follow :func:`fit_glm`, such as
            ``"poisson"``, ``"gamma"``, and ``"gaussian"``.
        exposure:
            Optional exposure column used as a log offset when fitting the GLM.
        weight:
            Optional row-weight column used for model fitting and diagnostics.
        prediction:
            Name of the prediction column written to scored data.
        factor_kinds:
            Optional mapping from raw factor name to ``"numeric"`` or
            ``"categorical"``. Unspecified factors are inferred when possible.
        min_bin_size:
            Default minimum bin size used by optimization, diagnostics, and
            interaction screening.
        seed:
            Default random seed for splitting and optimization.
        """

        self.df = df.copy()
        self.target = target
        self.family = family
        self.exposure = exposure
        self.weight = weight
        self.prediction = prediction
        self.factor_kinds = dict(factor_kinds or {})
        self.min_bin_size = min_bin_size
        self.seed = seed
        self.glm = GLM(
            target=target,
            family=family,
            exposure=exposure,
            weight=weight,
            prediction=prediction,
        )

        self.train: pd.DataFrame | None = None
        self.validation: pd.DataFrame | None = None
        self.holdout: pd.DataFrame | None = None

        self.specs: dict[str, JsonDict] = {}
        self.optimizations: dict[str, OptimizationResult] = {}
        self.accepted_raw_factors: list[str] = []
        self.rejected_factors: list[str] = []
        self.interaction_specs: dict[str, JsonDict] = {}
        self.pending_interactions: dict[str, dict[str, Any]] = {}

        self.ranking: pd.DataFrame | None = None
        self.interaction_diagnostics: pd.DataFrame | None = None
        self.history: list[dict[str, Any]] = []
        self.model_versions: list[dict[str, Any]] = []

        self.current_model: FittedGLM | None = None
        self.train_scored: pd.DataFrame | None = None
        self.validation_scored: pd.DataFrame | None = None
        self.holdout_scored: pd.DataFrame | None = None
        self.validation_reports: dict[str, pd.DataFrame] | None = None
        self.holdout_reports: dict[str, pd.DataFrame] | None = None
        self.finalized = False

    @property
    def selected_factors(self) -> list[str]:
        """Return accepted transformed model columns.

        Returns
        -------
        list[str]
            Model-ready columns created by accepted factor specs and accepted
            interaction specs.
        """

        outputs = [str(self.specs[factor]["output"]) for factor in self.accepted_raw_factors]
        outputs.extend(str(spec["output"]) for spec in self.interaction_specs.values())
        return outputs

    def split(
        self,
        *,
        train_fraction: float = 0.6,
        validation_fraction: float = 0.2,
        holdout_fraction: float = 0.2,
        seed: int | None = None,
        time: str | None = None,
        time_split: str = "exact",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, validation, and holdout samples.

        Parameters
        ----------
        train_fraction:
            Fraction of rows assigned to the training sample.
        validation_fraction:
            Fraction of rows assigned to the validation sample.
        holdout_fraction:
            Fraction of rows assigned to the holdout sample.
        seed:
            Optional split seed. When omitted, the study seed is used.
        time:
            Optional time-like column for ordered splitting instead of random
            splitting.
        time_split:
            Time split strategy. Pandas supports only ``"exact"``.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            The train, validation, and holdout samples.
        """

        self.train, self.validation, self.holdout = split_frame(
            self.df,
            train=train_fraction,
            validation=validation_fraction,
            holdout=holdout_fraction,
            seed=seed if seed is not None else self.seed or 42,
            time=time,
            time_split=time_split,
        )
        self._invalidate_model(include_holdout=True)
        self._record("split", comment=f"train={train_fraction}, validation={validation_fraction}, holdout={holdout_fraction}")
        return self.train, self.validation, self.holdout

    def rank_candidates(
        self,
        factors: list[str],
        *,
        bins: int = 5,
        max_groups: int = 6,
        min_bin_size: float | None = None,
        factor_kinds: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Rank candidate factors for notebook screening.

        Parameters
        ----------
        factors:
            Raw candidate factor columns to screen.
        bins:
            Number of simple numeric bins used during screening.
        max_groups:
            Maximum number of ordered categorical groups used during screening.
        min_bin_size:
            Optional minimum bin size override for screening diagnostics.
        factor_kinds:
            Optional per-call mapping from factor name to ``"numeric"`` or
            ``"categorical"``.

        Returns
        -------
        pandas.DataFrame
            Ranked factor table with validation improvement, stability columns,
            and JSON-serializable specs where available.
        """

        self.require_split()
        kinds = {**self.factor_kinds, **(factor_kinds or {})}
        self.ranking = rank_factors(
            self.train,
            self.validation,
            target=self.target,
            factors=factors,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            factor_kinds=kinds,
            bins=bins,
            max_groups=max_groups,
            prediction=self.prediction,
            min_bin_size=min_bin_size if min_bin_size is not None else self.min_bin_size,
        )
        self._record("rank", comment=f"ranked {len(factors)} factors")
        return self.ranking

    def factor(self, factor: str, kind: str | None = None) -> FactorBlock:
        """Return an interactive block for one raw factor.

        Parameters
        ----------
        factor:
            Raw input column to design, bin, or group.
        kind:
            Optional explicit factor kind, either ``"numeric"`` or
            ``"categorical"``. When omitted, the study uses configured kinds or
            infers from the data.

        Returns
        -------
        FactorBlock
            Notebook-friendly object for manual binning, optimization,
            comparison, acceptance, and rejection.
        """

        frame = self.train if self.train is not None else self.df
        if factor not in frame.columns:
            raise KeyError(f"Column {factor!r} is missing.")
        resolved_kind = kind or self.factor_kinds.get(factor) or _infer_kind(frame[factor])
        self.factor_kinds[factor] = resolved_kind
        block = FactorBlock(self, factor=factor, kind=resolved_kind)
        if factor in self.specs:
            block.spec = dict(self.specs[factor])
            block.metadata["source"] = "accepted"
        return block

    def accept(
        self,
        factor: str | FactorBlock,
        spec: JsonDict | None = None,
        *,
        comment: str | None = None,
    ) -> None:
        """Accept a factor block or explicit factor/spec into the study.

        Parameters
        ----------
        factor:
            A :class:`FactorBlock` with a proposed spec, or the raw factor name
            when ``spec`` is supplied separately.
        spec:
            JSON-serializable binning/grouping spec to accept when accepting by
            factor name.
        comment:
            Optional audit note stored in the study history.
        """

        self.require_split()
        factor_name, accepted_spec, optimization = _factor_parts(factor, spec)
        existed = factor_name in self.specs
        before_score = self._current_validation_deviance()
        self.specs[factor_name] = dict(accepted_spec)
        if isinstance(factor, FactorBlock):
            self.factor_kinds[factor_name] = factor.kind
        if optimization is not None:
            self.optimizations[factor_name] = optimization
        if factor_name not in self.accepted_raw_factors:
            self.accepted_raw_factors.append(factor_name)
        after_score = self._current_validation_deviance()
        self._invalidate_model(include_holdout=True)
        self._record(
            "refine" if existed else "accept",
            factor=factor_name,
            spec=accepted_spec,
            score_before=before_score,
            score_after=after_score,
            comment=comment,
        )

    def reject(self, factor: str | FactorBlock, *, comment: str | None = None) -> None:
        """Record a rejected factor proposal.

        Parameters
        ----------
        factor:
            Factor block or raw factor name being rejected.
        comment:
            Optional audit note stored in the study history.
        """

        factor_name = factor.factor if isinstance(factor, FactorBlock) else factor
        if factor_name not in self.rejected_factors:
            self.rejected_factors.append(factor_name)
        spec = factor.spec if isinstance(factor, FactorBlock) else None
        self._record("reject", factor=factor_name, spec=spec, comment=comment)

    def optimize_factor_block(
        self,
        factor: str,
        *,
        kind: str,
        trials: int = 50,
        max_bins: int = 8,
        n_prebins: int = 10,
        min_bin_size: float | None = None,
        bin_penalty: float = 0.001,
        small_bin_penalty: float = 0.01,
        penalties: PenaltyInput | None = None,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Optimize one factor against the current accepted model.

        Parameters
        ----------
        factor:
            Raw factor column to optimize.
        kind:
            Factor kind, either ``"numeric"`` or ``"categorical"``.
        trials:
            Number of Optuna trials.
        max_bins:
            Maximum number of bins or groups to allow.
        n_prebins:
            Number of numeric pre-bins used to define candidate cutpoints.
        min_bin_size:
            Optional minimum bin size override.
        bin_penalty:
            Penalty multiplier for additional bins.
        small_bin_penalty:
            Penalty multiplier for bins smaller than ``min_bin_size``.
        penalties:
            Optional extra penalty callables or named penalty mapping.
        seed:
            Optional Optuna sampler seed. When omitted, the study seed is used.

        Returns
        -------
        OptimizationResult
            Best spec, objective value, and trial history.
        """

        self.require_split()
        train, validation, _ = self._transformed_frames(exclude_factor=factor, include_holdout=False)
        return optimize_factor(
            train,
            validation,
            target=self.target,
            exposure=self.exposure,
            factor=factor,
            kind=kind,
            family=self.family,
            fixed_factors=self._fixed_factor_outputs(exclude_factor=factor),
            weight=self.weight,
            prediction=self.prediction,
            trials=trials,
            max_bins=max_bins,
            n_prebins=n_prebins,
            min_bin_size=min_bin_size if min_bin_size is not None else self.min_bin_size,
            bin_penalty=bin_penalty,
            small_bin_penalty=small_bin_penalty,
            penalties=penalties,
            seed=seed if seed is not None else self.seed,
        )

    def evaluate_candidate(
        self,
        factor: str,
        spec: JsonDict,
        *,
        return_scored: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Compare a proposed spec with the current accepted model.

        Parameters
        ----------
        factor:
            Raw factor name represented by the proposed spec.
        spec:
            JSON-serializable binning or grouping spec to evaluate.
        return_scored:
            When ``True``, also return the validation data scored by the
            candidate model.

        Returns
        -------
        pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame]
            Candidate comparison table, optionally paired with scored
            validation data.
        """

        self.require_split()
        baseline = self._current_validation_deviance()
        train, validation, _ = self._transformed_frames(exclude_factor=factor, include_holdout=False)
        train = apply_spec(train, spec)
        validation = apply_spec(validation, spec)
        candidate_factors = [*self._fixed_factor_outputs(exclude_factor=factor), str(spec["output"])]
        model = self.glm.fit(train, candidate_factors)
        train_scored = self.glm.predict(train, model)
        validation_scored = self.glm.predict(validation, model)
        train_deviance = self._scored_deviance(train_scored)
        validation_deviance = self._scored_deviance(validation_scored)
        table = _spec_bin_table(
            train,
            output=str(spec["output"]),
            target=self.target,
            exposure=self.exposure,
            weight=self.weight,
        )
        comparison = pd.DataFrame(
            [
                {
                    "factor": factor,
                    "output": str(spec["output"]),
                    "current_validation_deviance": baseline,
                    "candidate_train_deviance": train_deviance,
                    "candidate_validation_deviance": validation_deviance,
                    "validation_improvement": baseline - validation_deviance,
                    "bins": int(len(table)),
                    "min_bin_size": float(table["bin_size"].min()) if len(table) else np.nan,
                }
            ]
        )
        if return_scored:
            return comparison, validation_scored
        return comparison

    def refine_factor(
        self,
        factor: str,
        *,
        trials: int = 100,
        accept: bool = False,
        comment: str | None = None,
        **kwargs: Any,
    ) -> FactorBlock:
        """Re-optimize an accepted factor with all other factors fixed.

        Parameters
        ----------
        factor:
            Accepted raw factor to refine.
        trials:
            Number of Optuna trials for the refinement.
        accept:
            Whether to immediately accept the refined spec.
        comment:
            Optional audit note stored with the proposed or accepted refinement.
        **kwargs:
            Additional arguments forwarded to :meth:`FactorBlock.optimize`.

        Returns
        -------
        FactorBlock
            Factor block containing the refined proposal.
        """

        if factor not in self.specs:
            raise KeyError(f"Factor {factor!r} is not accepted yet.")
        block = self.factor(factor)
        block.optimize(trials=trials, **kwargs)
        block.compare()
        self._record("propose_refine", factor=factor, spec=block.spec, comment=comment)
        if accept:
            block.accept(comment=comment)
        return block

    def refine_all(
        self,
        *,
        trials: int = 50,
        accept: bool = False,
        **kwargs: Any,
    ) -> list[FactorBlock]:
        """Re-optimize every accepted main-effect factor.

        Parameters
        ----------
        trials:
            Number of Optuna trials per factor.
        accept:
            Whether each refined spec should be accepted immediately.
        **kwargs:
            Additional arguments forwarded to :meth:`refine_factor`.

        Returns
        -------
        list[FactorBlock]
            Refined factor blocks in accepted-factor order.
        """

        blocks = []
        for factor in list(self.accepted_raw_factors):
            blocks.append(self.refine_factor(factor, trials=trials, accept=accept, **kwargs))
        return blocks

    def fit_main_effects(self) -> FittedGLM:
        """Fit the current accepted main-effects and interaction model.

        Returns
        -------
        FittedGLM
            Fitted model using all currently selected transformed factors.
        """

        self.require_split()
        train, validation, _ = self._transformed_frames(include_holdout=False)
        factors = self.selected_factors
        model = self.glm.fit(train, factors)
        self.current_model = model
        self.train_scored = self.glm.predict(train, model)
        self.validation_scored = self.glm.predict(validation, model)
        self.model_versions.append(self._model_version_row())
        self.validation_reports = self.validation_report()
        return model

    def validation_report(self, bins: int = 10) -> dict[str, pd.DataFrame]:
        """Return validation tables for the current accepted model.

        Parameters
        ----------
        bins:
            Number of prediction bands used in calibration and lift tables.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Overall, calibration, lift, by-factor, train-validation, and model
            version tables.
        """

        if self.current_model is None or self.validation_scored is None or self.train_scored is None:
            self.fit_main_effects()
        train_scored, validation_scored = self._require_scored_frames()
        report = make_validation_report(
            validation_scored,
            target=self.target,
            prediction=self.prediction,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            factors=self.selected_factors,
            bins=bins,
        )
        report["train_validation"] = train_validation_comparison(
            train_scored,
            validation_scored,
            target=self.target,
            prediction=self.prediction,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
        )
        report["model_versions"] = model_version_comparison(self.model_versions)
        self.validation_reports = report
        return report

    def holdout_report(self, bins: int = 10) -> dict[str, pd.DataFrame]:
        """Explicitly score holdout and return final-style reports.

        Parameters
        ----------
        bins:
            Number of prediction bands used in calibration and lift tables.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Holdout summary, calibration, lift, and by-factor tables.
        """

        self.require_split()
        if self.current_model is None:
            self.fit_main_effects()
        current_model = self._require_current_model()
        _, _, holdout = self._transformed_frames(include_holdout=True)
        if holdout is None:
            raise RuntimeError("Internal error: holdout frame should exist after require_split().")
        self.holdout_scored = self.glm.predict(holdout, current_model)
        self.holdout_reports = holdout_final_report(
            self.holdout_scored,
            target=self.target,
            prediction=self.prediction,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            factors=self.selected_factors,
            bins=bins,
        )
        return self.holdout_reports

    def find_interactions(self, min_bin_size: float | None = None) -> pd.DataFrame:
        """Find candidate interactions among accepted transformed factors.

        Parameters
        ----------
        min_bin_size:
            Optional minimum combined-bin size required for interaction
            diagnostics. When omitted, the study default is used.

        Returns
        -------
        pandas.DataFrame
            Ranked pair diagnostics for accepted transformed factors.
        """

        if self.current_model is None or self.train_scored is None or self.validation_scored is None:
            self.fit_main_effects()
        train_scored, validation_scored = self._require_scored_frames()
        self.interaction_diagnostics = find_interaction_candidates(
            train_scored,
            validation_scored,
            factors=self.selected_factors,
            target=self.target,
            prediction=self.prediction,
            exposure=self.exposure,
            weight=self.weight,
            min_bin_size=min_bin_size if min_bin_size is not None else self.min_bin_size,
        )
        return self.interaction_diagnostics

    def test_interaction(self, factor_a: str, factor_b: str) -> pd.DataFrame:
        """Evaluate a coarse categorical interaction against the current model.

        Parameters
        ----------
        factor_a:
            Accepted raw factor name or selected transformed factor column.
        factor_b:
            Accepted raw factor name or selected transformed factor column.

        Returns
        -------
        pandas.DataFrame
            One-row comparison of current validation deviance against the
            candidate interaction model.
        """

        self.require_split()
        baseline = self._current_validation_deviance()
        spec = self._interaction_spec(factor_a, factor_b)
        train, validation, _ = self._transformed_frames(include_holdout=False)
        train = _apply_interaction_spec(train, spec)
        validation = _apply_interaction_spec(validation, spec)
        factors = [*self.selected_factors, str(spec["output"])]
        model = self.glm.fit(train, factors)
        train_scored = self.glm.predict(train, model)
        validation_scored = self.glm.predict(validation, model)
        result = pd.DataFrame(
            [
                {
                    "factor_a": spec["columns"][0],
                    "factor_b": spec["columns"][1],
                    "output": spec["output"],
                    "current_validation_deviance": baseline,
                    "interaction_train_deviance": self._scored_deviance(train_scored),
                    "interaction_validation_deviance": self._scored_deviance(validation_scored),
                    "validation_improvement": baseline - self._scored_deviance(validation_scored),
                }
            ]
        )
        self.pending_interactions[str(spec["output"])] = {"spec": spec, "result": result}
        return result

    def accept_interaction(
        self,
        factor_a: str,
        factor_b: str | None = None,
        *,
        comment: str | None = None,
    ) -> None:
        """Accept a previously tested interaction, or test and accept a pair.

        Parameters
        ----------
        factor_a:
            Interaction output name from a previous test, or the first factor
            in a pair to test and accept.
        factor_b:
            Optional second factor. When supplied, the pair is tested before
            acceptance.
        comment:
            Optional audit note stored in the study history.
        """

        if factor_b is None:
            output = factor_a
            if output not in self.pending_interactions:
                raise KeyError(f"Interaction {output!r} has not been tested.")
            payload = self.pending_interactions[output]
        else:
            result = self.test_interaction(factor_a, factor_b)
            output = str(result.iloc[0]["output"])
            payload = self.pending_interactions[output]
        spec = payload["spec"]
        before_score = self._current_validation_deviance()
        self.interaction_specs[str(spec["output"])] = spec
        after_score = self._current_validation_deviance()
        self._invalidate_model(include_holdout=True)
        self._record(
            "accept_interaction",
            factor=str(spec["output"]),
            spec=spec,
            score_before=before_score,
            score_after=after_score,
            comment=comment,
        )

    def auto_design(
        self,
        factors: list[str],
        *,
        top_n: int = 5,
        trials: int = 50,
        accept_interactions: bool = False,
        interaction_top_n: int = 3,
    ) -> GLMStudy:
        """Run a convenience automatic design workflow.

        Parameters
        ----------
        factors:
            Raw candidate factors to screen and optimize.
        top_n:
            Number of ranked factors to optimize and accept.
        trials:
            Number of Optuna trials per accepted factor.
        accept_interactions:
            Whether to test and accept top interaction candidates automatically.
        interaction_top_n:
            Number of interaction candidates to accept when
            ``accept_interactions`` is ``True``.

        Returns
        -------
        GLMStudy
            The same study instance after automatic design steps.
        """

        ranking = self.rank_candidates(factors)
        for factor in ranking.loc[ranking["error"].isna(), "factor"].head(top_n):
            block = self.factor(str(factor))
            block.optimize(trials=trials)
            block.accept(comment="auto_design")
        self.fit_main_effects()
        if accept_interactions:
            interactions = self.find_interactions()
            for _, row in interactions.head(interaction_top_n).iterrows():
                self.accept_interaction(str(row["factor_a"]), str(row["factor_b"]), comment="auto_design")
            self.fit_main_effects()
        return self

    def finalize(self, bins: int = 10) -> dict[str, pd.DataFrame]:
        """Fit the current model and score holdout as the final validation step.

        Parameters
        ----------
        bins:
            Number of prediction bands used in holdout calibration and lift
            tables.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Final holdout report tables.
        """

        self.fit_main_effects()
        report = self.holdout_report(bins=bins)
        holdout_deviance = np.nan
        if self.holdout_scored is not None:
            holdout_deviance = self._scored_deviance(self.holdout_scored)
        self.finalized = True
        self._record("finalize", score_after=holdout_deviance, comment="holdout scored")
        return report

    def save(self, output_dir: str | Path = "runs") -> Path:
        """Save study artifacts through :class:`RunLogger`.

        Parameters
        ----------
        output_dir:
            Parent directory where a timestamped study run folder is created.

        Returns
        -------
        pathlib.Path
            Path to the created run directory.
        """

        logger = RunLogger(output_dir, name="glm_study")
        logger.log_params(
            {
                "target": self.target,
                "family": self.family,
                "exposure": self.exposure,
                "weight": self.weight,
                "prediction": self.prediction,
                "accepted_raw_factors": self.accepted_raw_factors,
                "selected_factors": self.selected_factors,
                "finalized": self.finalized,
            }
        )
        logger.log_json("specs.json", self.specs)
        logger.log_json("interaction_specs.json", self.interaction_specs)
        logger.log_json("history.json", self.history)
        if self.ranking is not None:
            logger.log_frame("factor_ranking.csv", self.ranking.drop(columns=["spec"], errors="ignore"))
            logger.log_json(
                "factor_ranking_specs.json",
                {row["factor"]: row["spec"] for _, row in self.ranking.dropna(subset=["spec"]).iterrows()},
            )
        for factor, result in self.optimizations.items():
            logger.log_frame(f"trials/{factor}_trials.csv", result.trials)
        logger.log_frame("model_versions.csv", model_version_comparison(self.model_versions))
        if self.current_model is not None:
            logger.log_frame("coefficients.csv", self.current_model.coefficients())
        if self.validation_reports is not None:
            logger.log_report("validation", self.validation_reports)
        if self.holdout_reports is not None:
            logger.log_report("holdout", self.holdout_reports)
        if self.interaction_diagnostics is not None:
            logger.log_frame("interaction_diagnostics.csv", self.interaction_diagnostics)
        logger.close()
        return logger.path

    def require_split(self) -> None:
        """Raise if the study has not been split yet.

        Raises
        ------
        ValueError
            If train, validation, or holdout samples are missing.
        """

        if self.train is None or self.validation is None or self.holdout is None:
            raise ValueError("Call study.split(...) before this operation.")

    def _require_current_model(self) -> FittedGLM:
        current_model = self.current_model
        if current_model is None:
            raise RuntimeError("Call study.fit_main_effects() before this operation.")
        return current_model

    def _require_scored_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_scored = self.train_scored
        validation_scored = self.validation_scored
        if train_scored is None or validation_scored is None:
            raise RuntimeError("Call study.fit_main_effects() before this operation.")
        return train_scored, validation_scored

    def _require_train_validation_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.require_split()
        train = self.train
        validation = self.validation
        if train is None or validation is None:
            raise RuntimeError("Internal error: train and validation splits should exist after require_split().")
        return train, validation

    def sample_frame(self, sample: str) -> pd.DataFrame:
        """Return one raw split by name.

        Parameters
        ----------
        sample:
            Split name: ``"train"``, ``"validation"``, or ``"holdout"``.

        Returns
        -------
        pandas.DataFrame
            Requested raw split.
        """

        train, validation = self._require_train_validation_frames()
        if sample == "train":
            return train
        if sample == "validation":
            return validation
        if sample == "holdout":
            holdout = self.holdout
            if holdout is None:
                raise RuntimeError("Internal error: holdout split should exist after require_split().")
            return holdout
        raise ValueError("sample must be 'train', 'validation', or 'holdout'.")

    def _current_validation_deviance(self) -> float:
        self.require_split()
        train, validation, _ = self._transformed_frames(include_holdout=False)
        model = self.glm.fit(train, self.selected_factors)
        scored = self.glm.predict(validation, model)
        return self._scored_deviance(scored)

    def _model_version_row(self) -> dict[str, Any]:
        train_scored, validation_scored = self._require_scored_frames()
        return {
            "version": len(self.model_versions) + 1,
            "factors": list(self.selected_factors),
            "train_deviance": self._scored_deviance(train_scored),
            "validation_deviance": self._scored_deviance(validation_scored),
            "validation_mae": weighted_mae(
                validation_scored[self.target],
                validation_scored[self.prediction],
                validation_scored[self.weight] if self.weight else None,
            ),
            "validation_rmse": weighted_rmse(
                validation_scored[self.target],
                validation_scored[self.prediction],
                validation_scored[self.weight] if self.weight else None,
            ),
        }

    def _scored_deviance(self, scored: pd.DataFrame) -> float:
        return model_deviance(
            scored[self.target],
            scored[self.prediction],
            family=self.family,
            weight=scored[self.weight] if self.weight else None,
        )

    def _transformed_frames(
        self,
        *,
        exclude_factor: str | None = None,
        include_holdout: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        train_frame, validation_frame = self._require_train_validation_frames()
        train = train_frame.copy()
        validation = validation_frame.copy()
        holdout = self.holdout.copy() if include_holdout and self.holdout is not None else None

        excluded_output = str(self.specs[exclude_factor]["output"]) if exclude_factor in self.specs else None
        for factor in self.accepted_raw_factors:
            if factor == exclude_factor:
                continue
            spec = self.specs[factor]
            train = apply_spec(train, spec)
            validation = apply_spec(validation, spec)
            if holdout is not None:
                holdout = apply_spec(holdout, spec)

        for spec in self.interaction_specs.values():
            if excluded_output is not None and excluded_output in spec["columns"]:
                continue
            train = _apply_interaction_spec(train, spec)
            validation = _apply_interaction_spec(validation, spec)
            if holdout is not None:
                holdout = _apply_interaction_spec(holdout, spec)

        return train, validation, holdout

    def _fixed_factor_outputs(self, exclude_factor: str | None = None) -> list[str]:
        excluded_output = str(self.specs[exclude_factor]["output"]) if exclude_factor in self.specs else None
        outputs = [
            str(self.specs[factor]["output"])
            for factor in self.accepted_raw_factors
            if factor != exclude_factor
        ]
        for spec in self.interaction_specs.values():
            if excluded_output is not None and excluded_output in spec["columns"]:
                continue
            outputs.append(str(spec["output"]))
        return outputs

    def _interaction_spec(self, factor_a: str, factor_b: str) -> JsonDict:
        column_a = self._resolve_model_column(factor_a)
        column_b = self._resolve_model_column(factor_b)
        output = f"{_slug_column(column_a)}__x__{_slug_column(column_b)}"
        return {
            "type": "interaction",
            "columns": [column_a, column_b],
            "output": output,
            "method": "string_cross",
        }

    def _resolve_model_column(self, factor: str) -> str:
        if factor in self.specs:
            return str(self.specs[factor]["output"])
        if factor in self.selected_factors:
            return factor
        if factor in self.interaction_specs:
            return str(self.interaction_specs[factor]["output"])
        raise KeyError(f"Factor {factor!r} is not accepted in the current model.")

    def _invalidate_model(self, *, include_holdout: bool = False) -> None:
        self.current_model = None
        self.train_scored = None
        self.validation_scored = None
        self.validation_reports = None
        if include_holdout:
            self.holdout_scored = None
            self.holdout_reports = None
            self.finalized = False

    def _record(
        self,
        action: str,
        *,
        factor: str | None = None,
        spec: JsonDict | None = None,
        score_before: float | None = None,
        score_after: float | None = None,
        comment: str | None = None,
    ) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "factor": factor,
            "output": str(spec["output"]) if spec is not None and "output" in spec else None,
            "bins": len(spec.get("labels", [])) if spec is not None else None,
            "score_before": score_before,
            "score_after": score_after,
            "comment": comment,
            "spec": spec,
        }
        self.history.append(row)


def _factor_parts(
    factor: str | FactorBlock,
    spec: JsonDict | None,
) -> tuple[str, JsonDict, OptimizationResult | None]:
    if isinstance(factor, FactorBlock):
        if factor.spec is None:
            raise ValueError("FactorBlock has no proposed spec.")
        return factor.factor, factor.spec, factor.optimization
    if spec is None:
        raise ValueError("Provide spec when accepting by factor name.")
    return factor, spec, None


def _spec_bin_table(
    df: pd.DataFrame,
    *,
    output: str,
    target: str,
    exposure: str | None,
    weight: str | None,
) -> pd.DataFrame:
    if exposure is not None:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(exposure, "sum"), actual=(target, "sum"), rows=(target, "size"))
            .reset_index()
        )
    elif weight is not None:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(weight, "sum"), actual=(target, "sum"), rows=(target, "size"))
            .reset_index()
        )
    else:
        table = (
            df.groupby(output, dropna=False)
            .agg(bin_size=(target, "size"), actual=(target, "sum"), rows=(target, "size"))
            .reset_index()
        )
    return table.rename(columns={output: "bin"})


def _apply_interaction_spec(df: pd.DataFrame, spec: JsonDict) -> pd.DataFrame:
    columns = [str(column) for column in spec["columns"]]
    output = str(spec["output"])
    result = df.copy()
    left = result[columns[0]].astype("string").fillna("missing")
    right = result[columns[1]].astype("string").fillna("missing")
    result[output] = left + "|" + right
    return result


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _slug_column(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "interaction"
