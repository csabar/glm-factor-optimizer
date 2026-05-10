"""Spark-backed interactive GLM study workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..runs import RunLogger
from ..validation import model_version_comparison
from ._deps import require_pyspark
from .bins import apply_spec
from .core import SparkGLM
from .diagnostics import find_interactions as find_interaction_candidates
from .factor import SparkFactorBlock
from .model import FittedSparkGLM
from .optimize import PenaltyInput, SparkOptimizationResult, optimize_factor
from .screening import bin_table_for_spec, infer_kind, rank_factors
from .split import split as split_frame
from .validation import (
    holdout_final_report,
    scored_metrics,
    train_validation_comparison,
    validation_report as make_validation_report,
)

JsonDict = dict[str, Any]


class SparkGLMStudy:
    """Stateful notebook workbench for iterative Spark GLM model design."""

    def __init__(
        self,
        df: Any,
        *,
        target: str,
        family: str = "poisson",
        exposure: str | None = None,
        weight: str | None = None,
        prediction: str = "prediction",
        factor_kinds: dict[str, str] | None = None,
        min_bin_size: float = 100.0,
        seed: int | None = 42,
    ) -> None:
        self.df = df
        self.target = target
        self.family = family
        self.exposure = exposure
        self.weight = weight
        self.prediction = prediction
        self.factor_kinds = dict(factor_kinds or {})
        self.min_bin_size = min_bin_size
        self.seed = seed
        self.glm = SparkGLM(
            target=target,
            family=family,
            exposure=exposure,
            weight=weight,
            prediction=prediction,
        )

        self.train: Any | None = None
        self.validation: Any | None = None
        self.holdout: Any | None = None

        self.specs: dict[str, JsonDict] = {}
        self.optimizations: dict[str, SparkOptimizationResult] = {}
        self.accepted_raw_factors: list[str] = []
        self.rejected_factors: list[str] = []
        self.interaction_specs: dict[str, JsonDict] = {}
        self.pending_interactions: dict[str, dict[str, Any]] = {}

        self.ranking: pd.DataFrame | None = None
        self.interaction_diagnostics: pd.DataFrame | None = None
        self.history: list[dict[str, Any]] = []
        self.model_versions: list[dict[str, Any]] = []

        self.current_model: FittedSparkGLM | None = None
        self.train_scored: Any | None = None
        self.validation_scored: Any | None = None
        self.holdout_scored: Any | None = None
        self.validation_reports: dict[str, pd.DataFrame] | None = None
        self.holdout_reports: dict[str, pd.DataFrame] | None = None
        self.finalized = False
        self._revision = 0

    @property
    def selected_factors(self) -> list[str]:
        """Return accepted transformed model columns."""

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
    ) -> tuple[Any, Any, Any]:
        """Create Spark train, validation, and holdout samples."""

        self.train, self.validation, self.holdout = split_frame(
            self.df,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            holdout_fraction=holdout_fraction,
            seed=seed if seed is not None else self.seed,
            time=time,
            time_split=time_split,
        )
        self._invalidate_model(include_holdout=True)
        self._bump_revision()
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
        """Rank Spark candidate factors for notebook screening."""

        self.require_split()
        train, validation = self._require_train_validation_frames()
        kinds = {**self.factor_kinds, **(factor_kinds or {})}
        self.ranking = rank_factors(
            train,
            validation,
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

    def factor(self, factor: str, kind: str | None = None) -> SparkFactorBlock:
        """Return an interactive Spark block for one raw factor."""

        frame = self.train if self.train is not None else self.df
        if factor not in frame.columns:
            raise KeyError(f"Column {factor!r} is missing.")
        resolved_kind = kind or self.factor_kinds.get(factor) or infer_kind(frame, factor)
        self.factor_kinds[factor] = resolved_kind
        block = SparkFactorBlock(self, factor=factor, kind=resolved_kind)
        if factor in self.specs:
            block.spec = dict(self.specs[factor])
            block.metadata["source"] = "accepted"
        return block

    def accept(
        self,
        factor: str | SparkFactorBlock,
        spec: JsonDict | None = None,
        *,
        comment: str | None = None,
    ) -> None:
        """Accept a factor block or explicit factor/spec into the Spark study."""

        self.require_split()
        factor_name, accepted_spec, optimization = _factor_parts(factor, spec)
        existed = factor_name in self.specs
        comparison_scores = self._comparison_scores_from_block(factor) if isinstance(factor, SparkFactorBlock) else None
        before_score = comparison_scores[0] if comparison_scores is not None else self._current_validation_deviance()
        self.specs[factor_name] = dict(accepted_spec)
        if isinstance(factor, SparkFactorBlock):
            self.factor_kinds[factor_name] = factor.kind
        if optimization is not None:
            self.optimizations[factor_name] = optimization
        if factor_name not in self.accepted_raw_factors:
            self.accepted_raw_factors.append(factor_name)
        after_score = comparison_scores[1] if comparison_scores is not None else self._current_validation_deviance()
        self._invalidate_model(include_holdout=True)
        self._bump_revision()
        self._record(
            "refine" if existed else "accept",
            factor=factor_name,
            spec=accepted_spec,
            score_before=before_score,
            score_after=after_score,
            comment=comment,
        )

    def reject(self, factor: str | SparkFactorBlock, *, comment: str | None = None) -> None:
        """Record a rejected Spark factor proposal."""

        factor_name = factor.factor if isinstance(factor, SparkFactorBlock) else factor
        if factor_name not in self.rejected_factors:
            self.rejected_factors.append(factor_name)
        spec = factor.spec if isinstance(factor, SparkFactorBlock) else None
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
        cache_input: bool = True,
        cache_trials: bool = False,
    ) -> SparkOptimizationResult:
        """Optimize one Spark factor against the current accepted model."""

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
            cache_input=cache_input,
            cache_trials=cache_trials,
        )

    def evaluate_candidate(
        self,
        factor: str,
        spec: JsonDict,
        *,
        return_scored: bool = False,
        _return_model: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, Any] | tuple[pd.DataFrame, Any, FittedSparkGLM]:
        """Compare a proposed Spark spec with the current accepted model."""

        self.require_split()
        baseline = self._current_validation_deviance()
        train, validation, _ = self._transformed_frames(exclude_factor=factor, include_holdout=False)
        train = apply_spec(train, spec)
        validation = apply_spec(validation, spec)
        candidate_factors = [*self._fixed_factor_outputs(exclude_factor=factor), str(spec["output"])]
        model = None
        release_model = True
        try:
            model = self.glm.fit(train, candidate_factors)
            train_scored = self.glm.predict(train, model)
            validation_scored = self.glm.predict(validation, model)
            train_deviance = self._scored_deviance(train_scored)
            validation_deviance = self._scored_deviance(validation_scored)
            table = bin_table_for_spec(
                train,
                output=str(spec["output"]),
                target=self.target,
                exposure=self.exposure,
                weight=self.weight,
                labels=list(spec.get("labels", [])),
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
                release_model = False
                if _return_model:
                    return comparison, validation_scored, model
                return comparison, validation_scored
            return comparison
        finally:
            if model is not None and release_model:
                model.release()

    def refine_factor(
        self,
        factor: str,
        *,
        trials: int = 100,
        accept: bool = False,
        comment: str | None = None,
        **kwargs: Any,
    ) -> SparkFactorBlock:
        """Re-optimize an accepted Spark factor with all other factors fixed."""

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
    ) -> list[SparkFactorBlock]:
        """Re-optimize every accepted Spark main-effect factor."""

        blocks = []
        for factor in list(self.accepted_raw_factors):
            blocks.append(self.refine_factor(factor, trials=trials, accept=accept, **kwargs))
        return blocks

    def fit_main_effects(self) -> FittedSparkGLM:
        """Fit the current accepted Spark model."""

        self.require_split()
        train, validation, _ = self._transformed_frames(include_holdout=False)
        factors = self.selected_factors
        previous_model = self.current_model
        model = self.glm.fit(train, factors)
        self.current_model = model
        if previous_model is not None:
            previous_model.release()
        self.train_scored = self.glm.predict(train, model)
        self.validation_scored = self.glm.predict(validation, model)
        self.model_versions.append(self._model_version_row())
        self.validation_reports = self.validation_report()
        return model

    def validation_report(self, bins: int = 10) -> dict[str, pd.DataFrame]:
        """Return validation tables for the current accepted Spark model."""

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
        """Explicitly score Spark holdout and return final-style reports."""

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
        """Find candidate interactions among accepted Spark transformed factors."""

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
        """Evaluate a coarse Spark categorical interaction."""

        self.require_split()
        baseline = self._current_validation_deviance()
        spec = self._interaction_spec(factor_a, factor_b)
        train, validation, _ = self._transformed_frames(include_holdout=False)
        train = _apply_interaction_spec(train, spec)
        validation = _apply_interaction_spec(validation, spec)
        factors = [*self.selected_factors, str(spec["output"])]
        model = None
        try:
            model = self.glm.fit(train, factors)
            train_scored = self.glm.predict(train, model)
            validation_scored = self.glm.predict(validation, model)
            train_deviance = self._scored_deviance(train_scored)
            validation_deviance = self._scored_deviance(validation_scored)
            result = pd.DataFrame(
                [
                    {
                        "factor_a": spec["columns"][0],
                        "factor_b": spec["columns"][1],
                        "output": spec["output"],
                        "current_validation_deviance": baseline,
                        "interaction_train_deviance": train_deviance,
                        "interaction_validation_deviance": validation_deviance,
                        "validation_improvement": baseline - validation_deviance,
                    }
                ]
            )
            self.pending_interactions[str(spec["output"])] = {"spec": spec, "result": result}
            return result
        finally:
            if model is not None:
                model.release()

    def accept_interaction(
        self,
        factor_a: str,
        factor_b: str | None = None,
        *,
        comment: str | None = None,
    ) -> None:
        """Accept a previously tested Spark interaction, or test and accept one."""

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
        self._bump_revision()
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
    ) -> SparkGLMStudy:
        """Run a convenience automatic Spark design workflow."""

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
        """Fit the current model and score Spark holdout."""

        if self.current_model is None or self.train_scored is None or self.validation_scored is None:
            self.fit_main_effects()
        report = self.holdout_report(bins=bins)
        holdout_deviance = np.nan
        if self.holdout_scored is not None:
            holdout_deviance = self._scored_deviance(self.holdout_scored)
        self.finalized = True
        self._record("finalize", score_after=holdout_deviance, comment="holdout scored")
        return report

    def save(self, output_dir: str | Path = "runs") -> Path:
        """Save Spark study metadata artifacts through :class:`RunLogger`."""

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
                "backend": "spark",
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
        if self.current_model is not None and hasattr(self.current_model, "coefficients"):
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
        """Raise if the Spark study has not been split yet."""

        if self.train is None or self.validation is None or self.holdout is None:
            raise ValueError("Call study.split(...) before this operation.")

    def sample_frame(self, sample: str) -> Any:
        """Return one raw Spark split by name."""

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

    def _require_current_model(self) -> FittedSparkGLM:
        current_model = self.current_model
        if current_model is None:
            raise RuntimeError("Call study.fit_main_effects() before this operation.")
        return current_model

    def _require_scored_frames(self) -> tuple[Any, Any]:
        train_scored = self.train_scored
        validation_scored = self.validation_scored
        if train_scored is None or validation_scored is None:
            raise RuntimeError("Call study.fit_main_effects() before this operation.")
        return train_scored, validation_scored

    def _require_train_validation_frames(self) -> tuple[Any, Any]:
        self.require_split()
        train = self.train
        validation = self.validation
        if train is None or validation is None:
            raise RuntimeError("Internal error: train and validation splits should exist after require_split().")
        return train, validation

    def _comparison_scores_from_block(self, block: SparkFactorBlock) -> tuple[float, float] | None:
        comparison = block.last_comparison
        if comparison is None or len(comparison) != 1:
            return None
        if block.last_comparison_revision != self._revision:
            return None
        if block.last_comparison_spec_fingerprint != self._spec_fingerprint(block.spec):
            return None
        row = comparison.iloc[0]
        return float(row["current_validation_deviance"]), float(row["candidate_validation_deviance"])

    def _spec_fingerprint(self, spec: JsonDict | None) -> str | None:
        if spec is None:
            return None
        return json.dumps(spec, sort_keys=True, default=str)

    def _bump_revision(self) -> None:
        self._revision += 1

    def _current_validation_deviance(self) -> float:
        self.require_split()
        train, validation, _ = self._transformed_frames(include_holdout=False)
        model = self.glm.fit(train, self.selected_factors)
        try:
            scored = self.glm.predict(validation, model)
            return self._scored_deviance(scored)
        finally:
            model.release()

    def _model_version_row(self) -> dict[str, Any]:
        train_scored, validation_scored = self._require_scored_frames()
        train_metrics = scored_metrics(
            train_scored,
            target=self.target,
            prediction=self.prediction,
            family=self.family,
            weight=self.weight,
        )
        validation_metrics = scored_metrics(
            validation_scored,
            target=self.target,
            prediction=self.prediction,
            family=self.family,
            weight=self.weight,
        )
        return {
            "version": len(self.model_versions) + 1,
            "factors": list(self.selected_factors),
            "train_deviance": train_metrics["deviance"],
            "validation_deviance": validation_metrics["deviance"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
        }

    def _scored_deviance(self, scored: Any) -> float:
        return scored_metrics(
            scored,
            target=self.target,
            prediction=self.prediction,
            family=self.family,
            weight=self.weight,
        )["deviance"]

    def _transformed_frames(
        self,
        *,
        exclude_factor: str | None = None,
        include_holdout: bool = False,
    ) -> tuple[Any, Any, Any | None]:
        train_frame, validation_frame = self._require_train_validation_frames()
        train = train_frame
        validation = validation_frame
        holdout = self.holdout if include_holdout and self.holdout is not None else None

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
        if self.current_model is not None:
            self.current_model.release()
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
    factor: str | SparkFactorBlock,
    spec: JsonDict | None,
) -> tuple[str, JsonDict, SparkOptimizationResult | None]:
    if isinstance(factor, SparkFactorBlock):
        if factor.spec is None:
            raise ValueError("SparkFactorBlock has no proposed spec.")
        return factor.factor, factor.spec, factor.optimization
    if spec is None:
        raise ValueError("Provide spec when accepting by factor name.")
    return factor, spec, None


def _apply_interaction_spec(df: Any, spec: JsonDict) -> Any:
    spark = require_pyspark()
    F = spark.functions
    columns = [str(column) for column in spec["columns"]]
    output = str(spec["output"])
    left = F.coalesce(F.col(columns[0]).cast("string"), F.lit("missing"))
    right = F.coalesce(F.col(columns[1]).cast("string"), F.lit("missing"))
    return df.withColumn(output, F.concat(left, F.lit("|"), right))


def _slug_column(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "interaction"
