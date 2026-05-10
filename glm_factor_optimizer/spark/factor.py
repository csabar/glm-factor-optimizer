"""Notebook-friendly Spark factor design blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from .bins import apply_spec, category_target_order, make_categorical_groups, make_numeric_bins
from .optimize import SparkOptimizationResult
from .screening import bin_table_for_spec, even_cutpoints
from .validation import by_factor_report

if TYPE_CHECKING:
    from .study import SparkGLMStudy

JsonDict = dict[str, Any]


@dataclass(slots=True)
class SparkFactorBlock:
    """Interactive workbench for one Spark candidate factor."""

    study: SparkGLMStudy
    factor: str
    kind: str
    spec: JsonDict | None = None
    optimization: SparkOptimizationResult | None = None
    target_order_table_: pd.DataFrame | None = None
    last_comparison: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> str | None:
        """Return the current proposed output column."""

        if self.spec is None:
            return None
        return str(self.spec["output"])

    def coarse_bins(self, bins: int = 10, method: str = "quantile") -> JsonDict:
        """Create a simple train-derived Spark spec for the factor."""

        self.study.require_split()
        train = self.study._require_train_validation_frames()[0]
        if self.kind == "numeric":
            self.spec = make_numeric_bins(train, self.factor, bins=bins, method=method)
        elif self.kind == "categorical":
            self.spec = make_categorical_groups(
                train,
                self.factor,
                self.study.target,
                exposure=self.study.exposure,
                weight=self.study.weight,
            )
        else:
            raise ValueError("kind must be 'numeric' or 'categorical'.")
        self.metadata["source"] = "coarse_bins"
        return self.spec

    def target_order(self, max_groups: int | None = None) -> pd.DataFrame:
        """Calculate categorical target order and optionally create a spec."""

        self.study.require_split()
        train = self.study._require_train_validation_frames()[0]
        self.target_order_table_ = category_target_order(
            train,
            self.factor,
            self.study.target,
            exposure=self.study.exposure,
            weight=self.study.weight,
        )
        if max_groups is not None:
            group_count = max(1, min(max_groups, len(self.target_order_table_)))
            cutpoints = even_cutpoints(len(self.target_order_table_), group_count)
            self.spec = make_categorical_groups(
                train,
                self.factor,
                self.study.target,
                exposure=self.study.exposure,
                weight=self.study.weight,
                cutpoints=cutpoints,
            )
            self.metadata["source"] = "target_order"
        return self.target_order_table_

    def set_spec(self, spec: JsonDict) -> SparkFactorBlock:
        """Set a manual JSON-serializable spec for this factor."""

        if str(spec["column"]) != self.factor:
            raise ValueError("Spec column must match the factor.")
        self.spec = dict(spec)
        self.metadata["source"] = "manual"
        return self

    def optimize(
        self,
        *,
        trials: int = 50,
        max_bins: int = 8,
        n_prebins: int = 10,
        min_bin_size: float | None = None,
        bin_penalty: float = 0.001,
        small_bin_penalty: float = 0.01,
        penalties: Any = None,
        seed: int | None = None,
        cache_input: bool = True,
        cache_trials: bool = False,
    ) -> SparkOptimizationResult:
        """Optimize this Spark factor while accepted factors stay fixed."""

        result = self.study.optimize_factor_block(
            self.factor,
            kind=self.kind,
            trials=trials,
            max_bins=max_bins,
            n_prebins=n_prebins,
            min_bin_size=min_bin_size,
            bin_penalty=bin_penalty,
            small_bin_penalty=small_bin_penalty,
            penalties=penalties,
            seed=seed,
            cache_input=cache_input,
            cache_trials=cache_trials,
        )
        self.optimization = result
        self.spec = result.spec
        self.metadata["source"] = "optuna"
        return result

    def bin_table(self, sample: str = "train") -> pd.DataFrame:
        """Return a grouped summary for the current proposed bins/groups."""

        self._require_spec()
        df = self.study.sample_frame(sample)
        transformed = apply_spec(df, self.spec)
        return bin_table_for_spec(
            transformed,
            output=str(self.spec["output"]),
            target=self.study.target,
            exposure=self.study.exposure,
            weight=self.study.weight,
            labels=list(self.spec.get("labels", [])),
        )

    def validation_table(self) -> pd.DataFrame:
        """Fit the candidate model and return validation diagnostics."""

        self._require_spec()
        comparison, scored, model = self.study.evaluate_candidate(
            self.factor,
            self.spec,
            return_scored=True,
            _return_model=True,
        )
        self.last_comparison = comparison
        try:
            return by_factor_report(
                scored,
                factor=str(self.spec["output"]),
                target=self.study.target,
                prediction=self.study.prediction,
                exposure=self.study.exposure,
                weight=self.study.weight,
            )
        finally:
            model.release()

    def compare(self) -> pd.DataFrame:
        """Compare the current accepted model against this candidate spec."""

        self._require_spec()
        comparison = self.study.evaluate_candidate(self.factor, self.spec)
        self.last_comparison = comparison
        return comparison

    def accept(self, comment: str | None = None) -> SparkFactorBlock:
        """Accept this proposed spec into the owning Spark study."""

        self._require_spec()
        self.study.accept(self, comment=comment)
        return self

    def reject(self, comment: str | None = None) -> SparkFactorBlock:
        """Record that this proposed factor/spec was rejected."""

        self.study.reject(self, comment=comment)
        return self

    def _require_spec(self) -> None:
        if self.spec is None:
            raise ValueError("Create, optimize, or set a spec before using this method.")
