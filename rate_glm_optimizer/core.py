"""Manual-first facade for GLM modeling and factor optimization."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .bins import apply_spec, make_numeric_bins
from .metrics import calibration, lift_table, summary
from .model import FittedGLM, fit_glm
from .optimize import OptimizationResult, PenaltyInput, optimize_factor
from .screening import rank_factors


@dataclass(slots=True)
class GLM:
    """Small helper for GLM modeling with optional factor optimization."""

    target: str
    family: str = "poisson"
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "predicted"

    def fit(self, df: pd.DataFrame, factors: list[str]) -> FittedGLM:
        """Fit a GLM using the configured target, family, and optional exposure."""

        return fit_glm(
            df,
            target=self.target,
            family=self.family,
            exposure=self.exposure,
            factors=factors,
            weight=self.weight,
            prediction=self.prediction,
        )

    def predict(self, df: pd.DataFrame, model: FittedGLM) -> pd.DataFrame:
        """Return a copy of ``df`` with predictions attached."""

        result = df.copy()
        result[self.prediction] = model.predict(result)
        return result

    def report(self, df: pd.DataFrame, bins: int = 10) -> dict[str, pd.DataFrame]:
        """Return overall, calibration, and lift tables for scored data."""

        return {
            "summary": summary(
                df,
                target=self.target,
                prediction=self.prediction,
                exposure=self.exposure,
                weight=self.weight,
                family=self.family,
            ),
            "calibration": calibration(
                df,
                target=self.target,
                prediction=self.prediction,
                exposure=self.exposure,
                bins=bins,
            ),
            "lift": lift_table(
                df,
                target=self.target,
                prediction=self.prediction,
                exposure=self.exposure,
                bins=bins,
            ),
        }

    def bins(self, df: pd.DataFrame, factor: str, bins: int = 10, method: str = "quantile") -> dict:
        """Create a numeric binning spec from training data."""

        return make_numeric_bins(df[factor], bins=bins, column=factor, method=method)

    def apply(self, df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        """Apply a saved binning or grouping spec."""

        return apply_spec(df, spec)

    def optimize(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        factor: str,
        *,
        kind: str = "numeric",
        fixed: list[str] | None = None,
        fixed_factors: list[str] | None = None,
        trials: int = 50,
        max_bins: int = 8,
        n_prebins: int = 10,
        min_bin_size: float = 100.0,
        bin_penalty: float = 0.001,
        small_bin_penalty: float = 0.01,
        penalties: PenaltyInput | None = None,
        seed: int | None = 42,
    ) -> OptimizationResult:
        """Optimize one factor while keeping optional fixed factors unchanged."""

        return optimize_factor(
            train_df,
            validation_df,
            target=self.target,
            exposure=self.exposure,
            family=self.family,
            factor=factor,
            kind=kind,
            fixed=fixed,
            fixed_factors=fixed_factors,
            weight=self.weight,
            prediction=self.prediction,
            trials=trials,
            max_bins=max_bins,
            n_prebins=n_prebins,
            min_bin_size=min_bin_size,
            bin_penalty=bin_penalty,
            small_bin_penalty=small_bin_penalty,
            penalties=penalties,
            seed=seed,
        )

    def rank(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        factors: list[str],
        *,
        factor_kinds: dict[str, str] | None = None,
        bins: int = 5,
        max_groups: int = 6,
        min_bin_size: float = 1.0,
    ) -> pd.DataFrame:
        """Rank candidate factors by simple validation deviance improvement."""

        return rank_factors(
            train_df,
            validation_df,
            target=self.target,
            factors=factors,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            factor_kinds=factor_kinds,
            bins=bins,
            max_groups=max_groups,
            prediction=self.prediction,
            min_bin_size=min_bin_size,
        )


class RateGLM(GLM):
    """Convenience Poisson count-rate GLM with a required exposure column."""

    def __init__(
        self,
        target: str,
        exposure: str,
        weight: str | None = None,
        prediction: str = "predicted_count",
    ) -> None:
        super().__init__(
            target=target,
            family="poisson",
            exposure=exposure,
            weight=weight,
            prediction=prediction,
        )
