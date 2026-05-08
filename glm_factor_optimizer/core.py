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
    """Small helper for GLM modeling with optional factor optimization.

    Parameters
    ----------
    target:
        Name of the observed outcome column.
    family:
        GLM family name, such as ``"poisson"``, ``"gamma"``, or
        ``"gaussian"``.
    exposure:
        Optional exposure column used as a log offset during fitting.
    weight:
        Optional row-weight column used during fitting and diagnostics.
    prediction:
        Name of the prediction column written by :meth:`predict`.
    """

    target: str
    family: str = "poisson"
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "predicted"

    def fit(self, df: pd.DataFrame, factors: list[str]) -> FittedGLM:
        """Fit a GLM using the configured outcome settings.

        Parameters
        ----------
        df:
            Training data containing the target and factor columns.
        factors:
            Model factor columns to include in the design matrix.

        Returns
        -------
        FittedGLM
            Fitted model object that can score compatible data.
        """

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
        """Return a copy of ``df`` with predictions attached.

        Parameters
        ----------
        df:
            Data to score.
        model:
            Fitted GLM returned by :meth:`fit`.

        Returns
        -------
        pandas.DataFrame
            Copy of ``df`` with the configured prediction column added.
        """

        result = df.copy()
        result[self.prediction] = model.predict(result)
        return result

    def report(self, df: pd.DataFrame, bins: int = 10) -> dict[str, pd.DataFrame]:
        """Return overall, calibration, and lift tables for scored data.

        Parameters
        ----------
        df:
            Scored data containing target and prediction columns.
        bins:
            Number of prediction bands used by calibration and lift tables.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Summary, calibration, and lift tables.
        """

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
        """Create a numeric binning spec from training data.

        Parameters
        ----------
        df:
            Data containing the numeric factor.
        factor:
            Numeric column to bin.
        bins:
            Number of bins to request.
        method:
            Binning method, currently ``"quantile"`` or ``"uniform"``.

        Returns
        -------
        dict
            JSON-serializable numeric binning spec.
        """

        return make_numeric_bins(df[factor], bins=bins, column=factor, method=method)

    def apply(self, df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        """Apply a saved binning or grouping spec.

        Parameters
        ----------
        df:
            Data containing the raw factor referenced by ``spec``.
        spec:
            JSON-serializable numeric or categorical factor spec.

        Returns
        -------
        pandas.DataFrame
            Copy of ``df`` with the transformed output column added.
        """

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
        """Optimize one factor while keeping optional fixed factors unchanged.

        Parameters
        ----------
        train_df:
            Training data.
        validation_df:
            Validation data used to score candidate specs.
        factor:
            Raw factor column to optimize.
        kind:
            Factor kind, either ``"numeric"`` or ``"categorical"``.
        fixed:
            Already transformed model columns to keep in the GLM.
        fixed_factors:
            Alias for ``fixed`` kept for readability in manual calls.
        trials:
            Number of Optuna trials.
        max_bins:
            Maximum number of bins or groups allowed.
        n_prebins:
            Number of numeric pre-bins used to define candidate cutpoints.
        min_bin_size:
            Minimum bin size used by the default small-bin penalty.
        bin_penalty:
            Penalty multiplier for additional bins.
        small_bin_penalty:
            Penalty multiplier for bins smaller than ``min_bin_size``.
        penalties:
            Optional extra penalty callable or named penalty mapping.
        seed:
            Optional Optuna sampler seed.

        Returns
        -------
        OptimizationResult
            Best spec, objective value, and trial history.
        """

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
        """Rank candidate factors by validation deviance improvement.

        Parameters
        ----------
        train_df:
            Training data used to derive simple screening specs.
        validation_df:
            Validation data used to compare candidate factors.
        factors:
            Raw candidate factor columns to screen.
        factor_kinds:
            Optional mapping from factor name to ``"numeric"`` or
            ``"categorical"``.
        bins:
            Number of simple numeric bins used during screening.
        max_groups:
            Maximum number of ordered categorical groups used during screening.
        min_bin_size:
            Minimum bin size reported by screening diagnostics.

        Returns
        -------
        pandas.DataFrame
            Ranked screening table.
        """

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
    """Convenience Poisson count model with a required exposure column."""

    def __init__(
        self,
        target: str,
        exposure: str,
        weight: str | None = None,
        prediction: str = "predicted_count",
    ) -> None:
        """Create a Poisson GLM helper with a log-exposure offset.

        Parameters
        ----------
        target:
            Name of the observed count outcome column.
        exposure:
            Positive exposure column used as a log offset.
        weight:
            Optional row-weight column.
        prediction:
            Name of the prediction column written by :meth:`predict`.
        """

        super().__init__(
            target=target,
            family="poisson",
            exposure=exposure,
            weight=weight,
            prediction=prediction,
        )
