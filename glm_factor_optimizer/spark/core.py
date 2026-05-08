"""Manual Spark facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bins import apply_spec, make_numeric_bins
from .metrics import calibration, summary
from .model import FittedSparkGLM, fit_glm
from .optimize import PenaltyInput, SparkOptimizationResult, optimize_factor


@dataclass(slots=True)
class SparkGLM:
    """Small Spark helper mirroring the pandas ``GLM`` facade.

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
    """

    target: str
    family: str = "poisson"
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "prediction"
    link: str | None = None

    def fit(self, df: Any, factors: list[str]) -> FittedSparkGLM:
        """Fit a Spark ML GLR pipeline.

        Parameters
        ----------
        df:
            Spark dataframe containing target and factor columns.
        factors:
            Factor columns to include in the model.

        Returns
        -------
        FittedSparkGLM
            Fitted Spark pipeline model and metadata.
        """

        return fit_glm(
            df,
            target=self.target,
            factors=factors,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            prediction=self.prediction,
            link=self.link,
        )

    def predict(self, df: Any, model: FittedSparkGLM) -> Any:
        """Return a Spark dataframe with predictions attached.

        Parameters
        ----------
        df:
            Spark dataframe to score.
        model:
            Fitted Spark GLM returned by :meth:`fit`.

        Returns
        -------
        pyspark.sql.DataFrame
            Scored Spark dataframe.
        """

        return model.transform(df)

    def report(self, df: Any, bins: int = 10) -> dict[str, Any]:
        """Return Spark summary and calibration dataframes.

        Parameters
        ----------
        df:
            Scored Spark dataframe.
        bins:
            Number of prediction bands used by calibration.

        Returns
        -------
        dict[str, pyspark.sql.DataFrame]
            Summary and calibration tables.
        """

        return {
            "summary": summary(
                df,
                target=self.target,
                prediction=self.prediction,
                exposure=self.exposure,
                family=self.family,
                weight=self.weight,
            ),
            "calibration": calibration(
                df,
                target=self.target,
                prediction=self.prediction,
                exposure=self.exposure,
                bins=bins,
            ),
        }

    def bins(self, df: Any, factor: str, bins: int = 10) -> dict:
        """Create a numeric binning spec from Spark data.

        Parameters
        ----------
        df:
            Spark dataframe containing the numeric factor.
        factor:
            Numeric column to bin.
        bins:
            Number of approximate quantile bins to request.

        Returns
        -------
        dict
            JSON-serializable numeric binning spec.
        """

        return make_numeric_bins(df, factor, bins=bins)

    def apply(self, df: Any, spec: dict) -> Any:
        """Apply a saved binning or grouping spec to Spark data.

        Parameters
        ----------
        df:
            Spark dataframe containing the raw factor referenced by ``spec``.
        spec:
            JSON-serializable numeric or categorical spec.

        Returns
        -------
        pyspark.sql.DataFrame
            Spark dataframe with the transformed output column added.
        """

        return apply_spec(df, spec)

    def optimize(
        self,
        train_df: Any,
        validation_df: Any,
        factor: str,
        *,
        kind: str = "numeric",
        fixed: list[str] | None = None,
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
        """Optimize one Spark factor using Optuna on the driver.

        Parameters
        ----------
        train_df:
            Spark training dataframe.
        validation_df:
            Spark validation dataframe.
        factor:
            Raw factor column to optimize.
        kind:
            Factor kind, either ``"numeric"`` or ``"categorical"``.
        fixed:
            Already transformed model columns to keep in the GLM.
        trials:
            Number of Optuna trials.
        max_bins:
            Maximum number of bins or groups allowed.
        n_prebins:
            Number of numeric pre-bins used to define candidate cutpoints.
        min_bin_size:
            Minimum bin size used by the small-bin penalty.
        bin_penalty:
            Penalty multiplier for additional bins.
        small_bin_penalty:
            Penalty multiplier for bins smaller than ``min_bin_size``.
        penalties:
            Optional extra penalty callable or named penalty mapping.
        seed:
            Optional Optuna sampler seed.
        cache_input:
            Whether to cache train and validation Spark dataframes during
            optimization.
        cache_trials:
            Whether to cache transformed trial dataframes.

        Returns
        -------
        SparkOptimizationResult
            Best spec, objective value, and trial history.
        """

        return optimize_factor(
            train_df,
            validation_df,
            target=self.target,
            factor=factor,
            family=self.family,
            exposure=self.exposure,
            kind=kind,
            fixed=fixed,
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
            cache_input=cache_input,
            cache_trials=cache_trials,
        )
