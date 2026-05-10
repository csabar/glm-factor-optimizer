"""Manual-first facade for GLM modeling and factor optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._backend import is_spark_dataframe
from .bins import apply_spec, make_numeric_bins
from .metrics import calibration, lift_table, summary
from .model import fit_glm
from .optimize import PenaltyInput, optimize_factor
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

    def fit(self, df: Any, factors: list[str]) -> Any:
        """Fit a GLM using the configured outcome settings.

        Parameters
        ----------
        df:
            Pandas or Spark training data containing the target and factor
            columns.
        factors:
            Model factor columns to include in the design matrix.

        Returns
        -------
        object
            Fitted model object that can score compatible data.
        """

        if is_spark_dataframe(df):
            from .spark.model import fit_glm as spark_fit_glm

            return spark_fit_glm(
                df,
                target=self.target,
                family=self.family,
                exposure=self.exposure,
                factors=factors,
                weight=self.weight,
                prediction=self.prediction,
            )

        return fit_glm(
            df,
            target=self.target,
            family=self.family,
            exposure=self.exposure,
            factors=factors,
            weight=self.weight,
            prediction=self.prediction,
        )

    def predict(self, df: Any, model: Any) -> Any:
        """Return data with predictions attached.

        Parameters
        ----------
        df:
            Pandas or Spark data to score.
        model:
            Fitted GLM returned by :meth:`fit`.

        Returns
        -------
        object
            Scored pandas or Spark dataframe.
        """

        if is_spark_dataframe(df):
            return model.transform(df)

        result = df.copy()
        result[self.prediction] = model.predict(result)
        return result

    def report(self, df: Any, bins: int = 10) -> dict[str, Any]:
        """Return overall, calibration, and lift tables for scored data.

        Parameters
        ----------
        df:
            Scored pandas or Spark data containing target and prediction
            columns.
        bins:
            Number of prediction bands used by calibration and lift tables.

        Returns
        -------
        dict[str, object]
            Validation report tables. Pandas reports include summary,
            calibration, and lift; Spark reports include summary and
            calibration.
        """

        if is_spark_dataframe(df):
            from .spark.metrics import calibration as spark_calibration
            from .spark.metrics import summary as spark_summary

            return {
                "summary": spark_summary(
                    df,
                    target=self.target,
                    prediction=self.prediction,
                    exposure=self.exposure,
                    family=self.family,
                    weight=self.weight,
                ),
                "calibration": spark_calibration(
                    df,
                    target=self.target,
                    prediction=self.prediction,
                    exposure=self.exposure,
                    bins=bins,
                ),
            }

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

    def bins(self, df: Any, factor: str, bins: int = 10, method: str = "quantile") -> dict:
        """Create a numeric binning spec from training data.

        Parameters
        ----------
        df:
            Pandas or Spark data containing the numeric factor.
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

        if is_spark_dataframe(df):
            from .spark.bins import make_numeric_bins as spark_make_numeric_bins

            return spark_make_numeric_bins(df, factor, bins=bins, method=method)

        return make_numeric_bins(df[factor], bins=bins, column=factor, method=method)

    def apply(self, df: Any, spec: dict) -> Any:
        """Apply a saved binning or grouping spec.

        Parameters
        ----------
        df:
            Pandas or Spark data containing the raw factor referenced by
            ``spec``.
        spec:
            JSON-serializable numeric or categorical factor spec.

        Returns
        -------
        object
            Dataframe with the transformed output column added.
        """

        if is_spark_dataframe(df):
            from .spark.bins import apply_spec as spark_apply_spec

            return spark_apply_spec(df, spec)

        return apply_spec(df, spec)

    def optimize(
        self,
        train_df: Any,
        validation_df: Any,
        factor: str,
        *,
        kind: str = "numeric",
        fixed_factors: list[str] | None = None,
        trials: int = 50,
        max_bins: int = 8,
        n_prebins: int = 10,
        min_bin_size: float = 100.0,
        bin_penalty: float = 0.001,
        small_bin_penalty: float = 0.01,
        penalties: PenaltyInput | None = None,
        seed: int | None = 42,
    ) -> Any:
        """Optimize one factor while keeping optional fixed factors unchanged.

        Parameters
        ----------
        train_df:
            Pandas or Spark training data.
        validation_df:
            Pandas or Spark validation data used to score candidate specs.
        factor:
            Raw factor column to optimize.
        kind:
            Factor kind, either ``"numeric"`` or ``"categorical"``.
        fixed_factors:
            Already transformed model columns to keep in the GLM.
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
        object
            Best spec, objective value, and trial history.
        """

        fixed_factors_list = list(fixed_factors) if fixed_factors is not None else []

        if is_spark_dataframe(train_df):
            from .spark.optimize import optimize_factor as spark_optimize_factor

            return spark_optimize_factor(
                train_df,
                validation_df,
                target=self.target,
                exposure=self.exposure,
                family=self.family,
                factor=factor,
                kind=kind,
                fixed_factors=fixed_factors_list,
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

        return optimize_factor(
            train_df,
            validation_df,
            target=self.target,
            exposure=self.exposure,
            family=self.family,
            factor=factor,
            kind=kind,
            fixed_factors=fixed_factors_list,
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

        if is_spark_dataframe(train_df):
            from .spark.screening import rank_factors as spark_rank_factors

            return spark_rank_factors(
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
    """Convenience count/exposure GLM and optional factor-screening wrapper."""

    def __init__(
        self,
        target: str | None = None,
        exposure: str | None = None,
        weight: str | None = None,
        prediction: str = "predicted_count",
        *,
        candidate_factors: list[str] | None = None,
        target_col: str | None = None,
        exposure_col: str | None = None,
        weight_col: str | None = None,
        prediction_col: str | None = None,
        family: str = "poisson",
        factor_kinds: dict[str, str] | None = None,
        top_n: int | None = None,
        min_improvement: float = 0.0,
        seed: int | None = 42,
    ) -> None:
        """Create a count/exposure GLM helper.

        Parameters
        ----------
        target:
            Name of the observed outcome column.
        exposure:
            Optional positive exposure column used as a log offset.
        weight:
            Optional row-weight column.
        prediction:
            Name of the prediction column written by :meth:`predict`.
        candidate_factors:
            Optional raw factor columns to screen automatically when
            ``fit(df)`` is called without an explicit factor list.
        target_col:
            Alias for ``target`` for Spark-native notebook usage.
        exposure_col:
            Alias for ``exposure``.
        weight_col:
            Alias for ``weight``.
        prediction_col:
            Alias for ``prediction``.
        family:
            GLM family name. Defaults to ``"poisson"``.
        factor_kinds:
            Optional mapping from factor name to ``"numeric"`` or
            ``"categorical"`` for screening reports.
        top_n:
            Optional maximum number of positively ranked factors to select.
        min_improvement:
            Minimum validation deviance improvement required for automatic
            selection.
        seed:
            Random seed used when automatic fitting needs an internal
            validation split.
        """

        resolved_target = target or target_col
        if resolved_target is None:
            raise TypeError("Provide target or target_col.")
        resolved_exposure = exposure if exposure is not None else exposure_col
        resolved_weight = weight if weight is not None else weight_col
        resolved_prediction = prediction_col or prediction
        self.candidate_factors = list(candidate_factors or [])
        self.factor_kinds = dict(factor_kinds or {})
        self.top_n = top_n
        self.min_improvement = min_improvement
        self.seed = seed
        self.identified_factors_: Any | None = None
        self.selected_factors_: list[str] = []
        self.model_: Any | None = None
        self.train_scored_: Any | None = None
        self.validation_scored_: Any | None = None
        self.validation_report_: Any | None = None
        self._spark_auto_model: Any | None = None
        super().__init__(
            target=resolved_target,
            family=family,
            exposure=resolved_exposure,
            weight=resolved_weight,
            prediction=resolved_prediction,
        )

    def fit(
        self,
        df: Any,
        factors: list[str] | None = None,
        *,
        validation_df: Any | None = None,
        top_n: int | None = None,
        min_improvement: float | None = None,
    ) -> Any:
        """Fit a manual model or run candidate-factor screening.

        Parameters
        ----------
        df:
            Pandas or Spark dataframe.
        factors:
            Explicit model factors. When omitted and ``candidate_factors`` were
            configured, factors are screened automatically.
        validation_df:
            Optional validation dataframe for automatic screening.
        top_n:
            Optional per-call maximum number of selected factors.
        min_improvement:
            Optional per-call minimum validation deviance improvement.

        Returns
        -------
        object
            A fitted model for explicit-factor calls, or ``self`` for
            automatic candidate-factor screening.
        """

        if is_spark_dataframe(df):
            if factors is not None:
                return super().fit(df, factors)
            if not self.candidate_factors:
                raise TypeError("Provide factors or initialize RateGLM with candidate_factors.")
            from .spark.auto import RateGLM as SparkAutoRateGLM

            auto = SparkAutoRateGLM(
                candidate_factors=self.candidate_factors,
                target_col=self.target,
                family=self.family,
                exposure_col=self.exposure,
                weight_col=self.weight,
                prediction_col=self.prediction,
                factor_kinds=self.factor_kinds,
                top_n=self.top_n,
                min_improvement=self.min_improvement,
                seed=self.seed,
            ).fit(
                df,
                validation_df=validation_df,
                top_n=top_n,
                min_improvement=min_improvement,
            )
            self._spark_auto_model = auto
            self._copy_auto_state(auto)
            return self
        if factors is not None:
            return super().fit(df, factors)
        if not self.candidate_factors:
            raise TypeError("Provide factors or initialize RateGLM with candidate_factors.")
        return self._fit_pandas_auto(
            df,
            validation_df=validation_df,
            top_n=top_n,
            min_improvement=min_improvement,
        )

    def predict(self, df: Any, model: Any | None = None) -> Any:
        """Score data with either an explicit model or the automatic model.

        Parameters
        ----------
        df:
            Data to score.
        model:
            Optional explicit fitted model. When omitted after automatic
            candidate-factor screening, the stored model is used.

        Returns
        -------
        object
            Scored pandas or Spark dataframe.
        """

        if model is not None:
            return super().predict(df, model)
        if self._spark_auto_model is not None:
            return self._spark_auto_model.predict(df)
        if self.model_ is not None:
            return super().predict(df, self.model_)
        raise ValueError("Call fit(...) before predict(...), or pass an explicit model.")

    def _fit_pandas_auto(
        self,
        df: pd.DataFrame,
        *,
        validation_df: pd.DataFrame | None,
        top_n: int | None,
        min_improvement: float | None,
    ) -> RateGLM:
        from .split import split

        train_df = df
        if validation_df is None:
            train_df, validation_df, _ = split(df, train=0.8, validation=0.2, holdout=0.0, seed=self.seed or 42)
        ranking = rank_factors(
            train_df,
            validation_df,
            target=self.target,
            factors=self.candidate_factors,
            family=self.family,
            exposure=self.exposure,
            weight=self.weight,
            factor_kinds=self.factor_kinds,
            prediction=self.prediction,
        )
        threshold = self.min_improvement if min_improvement is None else float(min_improvement)
        limit = self.top_n if top_n is None else top_n
        selected = ranking.loc[
            ranking["error"].isna() & (ranking["deviance_improvement"] > threshold),
            "factor",
        ].tolist()
        if limit is not None:
            selected = selected[:limit]
        self.selected_factors_ = [str(factor) for factor in selected]
        self.identified_factors_ = ranking.assign(selected=ranking["factor"].isin(self.selected_factors_))
        self.model_ = super().fit(train_df, self.selected_factors_)
        self.train_scored_ = super().predict(train_df, self.model_)
        self.validation_scored_ = super().predict(validation_df, self.model_)
        self.validation_report_ = self.report(self.validation_scored_)
        return self

    def _copy_auto_state(self, auto: Any) -> None:
        self.identified_factors_ = auto.identified_factors_
        self.selected_factors_ = list(auto.selected_factors_)
        self.model_ = auto.model_
        self.train_scored_ = auto.train_scored_
        self.validation_scored_ = auto.validation_scored_
        self.validation_report_ = auto.validation_report_
