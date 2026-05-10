"""Spark-native wrappers for factor screening and fitting."""

from __future__ import annotations

from typing import Any

from ._deps import require_pyspark
from .core import SparkGLM
from .metrics import model_deviance
from .model import FittedSparkGLM, fit_glm
from .split import split


class RateGLM:
    """Spark-native factor-screening GLM wrapper.

    Parameters
    ----------
    candidate_factors:
        Raw factor columns to screen and optionally include in the final model.
    target_col:
        Observed outcome column.
    family:
        Spark GLR family name.
    exposure_col:
        Optional exposure column used as a log offset.
    weight_col:
        Optional row-weight column.
    prediction_col:
        Prediction column written by the fitted Spark model.
    factor_kinds:
        Optional mapping from factor name to ``"numeric"`` or
        ``"categorical"`` for reporting. Spark model fitting still infers the
        actual feature handling from the dataframe schema.
    top_n:
        Optional maximum number of positively ranked factors to include.
    min_improvement:
        Minimum validation deviance improvement required for selection.
    seed:
        Random seed used when ``fit`` needs to create an internal validation
        split.
    """

    def __init__(
        self,
        *,
        candidate_factors: list[str],
        target_col: str,
        family: str = "poisson",
        exposure_col: str | None = None,
        weight_col: str | None = None,
        prediction_col: str = "prediction",
        factor_kinds: dict[str, str] | None = None,
        top_n: int | None = None,
        min_improvement: float = 0.0,
        seed: int | None = 42,
    ) -> None:
        self.candidate_factors = list(candidate_factors)
        self.target_col = target_col
        self.family = family
        self.exposure_col = exposure_col
        self.weight_col = weight_col
        self.prediction_col = prediction_col
        self.factor_kinds = dict(factor_kinds or {})
        self.top_n = top_n
        self.min_improvement = min_improvement
        self.seed = seed

        self.identified_factors_: Any | None = None
        self.selected_factors_: list[str] = []
        self.model_: FittedSparkGLM | None = None
        self.train_scored_: Any | None = None
        self.validation_scored_: Any | None = None
        self.validation_report_: dict[str, Any] | None = None

    def fit(
        self,
        train_df: Any,
        validation_df: Any | None = None,
        *,
        top_n: int | None = None,
        min_improvement: float | None = None,
    ) -> RateGLM:
        """Screen candidate factors and fit the selected Spark GLM.

        Parameters
        ----------
        train_df:
            Spark dataframe used for fitting.
        validation_df:
            Optional Spark dataframe used for factor screening. When omitted,
            ``train_df`` is split internally into train and validation samples.
        top_n:
            Optional per-call maximum number of selected factors.
        min_improvement:
            Optional per-call minimum validation deviance improvement.

        Returns
        -------
        RateGLM
            This fitted wrapper with ``identified_factors_`` and
            ``selected_factors_`` populated.
        """

        if validation_df is None:
            train_df, validation_df, _ = split(
                train_df,
                train=0.8,
                validation=0.2,
                holdout=0.0,
                seed=self.seed or 42,
            )

        required_columns = [
            self.target_col,
            *self.candidate_factors,
            *([self.exposure_col] if self.exposure_col is not None else []),
            *([self.weight_col] if self.weight_col is not None else []),
        ]
        _check_columns(train_df, required_columns)
        _check_columns(validation_df, required_columns)

        baseline_model = fit_glm(
            train_df,
            target=self.target_col,
            factors=[],
            family=self.family,
            exposure=self.exposure_col,
            weight=self.weight_col,
            prediction=self.prediction_col,
        )
        try:
            baseline_scored = baseline_model.transform(validation_df)
            baseline_deviance = model_deviance(
                baseline_scored,
                self.target_col,
                self.prediction_col,
                family=self.family,
                weight=self.weight_col,
            )
        finally:
            baseline_model.release()

        rows = []
        for factor in self.candidate_factors:
            rows.append(self._screen_factor(train_df, validation_df, factor, baseline_deviance))

        clean_rows = [row for row in rows if row["error"] is None]
        clean_rows.sort(key=lambda row: (-row["deviance_improvement"], row["factor"]))
        threshold = self.min_improvement if min_improvement is None else float(min_improvement)
        limit = self.top_n if top_n is None else top_n
        selected = [row["factor"] for row in clean_rows if row["deviance_improvement"] > threshold]
        if limit is not None:
            selected = selected[:limit]
        self.selected_factors_ = selected

        rank_by_factor = {row["factor"]: index + 1 for index, row in enumerate(clean_rows)}
        for row in rows:
            row["rank"] = rank_by_factor.get(row["factor"])
            row["selected"] = row["factor"] in selected
        self.identified_factors_ = _rows_to_spark(train_df, rows)

        previous_model = self.model_
        model = fit_glm(
            train_df,
            target=self.target_col,
            factors=selected,
            family=self.family,
            exposure=self.exposure_col,
            weight=self.weight_col,
            prediction=self.prediction_col,
        )
        self.model_ = model
        if previous_model is not None:
            previous_model.release()
        self.train_scored_ = self.model_.transform(train_df)
        self.validation_scored_ = self.model_.transform(validation_df)
        self.validation_report_ = SparkGLM(
            target=self.target_col,
            family=self.family,
            exposure=self.exposure_col,
            weight=self.weight_col,
            prediction=self.prediction_col,
        ).report(self.validation_scored_)
        return self

    def predict(self, df: Any) -> Any:
        """Return a Spark dataframe scored by the fitted model.

        Parameters
        ----------
        df:
            Spark dataframe to score.

        Returns
        -------
        pyspark.sql.DataFrame
            Spark dataframe with the prediction column attached.
        """

        if self.model_ is None:
            raise ValueError("Call fit(...) before predict(...).")
        return self.model_.transform(df)

    def _screen_factor(
        self,
        train_df: Any,
        validation_df: Any,
        factor: str,
        baseline_deviance: float,
    ) -> dict[str, Any]:
        kind = self.factor_kinds.get(factor) or _infer_kind(train_df, factor)
        model = None
        try:
            model = fit_glm(
                train_df,
                target=self.target_col,
                factors=[factor],
                family=self.family,
                exposure=self.exposure_col,
                weight=self.weight_col,
                prediction=self.prediction_col,
            )
            scored = model.transform(validation_df)
            validation_deviance = model_deviance(
                scored,
                self.target_col,
                self.prediction_col,
                family=self.family,
                weight=self.weight_col,
            )
            improvement = float(baseline_deviance - validation_deviance)
            return {
                "rank": None,
                "factor": factor,
                "kind": kind,
                "baseline_validation_deviance": float(baseline_deviance),
                "validation_deviance": float(validation_deviance),
                "deviance_improvement": improvement,
                "relative_improvement": improvement / max(abs(float(baseline_deviance)), 1e-9),
                "selected": False,
                "error": None,
            }
        except Exception as exc:
            return {
                "rank": None,
                "factor": factor,
                "kind": kind,
                "baseline_validation_deviance": float(baseline_deviance),
                "validation_deviance": None,
                "deviance_improvement": None,
                "relative_improvement": None,
                "selected": False,
                "error": repr(exc),
            }
        finally:
            if model is not None:
                model.release()


SparkRateGLM = RateGLM


def _rows_to_spark(df: Any, rows: list[dict[str, Any]]) -> Any:
    spark = require_pyspark()
    T = spark.types
    schema = T.StructType(
        [
            T.StructField("rank", T.IntegerType(), True),
            T.StructField("factor", T.StringType(), False),
            T.StructField("kind", T.StringType(), False),
            T.StructField("baseline_validation_deviance", T.DoubleType(), False),
            T.StructField("validation_deviance", T.DoubleType(), True),
            T.StructField("deviance_improvement", T.DoubleType(), True),
            T.StructField("relative_improvement", T.DoubleType(), True),
            T.StructField("selected", T.BooleanType(), False),
            T.StructField("error", T.StringType(), True),
        ]
    )
    ordered = sorted(rows, key=lambda row: (row["rank"] is None, row["rank"] or 10**9, row["factor"]))
    return df.sparkSession.createDataFrame(ordered, schema=schema)


def _check_columns(df: Any, columns: list[str]) -> None:
    existing = set(df.columns)
    missing = [column for column in columns if column not in existing]
    if missing:
        raise KeyError(f"Missing Spark dataframe columns: {', '.join(missing)}")


def _infer_kind(df: Any, factor: str) -> str:
    simple_type = df.schema[factor].dataType.simpleString().split("(")[0]
    if simple_type in {"byte", "short", "integer", "long", "float", "double", "decimal"}:
        return "numeric"
    return "categorical"
