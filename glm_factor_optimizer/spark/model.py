"""Spark ML GLM wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._deps import require_pyspark

_NUMERIC_TYPES = {
    "byte",
    "short",
    "integer",
    "long",
    "float",
    "double",
    "decimal",
}


@dataclass(slots=True)
class FittedSparkGLM:
    """Fitted Spark PipelineModel with GLM metadata.

    Parameters
    ----------
    pipeline_model:
        Fitted Spark ``PipelineModel``.
    target:
        Observed outcome column.
    family:
        Spark GLR family used for fitting.
    factors:
        Factor columns included in the model.
    exposure:
        Optional exposure column used as a log offset.
    weight:
        Optional row-weight column.
    prediction:
        Prediction column written by Spark GLR.
    offset_col:
        Internal offset column name used when exposure is supplied.
    constant_feature_col:
        Internal constant feature column used for intercept-only models.
    """

    pipeline_model: Any
    target: str
    family: str
    factors: list[str]
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "prediction"
    offset_col: str = "__offset"
    constant_feature_col: str | None = None

    def transform(self, df: Any) -> Any:
        """Return a Spark dataframe with predictions.

        Parameters
        ----------
        df:
            Spark dataframe containing the factor columns.

        Returns
        -------
        pyspark.sql.DataFrame
            Dataframe transformed by the fitted Spark pipeline.
        """

        prepared = _prepare_frame(df, self.exposure, self.offset_col, self.constant_feature_col)
        return self.pipeline_model.transform(prepared)

    def coefficients(self) -> pd.DataFrame:
        """Return fitted Spark GLR coefficients as a small pandas table."""

        glm_model = self.pipeline_model.stages[-1]
        raw_coefficients = getattr(glm_model, "coefficients", None)
        coefficients = list(raw_coefficients) if raw_coefficients is not None else []
        rows: list[dict[str, float | str]] = []
        start = 0
        if self.constant_feature_col is not None and coefficients:
            rows.append({"term": "intercept", "coefficient": float(coefficients[0])})
            start = 1
        else:
            rows.append({"term": "intercept", "coefficient": float(getattr(glm_model, "intercept", 0.0))})
        for index, value in enumerate(coefficients[start:], start=1):
            rows.append({"term": f"feature_{index}", "coefficient": float(value)})
        return pd.DataFrame(rows)

    def release(self) -> None:
        """Best-effort release of Spark Connect cached ML model references."""

        release_model_cache(self.pipeline_model)


def fit_glm(
    df: Any,
    target: str,
    factors: list[str],
    *,
    family: str = "poisson",
    exposure: str | None = None,
    weight: str | None = None,
    prediction: str = "prediction",
    link: str | None = None,
    max_iter: int = 25,
    reg_param: float = 0.0,
) -> FittedSparkGLM:
    """Fit a Spark ML generalized linear regression pipeline.

    Parameters
    ----------
    df:
        Spark dataframe used for fitting.
    target:
        Observed outcome column.
    factors:
        Factor columns to include in the model.
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
    max_iter:
        Maximum Spark GLR iterations.
    reg_param:
        Spark GLR regularization parameter.

    Returns
    -------
    FittedSparkGLM
        Fitted Spark pipeline model and metadata.
    """

    spark = require_pyspark()
    F = spark.functions
    stages: list[Any] = []
    feature_cols: list[str] = []
    constant_feature_col = "__glm_intercept_feature" if not factors else None

    for factor in factors:
        field = df.schema[factor]
        if _is_numeric_type(field.dataType.simpleString()):
            feature_cols.append(factor)
            continue
        indexed = f"{factor}__idx"
        encoded = f"{factor}__oh"
        stages.append(spark.StringIndexer(inputCol=factor, outputCol=indexed, handleInvalid="keep"))
        stages.append(spark.OneHotEncoder(inputCols=[indexed], outputCols=[encoded], handleInvalid="keep"))
        feature_cols.append(encoded)

    if constant_feature_col is not None:
        feature_cols.append(constant_feature_col)

    assembler = spark.VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    offset_col = "__offset"
    estimator_kwargs = {
        "featuresCol": "features",
        "labelCol": target,
        "predictionCol": prediction,
        "family": family.lower().strip(),
        "maxIter": max_iter,
        "regParam": reg_param,
    }
    if weight is not None:
        estimator_kwargs["weightCol"] = weight
    if exposure is not None:
        estimator_kwargs["offsetCol"] = offset_col
    if constant_feature_col is not None:
        estimator_kwargs["fitIntercept"] = False
    selected_link = link or _default_link(family)
    if selected_link is not None:
        estimator_kwargs["link"] = selected_link

    stages.append(spark.GeneralizedLinearRegression(**estimator_kwargs))
    pipeline = spark.Pipeline(stages=stages)
    prepared = _with_offset(df, exposure, offset_col)
    if constant_feature_col is not None:
        prepared = prepared.withColumn(constant_feature_col, F.lit(1.0))
    fitted = pipeline.fit(prepared)
    return FittedSparkGLM(
        pipeline_model=fitted,
        target=target,
        family=family.lower().strip(),
        exposure=exposure,
        factors=list(factors),
        weight=weight,
        prediction=prediction,
        offset_col=offset_col,
        constant_feature_col=constant_feature_col,
    )


def release_model_cache(model: Any) -> None:
    """Evict Spark Connect ML cache entries for a fitted model when possible."""

    _release_ml_object(model)
    for stage in getattr(model, "stages", []) or []:
        _release_ml_object(stage)


def _release_ml_object(obj: Any) -> None:
    ref = getattr(obj, "_java_obj", None)
    ref_id = getattr(ref, "ref_id", None)
    if not isinstance(ref_id, str):
        return
    try:
        from pyspark.ml.util import del_remote_cache

        del_remote_cache(ref_id)
    except Exception:
        pass


def _prepare_frame(
    df: Any,
    exposure: str | None,
    offset_col: str,
    constant_feature_col: str | None,
) -> Any:
    prepared = _with_offset(df, exposure, offset_col)
    if constant_feature_col is None:
        return prepared
    spark = require_pyspark()
    F = spark.functions
    return prepared.withColumn(constant_feature_col, F.lit(1.0))


def _with_offset(df: Any, exposure: str | None, offset_col: str) -> Any:
    if exposure is None:
        return df
    spark = require_pyspark()
    F = spark.functions
    return df.withColumn(offset_col, F.log(F.col(exposure).cast("double")))


def _default_link(family: str) -> str | None:
    normalized = family.lower().strip()
    if normalized in {"poisson", "gamma"}:
        return "log"
    if normalized == "gaussian":
        return "identity"
    return None


def _is_numeric_type(simple_type: str) -> bool:
    return simple_type.split("(")[0] in _NUMERIC_TYPES
