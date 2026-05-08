"""Spark ML GLM wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    """

    pipeline_model: Any
    target: str
    family: str
    factors: list[str]
    exposure: str | None = None
    weight: str | None = None
    prediction: str = "prediction"
    offset_col: str = "__offset"

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

        prepared = _with_offset(df, self.exposure, self.offset_col)
        return self.pipeline_model.transform(prepared)


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
    stages: list[Any] = []
    feature_cols: list[str] = []

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
    selected_link = link or _default_link(family)
    if selected_link is not None:
        estimator_kwargs["link"] = selected_link

    stages.append(spark.GeneralizedLinearRegression(**estimator_kwargs))
    pipeline = spark.Pipeline(stages=stages)
    prepared = _with_offset(df, exposure, offset_col)
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
    )


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
