"""Lazy optional PySpark imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SparkDeps:
    functions: Any
    types: Any
    Window: Any
    Pipeline: Any
    StringIndexer: Any
    OneHotEncoder: Any
    VectorAssembler: Any
    GeneralizedLinearRegression: Any


def require_pyspark() -> SparkDeps:
    """Import PySpark lazily and return the pieces used by this backend."""

    try:
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
        from pyspark.ml.regression import GeneralizedLinearRegression
        from pyspark.sql import Window, functions, types
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The Spark backend requires PySpark. Install with "
            "`pip install rate-glm-optimizer[spark]` or run inside Databricks."
        ) from exc

    return SparkDeps(
        functions=functions,
        types=types,
        Window=Window,
        Pipeline=Pipeline,
        StringIndexer=StringIndexer,
        OneHotEncoder=OneHotEncoder,
        VectorAssembler=VectorAssembler,
        GeneralizedLinearRegression=GeneralizedLinearRegression,
    )
