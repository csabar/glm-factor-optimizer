"""Lazy optional PySpark imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SparkDeps:
    """Container for lazily imported PySpark objects.

    Parameters
    ----------
    functions:
        ``pyspark.sql.functions`` module.
    types:
        ``pyspark.sql.types`` module.
    Window:
        Spark SQL window helper.
    Pipeline:
        Spark ML pipeline class.
    StringIndexer:
        Spark ML string indexer class.
    OneHotEncoder:
        Spark ML one-hot encoder class.
    VectorAssembler:
        Spark ML vector assembler class.
    GeneralizedLinearRegression:
        Spark ML generalized linear regression estimator class.
    """

    functions: Any
    types: Any
    Window: Any
    Pipeline: Any
    StringIndexer: Any
    OneHotEncoder: Any
    VectorAssembler: Any
    GeneralizedLinearRegression: Any


def require_pyspark() -> SparkDeps:
    """Import PySpark lazily and return the pieces used by this backend.

    Returns
    -------
    SparkDeps
        Container of PySpark modules and classes used by the Spark backend.
    """

    try:
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
        from pyspark.ml.regression import GeneralizedLinearRegression
        from pyspark.sql import Window, functions, types
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The Spark backend requires PySpark. Install with "
            "`pip install glm-factor-optimizer[spark]` or run inside an environment that already provides PySpark."
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
