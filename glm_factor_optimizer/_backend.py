"""Private dataframe backend detection helpers."""

from __future__ import annotations


def is_spark_dataframe(df: object) -> bool:
    """Return whether ``df`` looks like a Spark DataFrame."""

    return hasattr(df, "randomSplit") and hasattr(df, "schema")
