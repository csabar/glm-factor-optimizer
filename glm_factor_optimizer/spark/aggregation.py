"""Spark aggregation helpers for model development tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._deps import require_pyspark

AggregationSpec = Mapping[str, tuple[str, str]]


def aggregate_table(
    df: Any,
    group_by: str | Sequence[str] | None = None,
    aggregations: AggregationSpec | None = None,
) -> Any:
    """Return grouped rows plus optional named Spark aggregations.

    Parameters
    ----------
    df:
        Spark dataframe to aggregate.
    group_by:
        Optional column or columns to group by. When omitted, a single overall
        row is returned.
    aggregations:
        Optional mapping from output column name to ``(source_column,
        function)``. Supported functions include ``sum``, ``mean``, ``min``,
        ``max``, ``count``, and ``std``.

    Returns
    -------
    pyspark.sql.DataFrame
        Grouped Spark dataframe with a ``rows`` column and requested
        aggregations.
    """

    spark = require_pyspark()
    F = spark.functions
    group_cols = _as_list(group_by)
    expressions = [F.count(F.lit(1)).alias("rows")]
    for name, (column, function) in (aggregations or {}).items():
        expressions.append(_agg(F, column, function).alias(name))
    if group_cols:
        return df.groupBy(*group_cols).agg(*expressions)
    return df.agg(*expressions)


def aggregate_rate_table(
    df: Any,
    group_by: str | Sequence[str] | None,
    *,
    target: str,
    exposure: str | None = None,
    weight: str | None = None,
    prediction: str | None = None,
    extras: AggregationSpec | None = None,
) -> Any:
    """Return grouped observed/predicted totals and level diagnostics.

    Parameters
    ----------
    df:
        Spark dataframe to aggregate.
    group_by:
        Optional column or columns to group by.
    target:
        Observed outcome column.
    exposure:
        Optional exposure column for exposure-adjusted level columns.
    weight:
        Optional weight column to total.
    prediction:
        Optional predicted outcome column.
    extras:
        Optional extra named aggregations, using the same format as
        :func:`aggregate_table`.

    Returns
    -------
    pyspark.sql.DataFrame
        Grouped observed and predicted totals with mean or exposure-adjusted
        diagnostics.
    """

    spark = require_pyspark()
    F = spark.functions
    aggregations: dict[str, tuple[str, str]] = {"actual": (target, "sum")}
    if exposure is not None:
        aggregations["exposure"] = (exposure, "sum")
    if weight is not None:
        aggregations["weight"] = (weight, "sum")
    if prediction is not None:
        aggregations["predicted"] = (prediction, "sum")
    aggregations.update(extras or {})
    table = aggregate_table(df, group_by, aggregations)
    if exposure is not None:
        table = table.withColumn("actual_rate", F.col("actual") / F.greatest(F.col("exposure"), F.lit(1e-9)))
        if prediction is not None:
            table = table.withColumn(
                "predicted_rate",
                F.col("predicted") / F.greatest(F.col("exposure"), F.lit(1e-9)),
            )
    else:
        table = table.withColumn("actual_mean", F.col("actual") / F.greatest(F.col("rows"), F.lit(1)))
        if prediction is not None:
            table = table.withColumn(
                "predicted_mean",
                F.col("predicted") / F.greatest(F.col("rows"), F.lit(1)),
            )
    if prediction is not None:
        table = table.withColumn(
            "actual_to_predicted",
            F.col("actual") / F.greatest(F.col("predicted"), F.lit(1e-9)),
        )
    return table


def _agg(functions: Any, column: str, function: str) -> Any:
    normalized = function.lower().strip()
    col = functions.col(column).cast("double")
    if normalized == "sum":
        return functions.sum(col)
    if normalized in {"mean", "avg"}:
        return functions.avg(col)
    if normalized == "min":
        return functions.min(col)
    if normalized == "max":
        return functions.max(col)
    if normalized == "count":
        return functions.count(functions.col(column))
    if normalized == "std":
        return functions.stddev(col)
    raise ValueError(f"Unsupported aggregation function {function!r}.")


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
