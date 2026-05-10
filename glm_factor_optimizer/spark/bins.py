"""Spark-native binning and grouping spec helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._deps import require_pyspark

JsonDict = dict[str, Any]

_MISSING = "__missing__"


def make_numeric_bins(
    df: Any,
    column: str,
    bins: int = 10,
    method: str = "quantile",
    relative_error: float = 0.001,
) -> JsonDict:
    """Create a JSON-serializable numeric binning spec from Spark data.

    Parameters
    ----------
    df:
        Spark dataframe containing the numeric column.
    column:
        Numeric column to bin.
    bins:
        Number of approximate quantile bins to request.
    method:
        Binning method. Spark currently supports ``"quantile"``.
    relative_error:
        Relative error passed to Spark ``approxQuantile``.

    Returns
    -------
    dict
        JSON-serializable numeric binning spec.
    """

    if bins < 1:
        raise ValueError("bins must be at least 1.")
    if method != "quantile":
        raise ValueError("Spark make_numeric_bins currently supports method='quantile'.")

    probabilities = [index / bins for index in range(bins + 1)]
    edges = [float(value) for value in df.approxQuantile(column, probabilities, relative_error)]
    if not edges:
        raise ValueError(f"Cannot create bins from empty column {column!r}.")
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    final_edges: list[float] = []
    for edge in edges:
        if not final_edges or edge > final_edges[-1]:
            final_edges.append(edge)
    if len(final_edges) < 2:
        final_edges = [edges[0], edges[-1]]
    labels = [f"bin_{index + 1}" for index in range(len(final_edges) - 1)]
    return {
        "type": "numeric",
        "column": column,
        "output": f"{column}_bin",
        "method": "quantile",
        "edges": final_edges,
        "labels": labels,
    }


def category_target_order(
    df: Any,
    factor: str,
    target: str,
    exposure: str | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Return a pandas table of categories ordered by observed target level.

    Parameters
    ----------
    df:
        Spark training dataframe.
    factor:
        Categorical factor column to order.
    target:
        Observed outcome column.
    exposure:
        Optional exposure column used as the denominator for the level.
    weight:
        Optional row-weight column used for weighted levels.

    Returns
    -------
    pandas.DataFrame
        Category table sorted by observed level.
    """

    spark = require_pyspark()
    F = spark.functions
    key = F.coalesce(F.col(factor).cast("string"), F.lit(_MISSING)).alias("__category")
    work = df.select(key, F.col(target).cast("double").alias("__target"), *(
        [F.col(exposure).cast("double").alias("__exposure")] if exposure is not None else []
    ), *(
        [F.col(weight).cast("double").alias("__weight")] if weight is not None else []
    ))

    if exposure is not None:
        grouped = work.groupBy("__category").agg(
            F.sum("__exposure").alias("measure"),
            F.sum("__target").alias("actual"),
        )
    elif weight is not None:
        grouped = work.groupBy("__category").agg(
            F.sum("__weight").alias("measure"),
            F.sum(F.col("__target") * F.col("__weight")).alias("actual"),
        )
    else:
        grouped = work.groupBy("__category").agg(
            F.count(F.lit(1)).cast("double").alias("measure"),
            F.sum("__target").alias("actual"),
        )

    ordered = (
        grouped.withColumn("level", F.col("actual") / F.greatest(F.col("measure"), F.lit(1e-9)))
        .withColumn("credibility", F.sqrt(F.greatest(F.col("actual"), F.lit(0.0)) + F.lit(1.0)))
        .orderBy("level", "measure")
    )
    return ordered.toPandas().rename(columns={"__category": factor})


def make_categorical_groups(
    df: Any,
    factor: str,
    target: str,
    exposure: str | None = None,
    weight: str | None = None,
    cutpoints: list[int] | None = None,
) -> JsonDict:
    """Create a target-ordered categorical grouping spec from Spark data.

    Parameters
    ----------
    df:
        Spark training dataframe.
    factor:
        Categorical factor column to group.
    target:
        Observed outcome column used for ordering categories.
    exposure:
        Optional exposure column used as the denominator for category levels.
    weight:
        Optional row-weight column used for weighted category levels.
    cutpoints:
        Ordered category positions where new groups should start.

    Returns
    -------
    dict
        JSON-serializable categorical grouping spec.
    """

    target_order = category_target_order(df, factor, target, exposure=exposure, weight=weight)
    return categorical_group_spec_from_order(factor, target_order, cutpoints=cutpoints)


def categorical_group_spec_from_order(
    factor: str,
    target_order: pd.DataFrame,
    cutpoints: list[int] | None = None,
) -> JsonDict:
    """Create a categorical grouping spec from precomputed target order."""

    cutpoints = cutpoints or []
    ordered = [str(value) for value in target_order[factor].tolist()]
    clean_cutpoints = sorted({int(point) for point in cutpoints if 0 < int(point) < len(ordered)})
    boundaries = [0, *clean_cutpoints, len(ordered)]
    labels: list[str] = []
    mapping: dict[str, str] = {}
    for index in range(len(boundaries) - 1):
        label = f"group_{index + 1:02d}"
        labels.append(label)
        for category in ordered[boundaries[index] : boundaries[index + 1]]:
            mapping[category] = label
    return {
        "type": "categorical",
        "column": factor,
        "output": f"{factor}_group",
        "order": ordered,
        "cutpoints": clean_cutpoints,
        "mapping": mapping,
        "labels": labels,
        "default": "other",
        "missing": _MISSING,
        "stats": target_order.rename(columns={factor: "category"}).to_dict("records"),
    }


def apply_spec(df: Any, spec: JsonDict, output: str | None = None) -> Any:
    """Apply a saved numeric or categorical spec to a Spark dataframe.

    Parameters
    ----------
    df:
        Spark dataframe containing the raw column referenced by ``spec``.
    spec:
        JSON-serializable numeric or categorical spec.
    output:
        Optional output column override. When omitted, the spec output is used.

    Returns
    -------
    pyspark.sql.DataFrame
        Spark dataframe with the transformed output column added.
    """

    spark = require_pyspark()
    F = spark.functions
    kind = str(spec["type"])
    column = str(spec["column"])
    output = output or str(spec.get("output") or f"{column}_{kind}")

    if kind == "numeric":
        value = F.col(column).cast("double")
        expression = None
        edges = [float(edge) for edge in spec["edges"]]
        labels = [str(label) for label in spec["labels"]]
        for index, label in enumerate(labels):
            lower = edges[index]
            upper = edges[index + 1]
            condition = (
                (value >= F.lit(lower)) & (value <= F.lit(upper))
                if index == 0
                else (value > F.lit(lower)) & (value <= F.lit(upper))
            )
            expression = F.when(condition, F.lit(label)) if expression is None else expression.when(
                condition,
                F.lit(label),
            )
        expression = expression.otherwise(F.lit("missing")) if expression is not None else F.lit("missing")
        return df.withColumn(output, expression)

    if kind == "categorical":
        keys = F.coalesce(F.col(column).cast("string"), F.lit(str(spec.get("missing", _MISSING))))
        items = []
        for key, value in spec["mapping"].items():
            items.extend([F.lit(str(key)), F.lit(str(value))])
        if not items:
            return df.withColumn(output, F.lit(str(spec.get("default", "other"))))
        mapping_expr = F.create_map(*items)
        return df.withColumn(
            output,
            F.coalesce(F.element_at(mapping_expr, keys), F.lit(str(spec.get("default", "other")))),
        )

    raise ValueError("spec type must be 'numeric' or 'categorical'.")
