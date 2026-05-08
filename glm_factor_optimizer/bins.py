"""Binning and grouping specs for model factors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

JsonDict = dict[str, Any]

_MISSING = "__missing__"


def _final_edges(raw_edges: list[float], series: pd.Series) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Cannot create bins from an empty numeric column.")
    edges = sorted({float(value) for value in raw_edges})
    lower = float(clean.min()) - 1e-9
    upper = float(clean.max()) + 1e-9
    if len(edges) < 2:
        midpoint = float(clean.median())
        edges = [lower, midpoint, upper]
    else:
        edges[0] = min(edges[0], lower)
        edges[-1] = max(edges[-1], upper)
    final = [edges[0]]
    for edge in edges[1:]:
        if edge > final[-1]:
            final.append(edge)
    return final if len(final) >= 2 else [lower, upper]


def make_numeric_bins(
    series: pd.Series,
    bins: int = 10,
    column: str | None = None,
    method: str = "quantile",
) -> JsonDict:
    """Create a JSON-serializable numeric binning spec."""

    if bins < 1:
        raise ValueError("bins must be at least 1.")
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if method == "quantile":
        raw_edges = clean.quantile(np.linspace(0.0, 1.0, bins + 1)).tolist()
    elif method == "uniform":
        raw_edges = np.linspace(float(clean.min()), float(clean.max()), bins + 1).tolist()
    else:
        raise ValueError("method must be 'quantile' or 'uniform'.")

    edges = _final_edges(raw_edges, clean)
    labels = [f"bin_{index + 1}" for index in range(len(edges) - 1)]
    source_column = column or series.name or "value"
    return {
        "type": "numeric",
        "column": source_column,
        "output": f"{source_column}_bin",
        "method": method,
        "edges": edges,
        "labels": labels,
    }


def _category_key(value: object) -> str:
    if pd.isna(value):
        return _MISSING
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def category_target_order(
    df: pd.DataFrame,
    factor: str,
    target: str,
    exposure: str | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Order categories by observed training target level."""

    columns = [factor, target]
    if exposure is not None:
        columns.append(exposure)
    if weight is not None and weight not in columns:
        columns.append(weight)
    work = df[columns].copy()
    work[factor] = work[factor].map(_category_key)
    if exposure is not None:
        table = (
            work.groupby(factor, dropna=False)
            .agg(measure=(exposure, "sum"), actual=(target, "sum"))
            .reset_index()
        )
        table["level"] = table["actual"] / table["measure"].clip(lower=1e-9)
    elif weight is not None:
        work["_weighted_target"] = work[target] * work[weight]
        table = (
            work.groupby(factor, dropna=False)
            .agg(measure=(weight, "sum"), actual=("_weighted_target", "sum"))
            .reset_index()
        )
        table["level"] = table["actual"] / table["measure"].clip(lower=1e-9)
    else:
        table = (
            work.groupby(factor, dropna=False)
            .agg(measure=(target, "size"), actual=(target, "sum"))
            .reset_index()
        )
        table["level"] = table["actual"] / table["measure"].clip(lower=1e-9)
    table["credibility"] = np.sqrt(table["actual"].clip(lower=0.0) + 1.0)
    return table.sort_values(["level", "measure"]).reset_index(drop=True)


def make_categorical_groups(
    df: pd.DataFrame,
    factor: str,
    target: str,
    exposure: str | None = None,
    weight: str | None = None,
    cutpoints: list[int] | None = None,
) -> JsonDict:
    """Create a target-ordered categorical grouping spec from training data."""

    cutpoints = cutpoints or []
    target_order = category_target_order(df, factor, target, exposure=exposure, weight=weight)
    ordered = [str(value) for value in target_order[factor].tolist()]
    clean_cutpoints = sorted({int(point) for point in cutpoints if 0 < int(point) < len(ordered)})
    boundaries = [0, *clean_cutpoints, len(ordered)]
    mapping: dict[str, str] = {}
    labels: list[str] = []

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


def apply_spec(
    df: pd.DataFrame,
    spec: JsonDict,
    output: str | None = None,
) -> pd.DataFrame:
    """Apply a saved numeric or categorical spec to a dataframe."""

    kind = str(spec["type"])
    column = str(spec["column"])
    output = output or str(spec.get("output") or f"{column}_{kind}")
    result = df.copy()

    if kind == "numeric":
        binned = pd.cut(
            pd.to_numeric(result[column], errors="coerce"),
            bins=spec["edges"],
            labels=spec["labels"],
            include_lowest=True,
        )
        result[output] = binned.astype(object)
        result[output] = result[output].where(~result[output].isna(), other="missing")
        return result

    if kind == "categorical":
        keys = result[column].map(_category_key)
        result[output] = keys.map(spec["mapping"]).fillna(str(spec.get("default", "other")))
        return result

    raise ValueError("spec type must be 'numeric' or 'categorical'.")
