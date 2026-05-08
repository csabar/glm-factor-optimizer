"""Generic aggregation helpers for model development tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

AggregationSpec = Mapping[str, tuple[str, str]]


def aggregate_table(
    df: pd.DataFrame,
    group_by: str | Sequence[str] | None = None,
    aggregations: AggregationSpec | None = None,
) -> pd.DataFrame:
    """Return grouped rows plus optional named aggregations."""

    group_cols = _as_list(group_by)
    named = {name: tuple(spec) for name, spec in (aggregations or {}).items()}
    if group_cols:
        return (
            df.groupby(group_cols, dropna=False)
            .agg(rows=(group_cols[0], "size"), **named)
            .reset_index()
        )
    row: dict[str, Any] = {"rows": int(len(df))}
    for name, (column, function) in named.items():
        row[name] = _aggregate_series(df[column], function)
    return pd.DataFrame([row])


def aggregate_rate_table(
    df: pd.DataFrame,
    group_by: str | Sequence[str] | None,
    *,
    target: str,
    exposure: str | None = None,
    weight: str | None = None,
    prediction: str | None = None,
    extras: AggregationSpec | None = None,
) -> pd.DataFrame:
    """Return grouped actual/predicted totals and rate-style diagnostics."""

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
        table["actual_rate"] = table["actual"] / table["exposure"].clip(lower=1e-9)
        if prediction is not None:
            table["predicted_rate"] = table["predicted"] / table["exposure"].clip(lower=1e-9)
    else:
        table["actual_mean"] = table["actual"] / table["rows"].clip(lower=1)
        if prediction is not None:
            table["predicted_mean"] = table["predicted"] / table["rows"].clip(lower=1)
    if prediction is not None:
        table["actual_to_predicted"] = table["actual"] / table["predicted"].clip(lower=1e-9)
    return table


def _aggregate_series(series: pd.Series, function: str) -> Any:
    normalized = function.lower().strip()
    if normalized == "sum":
        return series.sum()
    if normalized in {"mean", "avg"}:
        return series.mean()
    if normalized == "min":
        return series.min()
    if normalized == "max":
        return series.max()
    if normalized == "count":
        return int(series.count())
    if normalized == "size":
        return int(len(series))
    if normalized == "std":
        return series.std()
    raise ValueError(f"Unsupported aggregation function {function!r}.")


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
