"""Small train/validation/holdout split helper."""

from __future__ import annotations

import pandas as pd


def split(
    df: pd.DataFrame,
    train: float = 0.6,
    validation: float = 0.2,
    holdout: float = 0.2,
    *,
    seed: int = 42,
    time: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and holdout dataframes."""

    total = train + validation + holdout
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train + validation + holdout must equal 1.0.")
    if len(df) == 0:
        raise ValueError("Cannot split an empty dataframe.")

    ordered = df.sort_values(time).reset_index(drop=True) if time else df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)
    train_end = int(round(len(ordered) * train))
    validation_end = train_end + int(round(len(ordered) * validation))
    validation_end = min(validation_end, len(ordered))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )
