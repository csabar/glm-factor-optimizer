"""Synthetic example data for local demos.

This file is intentionally outside the installable package API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_count_rate_data(n_rows: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """Return a generic count-rate modeling dataset."""

    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.2, 2.5, size=n_rows)
    age = rng.normal(45, 14, size=n_rows).clip(18, 85)
    score = rng.normal(size=n_rows)
    segment = rng.choice(["basic", "standard", "advanced"], p=[0.35, 0.45, 0.20], size=n_rows)
    region = rng.choice(["north", "south", "east", "west"], size=n_rows)
    segment_effect = pd.Series(segment).map({"basic": 0.20, "standard": 0.0, "advanced": -0.25}).to_numpy()
    region_effect = pd.Series(region).map({"north": -0.10, "south": 0.15, "east": 0.0, "west": 0.08}).to_numpy()
    mean = np.exp(-1.0 + 0.25 * (age < 30) + 0.35 * (score > 0.5) + segment_effect + region_effect)
    events = rng.poisson(mean * exposure)
    return pd.DataFrame(
        {
            "events": events,
            "exposure": exposure,
            "age": age,
            "score": score,
            "segment": segment,
            "region": region,
        }
    )


def make_gamma_severity_data(n_rows: int = 2_000, seed: int = 43) -> pd.DataFrame:
    """Return a generic positive continuous target dataset."""

    rng = np.random.default_rng(seed)
    age = rng.normal(45, 14, size=n_rows).clip(18, 85)
    score = rng.normal(size=n_rows)
    segment = rng.choice(["basic", "standard", "advanced"], p=[0.35, 0.45, 0.20], size=n_rows)
    segment_effect = pd.Series(segment).map({"basic": -0.10, "standard": 0.0, "advanced": 0.30}).to_numpy()
    mean = np.exp(7.8 + 0.25 * (age > 60) + 0.20 * (score > 0.5) + segment_effect)
    severity = rng.gamma(shape=2.5, scale=mean / 2.5)
    return pd.DataFrame({"severity": severity, "age": age, "score": score, "segment": segment})
