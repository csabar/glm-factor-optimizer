"""Tests for candidate factor screening."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from rate_glm_optimizer import RateGLM, rank_factors, split


def make_data(n_rows: int = 450, seed: int = 51) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, size=n_rows)
    score = rng.normal(size=n_rows)
    noise = rng.normal(size=n_rows)
    segment = rng.choice(["a", "b", "c"], size=n_rows)
    effect = pd.Series(segment).map({"a": -0.2, "b": 0.0, "c": 0.45}).to_numpy()
    events = rng.poisson(np.exp(-0.8 + effect + 0.5 * (score > 0.0)) * exposure)
    return pd.DataFrame(
        {
            "events": events,
            "exposure": exposure,
            "score": score,
            "noise": noise,
            "segment": segment,
        }
    )


class ScreeningTests(unittest.TestCase):
    def test_rank_factors_supports_numeric_and_categorical_specs(self) -> None:
        train, validation, _ = split(make_data(), seed=51)
        ranking = rank_factors(
            train,
            validation,
            target="events",
            exposure="exposure",
            factors=["noise", "score", "segment"],
            factor_kinds={"segment": "categorical"},
            bins=4,
            max_groups=3,
            min_bin_size=5.0,
        )

        self.assertEqual(set(ranking["factor"]), {"noise", "score", "segment"})
        self.assertIn(ranking.iloc[0]["factor"], {"score", "segment"})
        self.assertIn("p_value", ranking.columns)
        self.assertIn("train_missing_rate", ranking.columns)
        self.assertIn("validation_measure_coverage", ranking.columns)
        for spec in ranking["spec"]:
            json.dumps(spec)

    def test_glm_rank_uses_model_configuration(self) -> None:
        train, validation, _ = split(make_data(seed=52), seed=52)
        glm = RateGLM(target="events", exposure="exposure")
        ranking = glm.rank(
            train,
            validation,
            ["score", "segment"],
            factor_kinds={"segment": "categorical"},
            bins=4,
        )

        self.assertEqual(len(ranking), 2)
        self.assertTrue((ranking["baseline_validation_deviance"] > 0.0).all())


if __name__ == "__main__":
    unittest.main()
