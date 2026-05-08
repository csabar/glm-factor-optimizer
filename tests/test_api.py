"""Tests for the simple public API."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from glm_factor_optimizer import RateGLM, apply_spec, optimize_bins, split

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    pass


def make_data(n_rows: int = 500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0.5, 2.0, size=n_rows)
    score = rng.normal(size=n_rows)
    segment = rng.choice(["small", "medium", "large"], size=n_rows)
    segment_effect = pd.Series(segment).map({"small": -0.2, "medium": 0.0, "large": 0.35}).to_numpy()
    events = rng.poisson(np.exp(-0.7 + 0.4 * (score > 0.2) + segment_effect) * hours)
    return pd.DataFrame({"events": events, "hours": hours, "score": score, "segment": segment})


class ApiTests(unittest.TestCase):
    def test_manual_fit_and_report(self) -> None:
        train_df, validation_df, _ = split(make_data(), seed=3)
        glm = RateGLM(target="events", exposure="hours")

        spec = glm.bins(train_df, "score", bins=4)
        train_binned = glm.apply(train_df, spec)
        validation_binned = glm.apply(validation_df, spec)
        model = glm.fit(train_binned, factors=[spec["output"], "segment"])
        scored = glm.predict(validation_binned, model)
        report = glm.report(scored, bins=5)

        self.assertIn("predicted_count", scored.columns)
        self.assertGreater(float(scored["predicted_count"].sum()), 0.0)
        self.assertIn("summary", report)
        self.assertIn("calibration", report)

    def test_optimize_then_apply_spec(self) -> None:
        train_df, validation_df, _ = split(make_data(n_rows=350, seed=4), seed=4)
        glm = RateGLM(target="events", exposure="hours")

        result = glm.optimize(
            train_df,
            validation_df,
            "score",
            fixed=["segment"],
            trials=5,
            n_prebins=4,
            min_bin_size=5.0,
            seed=4,
        )

        json.dumps(result.spec)
        transformed = apply_spec(validation_df, result.spec)
        self.assertIn(result.output, transformed.columns)
        self.assertEqual(result.fixed, ["segment"])

    def test_optimize_bins_alias_is_public(self) -> None:
        train_df, validation_df, _ = split(make_data(n_rows=320, seed=5), seed=5)
        result = optimize_bins(
            train_df,
            validation_df,
            target="events",
            exposure="hours",
            factor="score",
            trials=4,
            n_prebins=4,
            min_bin_size=5.0,
            seed=5,
        )

        self.assertEqual(result.factor, "score")


if __name__ == "__main__":
    unittest.main()
