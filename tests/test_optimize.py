"""Tests for factor optimization."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from rate_glm_optimizer import GLM, RateGLM, small_count_penalty, split

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    pass


def make_data(n_rows: int = 400, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.4, 1.8, size=n_rows)
    score = rng.normal(size=n_rows)
    segment = rng.choice(["a", "b", "c"], size=n_rows)
    effect = pd.Series(segment).map({"a": -0.2, "b": 0.0, "c": 0.35}).to_numpy()
    events = rng.poisson(np.exp(-0.7 + effect + 0.4 * (score > 0.0)) * exposure)
    return pd.DataFrame({"events": events, "exposure": exposure, "score": score, "segment": segment})


class OptimizeTests(unittest.TestCase):
    def test_numeric_optimization_with_fixed_factor(self) -> None:
        train_df, validation_df, _ = split(make_data(), seed=31)
        glm = RateGLM(target="events", exposure="exposure")
        result = glm.optimize(
            train_df,
            validation_df,
            "score",
            fixed=["segment"],
            trials=5,
            n_prebins=4,
            min_bin_size=5.0,
            seed=31,
        )
        json.dumps(result.spec)
        self.assertEqual(result.kind, "numeric")
        self.assertEqual(result.fixed, ["segment"])

    def test_fixed_factors_alias(self) -> None:
        train_df, validation_df, _ = split(make_data(seed=35), seed=35)
        glm = RateGLM(target="events", exposure="exposure")
        result = glm.optimize(
            train_df,
            validation_df,
            "score",
            fixed_factors=["segment"],
            trials=4,
            n_prebins=4,
            min_bin_size=5.0,
            seed=35,
        )

        self.assertEqual(result.fixed, ["segment"])

    def test_categorical_optimization(self) -> None:
        train_df, validation_df, _ = split(make_data(seed=32), seed=32)
        glm = RateGLM(target="events", exposure="exposure")
        result = glm.optimize(
            train_df,
            validation_df,
            "segment",
            kind="categorical",
            trials=5,
            min_bin_size=5.0,
            seed=32,
        )
        json.dumps(result.spec)
        self.assertEqual(result.kind, "categorical")
        self.assertIn("mapping", result.spec)

    def test_custom_penalties_are_added_to_trials(self) -> None:
        train_df, validation_df, _ = split(make_data(seed=33), seed=33)
        glm = RateGLM(target="events", exposure="exposure")
        result = glm.optimize(
            train_df,
            validation_df,
            "score",
            trials=5,
            n_prebins=4,
            min_bin_size=5.0,
            seed=33,
            penalties={
                "small_count": small_count_penalty(min_count=3, penalty=0.25),
                "many_bins": lambda context: 0.1 * max(context["bin_count"] - 2, 0),
            },
        )

        self.assertIn("custom_penalty", result.trials.columns)
        self.assertIn("penalty_small_count", result.trials.columns)
        self.assertIn("penalty_many_bins", result.trials.columns)
        self.assertGreaterEqual(result.penalty_breakdown["custom"], 0.0)

    def test_gamma_optimization_without_exposure(self) -> None:
        rng = np.random.default_rng(34)
        n_rows = 350
        score = rng.normal(size=n_rows)
        segment = rng.choice(["a", "b"], size=n_rows)
        segment_effect = pd.Series(segment).map({"a": 0.0, "b": 0.3}).to_numpy()
        mean = np.exp(7.0 + 0.4 * (score > 0.0) + segment_effect)
        severity = rng.gamma(shape=2.0, scale=mean / 2.0)
        frame = pd.DataFrame({"severity": severity, "score": score, "segment": segment})
        train_df, validation_df, _ = split(frame, seed=34)

        glm = GLM(target="severity", family="gamma", prediction="predicted_severity")
        result = glm.optimize(
            train_df,
            validation_df,
            "score",
            fixed=["segment"],
            trials=5,
            n_prebins=4,
            min_bin_size=5.0,
            seed=34,
        )
        transformed = glm.apply(validation_df, result.spec)
        model = glm.fit(glm.apply(train_df, result.spec), [result.output, "segment"])
        scored = glm.predict(transformed, model)

        self.assertEqual(result.kind, "numeric")
        self.assertIn("predicted_severity", scored.columns)
        self.assertGreater(float(scored["predicted_severity"].min()), 0.0)


if __name__ == "__main__":
    unittest.main()
