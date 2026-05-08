"""Tests for count-rate metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rate_glm_optimizer import poisson_deviance, summary, weighted_mae, weighted_rmse


class MetricsTests(unittest.TestCase):
    def test_poisson_deviance_is_zero_for_perfect_predictions(self) -> None:
        actual = np.array([0.0, 1.0, 3.0, 2.0])
        predicted = np.array([0.0, 1.0, 3.0, 2.0])
        self.assertAlmostEqual(poisson_deviance(actual, predicted), 0.0, places=8)

    def test_summary_contains_rates(self) -> None:
        frame = pd.DataFrame(
            {
                "events": [0, 1, 2],
                "predicted_count": [0.2, 0.8, 1.7],
                "hours": [0.5, 1.0, 1.5],
            }
        )
        table = summary(frame, target="events", prediction="predicted_count", exposure="hours")
        self.assertEqual(table.loc[0, "rows"], 3)
        self.assertGreater(table.loc[0, "actual_rate"], 0.0)

    def test_weighted_error_metrics(self) -> None:
        actual = np.array([1.0, 3.0])
        predicted = np.array([2.0, 1.0])
        weight = np.array([1.0, 3.0])

        self.assertAlmostEqual(weighted_mae(actual, predicted, weight), 1.75)
        self.assertAlmostEqual(weighted_rmse(actual, predicted, weight), np.sqrt(3.25))


if __name__ == "__main__":
    unittest.main()
