"""Tests for helper modules around the core optimizer."""

from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from glm_factor_optimizer import (
    RunLogger,
    aggregate_rate_table,
    find_interactions,
    small_bin_size_penalty,
    small_count_penalty,
    stratified_sample,
)


class HelperTests(unittest.TestCase):
    def test_stratified_sample_keeps_groups_and_adds_weights(self) -> None:
        frame = pd.DataFrame(
            {
                "segment": ["a"] * 10 + ["b"] * 20 + ["c"] * 30,
                "value": np.arange(60),
                "priority": np.ones(60),
            }
        )

        sample = stratified_sample(
            frame,
            "segment",
            size=12,
            min_per_group=2,
            seed=61,
            weight_col="priority",
            add_sample_weight=True,
        )

        self.assertEqual(len(sample), 12)
        self.assertEqual(set(sample["segment"]), {"a", "b", "c"})
        self.assertIn("sample_weight", sample.columns)

    def test_aggregate_rate_table_returns_rates(self) -> None:
        frame = pd.DataFrame(
            {
                "group": ["a", "a", "b"],
                "actual": [1.0, 2.0, 3.0],
                "expected": [1.5, 1.5, 2.5],
                "measure": [2.0, 2.0, 3.0],
            }
        )

        table = aggregate_rate_table(
            frame,
            "group",
            target="actual",
            exposure="measure",
            prediction="expected",
        )

        self.assertEqual(set(table["group"]), {"a", "b"})
        self.assertIn("actual_rate", table.columns)
        self.assertIn("actual_to_predicted", table.columns)

    def test_find_interactions_returns_stable_sorted_candidates(self) -> None:
        frame = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y"] * 20,
                "b": ["m", "n", "m", "n"] * 20,
                "actual": [2.0, 1.0, 1.0, 3.0] * 20,
                "predicted": [1.0, 1.0, 1.0, 1.0] * 20,
            }
        )

        result = find_interactions(
            frame.iloc[:40],
            frame.iloc[40:],
            factors=["a", "b"],
            target="actual",
            prediction="predicted",
            min_bin_size=1.0,
        )

        self.assertEqual(result.iloc[0]["pair"], "a:b")
        self.assertGreater(result.iloc[0]["score"], 0.0)

    def test_penalty_factories_are_json_compatible_in_context(self) -> None:
        table = pd.DataFrame({"bin_size": [10.0, 2.0], "actual": [5.0, 0.0]})
        context = {
            "bin_table": table,
            "bin_count": 2,
            "train_deviance": 0.5,
            "validation_deviance": 0.7,
        }

        penalties = {
            "small_bin": small_bin_size_penalty(5.0)(context),
            "small_count": small_count_penalty(1.0)(context),
        }

        json.dumps(penalties)
        self.assertGreater(penalties["small_bin"], 0.0)
        self.assertGreater(penalties["small_count"], 0.0)

    def test_run_logger_writes_json_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            logger = RunLogger(temp, name="unit test")
            logger.log_params({"family": "poisson", "trials": 3})
            logger.log_frame("table.csv", pd.DataFrame({"x": [1, 2]}))

            self.assertTrue((logger.path / "params.json").exists())
            self.assertTrue((logger.path / "table.csv").exists())


if __name__ == "__main__":
    unittest.main()

