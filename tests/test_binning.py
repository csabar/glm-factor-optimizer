"""Tests for factor specs."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from glm_factor_optimizer import apply_spec, category_target_order, make_numeric_bins
from glm_factor_optimizer.bins import make_categorical_groups


class BinningTests(unittest.TestCase):
    def test_numeric_spec_is_json_serializable_and_applicable(self) -> None:
        frame = pd.DataFrame({"score": [1, 2, 3, 4, 5, 6]})
        spec = make_numeric_bins(frame["score"], bins=3, column="score")
        json.dumps(spec)
        result = apply_spec(frame, spec)
        self.assertIn("score_bin", result.columns)
        self.assertEqual(result["score_bin"].isna().sum(), 0)

    def test_categorical_spec_is_json_serializable_and_applicable(self) -> None:
        frame = pd.DataFrame(
            {
                "segment": ["a", "a", "b", "b", "c", "c"],
                "events": [0, 1, 0, 0, 2, 1],
                "exposure": [1, 1, 1, 1, 1, 1],
            }
        )
        spec = make_categorical_groups(frame, "segment", "events", "exposure", cutpoints=[1])
        json.dumps(spec)
        result = apply_spec(frame, spec)
        self.assertIn("segment_group", result.columns)

    def test_category_target_order_is_public(self) -> None:
        frame = pd.DataFrame(
            {
                "segment": ["a", "a", "b", "b", "c", "c"],
                "events": [0, 1, 0, 0, 2, 1],
                "exposure": [1, 1, 1, 1, 1, 1],
            }
        )

        ordered = category_target_order(frame, "segment", "events", exposure="exposure")

        self.assertEqual(ordered.iloc[0]["segment"], "b")
        self.assertIn("level", ordered.columns)


if __name__ == "__main__":
    unittest.main()
