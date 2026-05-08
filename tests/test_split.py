"""Tests for dataframe splitting."""

from __future__ import annotations

import unittest

import pandas as pd

from glm_factor_optimizer import split


class SplitTests(unittest.TestCase):
    def test_split_returns_three_copies(self) -> None:
        frame = pd.DataFrame({"value": range(10)})
        train_df, validation_df, holdout_df = split(frame, seed=1)
        self.assertEqual(len(train_df) + len(validation_df) + len(holdout_df), len(frame))
        self.assertEqual(len(train_df), 6)
        self.assertEqual(len(validation_df), 2)
        self.assertEqual(len(holdout_df), 2)


if __name__ == "__main__":
    unittest.main()
