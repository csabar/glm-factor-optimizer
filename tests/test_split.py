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

    def test_split_accepts_fraction_aliases(self) -> None:
        frame = pd.DataFrame({"value": range(10)})
        train_df, validation_df, holdout_df = split(
            frame,
            train_fraction=0.5,
            validation_fraction=0.3,
            holdout_fraction=0.2,
            seed=1,
        )
        self.assertEqual(len(train_df), 5)
        self.assertEqual(len(validation_df), 3)
        self.assertEqual(len(holdout_df), 2)

    def test_split_rejects_duplicate_fraction_names(self) -> None:
        frame = pd.DataFrame({"value": range(10)})
        with self.assertRaisesRegex(ValueError, "either train or train_fraction"):
            split(frame, train=0.6, train_fraction=0.5)


if __name__ == "__main__":
    unittest.main()
