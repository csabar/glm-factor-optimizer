"""Tests for higher-level workflows."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rate_glm_optimizer import GLMWorkflow

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    pass


class WorkflowTests(unittest.TestCase):
    def test_workflow_optimizes_factors_and_scores_holdout(self) -> None:
        rng = np.random.default_rng(41)
        n_rows = 350
        exposure = rng.uniform(0.5, 2.0, size=n_rows)
        score = rng.normal(size=n_rows)
        segment = rng.choice(["a", "b"], size=n_rows)
        effect = pd.Series(segment).map({"a": 0.0, "b": 0.3}).to_numpy()
        events = rng.poisson(np.exp(-0.8 + 0.4 * (score > 0.0) + effect) * exposure)
        frame = pd.DataFrame(
            {"events": events, "exposure": exposure, "score": score, "segment": segment}
        )

        workflow = GLMWorkflow(
            target="events",
            family="poisson",
            exposure="exposure",
            prediction="predicted_count",
            trials=4,
            n_prebins=4,
            min_bin_size=5.0,
            factor_kinds={"segment": "categorical"},
            rank_candidates=True,
            interaction_diagnostics=True,
            seed=41,
        )
        result = workflow.fit(frame, ["score", "segment"])

        self.assertEqual(len(result.specs), 2)
        self.assertEqual(len(result.selected_factors), 2)
        self.assertEqual(len(result.selected_raw_factors), 2)
        self.assertIsNotNone(result.ranking)
        self.assertIn("predicted_count", result.holdout.columns)
        self.assertIn("summary", result.validation_report)
        self.assertFalse(result.coefficients.empty)
        self.assertIsNotNone(result.diagnostics)


if __name__ == "__main__":
    unittest.main()
