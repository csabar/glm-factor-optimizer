"""Tests for the notebook-oriented study workflow."""

from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from glm_factor_optimizer import GLMStudy

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    pass


def make_data(n_rows: int = 420, seed: int = 81) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, size=n_rows)
    score = rng.normal(size=n_rows)
    age = rng.normal(45, 14, size=n_rows)
    segment = rng.choice(["a", "b", "c"], size=n_rows)
    segment_effect = pd.Series(segment).map({"a": -0.2, "b": 0.0, "c": 0.45}).to_numpy()
    events = rng.poisson(
        np.exp(-0.8 + 0.4 * (score > 0.0) + 0.2 * (age < 30.0) + segment_effect)
        * exposure
    )
    return pd.DataFrame(
        {
            "events": events,
            "exposure": exposure,
            "score": score,
            "age": age,
            "segment": segment,
        }
    )


class StudyTests(unittest.TestCase):
    def test_interactive_factor_flow_does_not_score_holdout_until_requested(self) -> None:
        study = GLMStudy(
            make_data(),
            target="events",
            exposure="exposure",
            prediction="predicted_count",
            factor_kinds={"segment": "categorical"},
            min_bin_size=5.0,
            seed=81,
        )
        study.split()
        ranking = study.rank_candidates(["score", "age", "segment"], bins=4)
        self.assertEqual(len(ranking), 3)
        self.assertIsNone(study.holdout_scored)

        score = study.factor("score")
        score.coarse_bins(bins=4)
        comparison = score.compare()
        self.assertIn("candidate_validation_deviance", comparison.columns)
        score.accept(comment="manual coarse score")

        segment = study.factor("segment")
        segment.optimize(trials=4, min_bin_size=5.0)
        segment.accept(comment="optimized segment")

        model = study.fit_main_effects()
        self.assertIn("predicted_count", study.validation_scored.columns)
        self.assertEqual(len(model.factors), 2)
        self.assertIsNone(study.holdout_scored)

        report = study.validation_report()
        self.assertIn("train_validation", report)
        self.assertIn("model_versions", report)

        holdout = study.finalize()
        self.assertIn("summary", holdout)
        self.assertIsNotNone(study.holdout_scored)

    def test_refine_factor_keeps_other_factors_fixed_and_manual_specs_save(self) -> None:
        study = GLMStudy(
            make_data(seed=82),
            target="events",
            exposure="exposure",
            prediction="predicted_count",
            factor_kinds={"segment": "categorical"},
            min_bin_size=5.0,
            seed=82,
        )
        study.split()
        score = study.factor("score")
        score.optimize(trials=4, n_prebins=4, min_bin_size=5.0)
        score.accept()
        segment = study.factor("segment")
        segment.optimize(trials=4, min_bin_size=5.0)
        segment.accept()

        refined = study.refine_factor("score", trials=4, n_prebins=4)
        self.assertEqual(refined.optimization.fixed, [study.specs["segment"]["output"]])
        json.dumps(refined.spec)
        refined.accept(comment="accepted refined score")

        manual = study.factor("age")
        manual.coarse_bins(bins=3)
        manual.set_spec(manual.spec)
        manual.accept(comment="manual age")

        with tempfile.TemporaryDirectory() as temp:
            path = study.save(temp)
            self.assertTrue((path / "specs.json").exists())
            self.assertTrue((path / "history.json").exists())

    def test_interaction_testing_and_acceptance(self) -> None:
        study = GLMStudy(
            make_data(seed=83),
            target="events",
            exposure="exposure",
            prediction="predicted_count",
            factor_kinds={"segment": "categorical"},
            min_bin_size=5.0,
            seed=83,
        )
        study.split()
        for factor in ["score", "segment"]:
            block = study.factor(factor)
            block.optimize(trials=4, n_prebins=4, min_bin_size=5.0)
            block.accept()

        study.fit_main_effects()
        candidates = study.find_interactions(min_bin_size=2.0)
        self.assertEqual(candidates.iloc[0]["pair"], "score_bin:segment_group")

        tested = study.test_interaction("score", "segment")
        output = str(tested.iloc[0]["output"])
        self.assertIn(output, study.pending_interactions)
        study.accept_interaction(output)
        self.assertIn(output, study.selected_factors)

    def test_auto_design_uses_same_acceptance_path(self) -> None:
        study = GLMStudy(
            make_data(seed=84),
            target="events",
            exposure="exposure",
            prediction="predicted_count",
            factor_kinds={"segment": "categorical"},
            min_bin_size=5.0,
            seed=84,
        )
        study.split()
        study.auto_design(["score", "segment"], top_n=1, trials=3)

        self.assertEqual(len(study.accepted_raw_factors), 1)
        self.assertTrue(any(row["action"] == "accept" for row in study.history))
        self.assertIsNone(study.holdout_scored)


if __name__ == "__main__":
    unittest.main()
