"""Documentation smoke and language-guard tests."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from glm_factor_optimizer import GLMStudy

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    pass


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_main_tutorial_workflow_runs(self) -> None:
        rng = np.random.default_rng(42)
        rows = 800
        hours = rng.uniform(0.5, 2.5, size=rows)
        machine_age = rng.normal(6.0, 2.5, size=rows).clip(0.2, 15.0)
        usage_score = rng.normal(size=rows)
        site_region = rng.choice(["north", "south", "east", "west"], size=rows)
        equipment_type = rng.choice(["standard", "compact", "heavy"], size=rows)
        region_effect = pd.Series(site_region).map(
            {"north": -0.10, "south": 0.15, "east": 0.0, "west": 0.08}
        ).to_numpy()
        equipment_effect = pd.Series(equipment_type).map(
            {"standard": 0.0, "compact": -0.15, "heavy": 0.25}
        ).to_numpy()
        mean_rate = np.exp(
            -1.0
            + 0.25 * (machine_age > 8.0)
            + 0.35 * (usage_score > 0.7)
            + region_effect
            + equipment_effect
        )
        events = rng.poisson(mean_rate * hours)
        df = pd.DataFrame(
            {
                "events": events,
                "hours": hours,
                "machine_age": machine_age,
                "usage_score": usage_score,
                "site_region": site_region,
                "equipment_type": equipment_type,
            }
        )

        study = GLMStudy(
            df,
            target="events",
            exposure="hours",
            family="poisson",
            prediction="predicted_count",
            factor_kinds={
                "site_region": "categorical",
                "equipment_type": "categorical",
            },
            min_bin_size=20.0,
            seed=42,
        )
        study.split(seed=42)
        ranking = study.rank_candidates(
            ["machine_age", "usage_score", "site_region", "equipment_type"],
            bins=5,
            max_groups=4,
        )
        self.assertEqual(len(ranking), 4)

        age = study.factor("machine_age", kind="numeric")
        age.coarse_bins(bins=5)
        age.optimize(trials=5, max_bins=5, n_prebins=8, min_bin_size=20.0)
        age.compare()
        age.validation_table()
        age.accept(comment="tutorial smoke")

        equipment = study.factor("equipment_type", kind="categorical")
        equipment.optimize(trials=5, min_bin_size=20.0)
        equipment.compare()
        equipment.accept(comment="tutorial smoke")

        study.fit_main_effects()
        report = study.validation_report()
        self.assertIn("summary", report)

        refined_age = study.refine_factor("machine_age", trials=5, max_bins=5, n_prebins=8)
        refined_age.compare()
        refined_age.accept(comment="tutorial smoke")

        interactions = study.find_interactions(min_bin_size=20.0)
        self.assertIn("pair", interactions.columns)
        test = study.test_interaction("machine_age", "equipment_type")
        self.assertIn("interaction_validation_deviance", test.columns)
        study.accept_interaction("machine_age", "equipment_type", comment="tutorial smoke")
        study.fit_main_effects()

        holdout_report = study.finalize()
        self.assertIn("summary", holdout_report)

    def test_public_docs_do_not_use_legacy_terms(self) -> None:
        public_paths = [ROOT / "README.md", ROOT / "docs"]
        banned = [
            "rate_glm_optimizer",
            "rate-glm-optimizer",
        ]
        violations: list[str] = []
        for path in public_paths:
            files = [path] if path.is_file() else list(path.rglob("*.md"))
            for file in files:
                text = file.read_text(encoding="utf-8").lower()
                for term in banned:
                    if term in text:
                        violations.append(f"{file.relative_to(ROOT)} contains {term!r}")
        self.assertEqual([], violations)

    def test_public_examples_do_not_convert_spark_inputs_to_pandas(self) -> None:
        # Public examples should not suggest converting large Spark modeling
        # tables. Internal bounded aggregate metadata may still use pandas.
        public_paths = [ROOT / "README.md", ROOT / "docs", ROOT / "examples"]
        banned = [
            ".topandas(",
            ".to_pandas(",
        ]
        violations: list[str] = []
        for path in public_paths:
            files = [path] if path.is_file() else list(path.rglob("*.md")) + list(path.rglob("*.py"))
            for file in files:
                text = file.read_text(encoding="utf-8").lower()
                for term in banned:
                    if term in text:
                        violations.append(f"{file.relative_to(ROOT)} contains {term!r}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
