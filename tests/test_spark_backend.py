"""Tests for Spark backend packaging without requiring a Spark runtime."""

from __future__ import annotations

import json
import os
import sys
import unittest
import warnings


class SparkBackendTests(unittest.TestCase):
    def test_spark_backend_imports_without_pyspark(self) -> None:
        from glm_factor_optimizer.spark import (
            SparkGLM,
            SparkGLMWorkflow,
            aggregate_rate_table,
            category_target_order,
        )

        self.assertEqual(SparkGLM.__name__, "SparkGLM")
        self.assertEqual(SparkGLMWorkflow.__name__, "SparkGLMWorkflow")
        self.assertEqual(aggregate_rate_table.__name__, "aggregate_rate_table")
        self.assertEqual(category_target_order.__name__, "category_target_order")

    def test_lazy_pyspark_error_is_actionable_when_missing(self) -> None:
        from glm_factor_optimizer.spark._deps import require_pyspark

        try:
            require_pyspark()
        except ImportError as exc:
            self.assertIn("glm-factor-optimizer[spark]", str(exc))

    def test_local_spark_runtime_with_sample_data(self) -> None:
        if os.environ.get("GLM_FACTOR_OPTIMIZER_RUN_SPARK_TESTS") != "1":
            self.skipTest("Set GLM_FACTOR_OPTIMIZER_RUN_SPARK_TESTS=1 to run local Spark integration tests.")

        try:
            from pyspark.sql import SparkSession
        except ModuleNotFoundError:
            self.skipTest("PySpark is not installed.")

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        warnings.filterwarnings("ignore", category=ResourceWarning)

        from glm_factor_optimizer.spark import (
            SparkGLM,
            SparkGLMWorkflow,
            aggregate_rate_table,
            apply_spec,
            make_categorical_groups,
            make_numeric_bins,
            optimize_factor,
            split,
        )

        spark = (
            SparkSession.builder.master("local[1]")
            .appName("glm-factor-optimizer-test")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        try:
            df = spark.createDataFrame(
                _spark_sample_rows(),
                ["events", "hours", "machine_age", "site_region", "equipment_type"],
            )
            train, validation, holdout = split(df, seed=42)
            self.assertGreater(train.count(), 0)
            self.assertGreater(validation.count(), 0)
            self.assertGreater(holdout.count(), 0)

            numeric_spec = make_numeric_bins(train, "machine_age", bins=4)
            categorical_spec = make_categorical_groups(train, "site_region", "events", exposure="hours")
            json.dumps(numeric_spec)
            json.dumps(categorical_spec)

            train_transformed = apply_spec(apply_spec(train, numeric_spec), categorical_spec)
            validation_transformed = apply_spec(apply_spec(validation, numeric_spec), categorical_spec)
            self.assertIn("machine_age_bin", train_transformed.columns)
            self.assertIn("site_region_group", train_transformed.columns)

            aggregated = aggregate_rate_table(
                train_transformed,
                "site_region_group",
                target="events",
                exposure="hours",
            )
            self.assertGreater(aggregated.count(), 0)
            self.assertIn("actual_rate", aggregated.columns)

            glm = SparkGLM(target="events", exposure="hours", prediction="predicted_events")
            model = glm.fit(train_transformed, ["machine_age_bin", "site_region_group"])
            scored_validation = glm.predict(validation_transformed, model)
            self.assertEqual(scored_validation.select("predicted_events").count(), validation_transformed.count())
            self.assertGreater(glm.report(scored_validation)["summary"].collect()[0]["predicted"], 0.0)

            optimized = optimize_factor(
                train,
                validation,
                target="events",
                exposure="hours",
                factor="machine_age",
                kind="numeric",
                prediction="predicted_events",
                trials=2,
                n_prebins=4,
                min_bin_size=1.0,
            )
            json.dumps(optimized.spec)
            self.assertEqual(optimized.output, "machine_age_bin")
            self.assertEqual(len(optimized.trials), 2)

            workflow = SparkGLMWorkflow(
                target="events",
                exposure="hours",
                prediction="predicted_events",
                trials=2,
                n_prebins=4,
                min_bin_size=1.0,
                factor_kinds={"machine_age": "numeric", "site_region": "categorical"},
            )
            result = workflow.fit(df, ["machine_age", "site_region"])
            self.assertEqual(result.selected_factors, ["machine_age_bin", "site_region_group"])
            self.assertGreater(result.validation_report["summary"].count(), 0)
        finally:
            spark.stop()


def _spark_sample_rows() -> list[tuple[int, float, float, str, str]]:
    rows: list[tuple[int, float, float, str, str]] = []
    regions = ["north", "south", "east", "west"]
    equipment = ["pump", "press", "lathe"]
    for index in range(80):
        hours = float(1 + (index % 5))
        machine_age = float(1 + (index % 20))
        region = regions[index % len(regions)]
        equipment_type = equipment[index % len(equipment)]
        base_level = 0.15 * hours
        age_effect = 0.03 * (machine_age > 10)
        region_effect = {"north": 0.02, "south": 0.08, "east": 0.04, "west": 0.01}[region]
        equipment_effect = {"pump": 0.03, "press": 0.06, "lathe": 0.02}[equipment_type]
        events = int((base_level + age_effect + region_effect + equipment_effect) > 0.25) + int(index % 17 == 0)
        rows.append((events, hours, machine_age, region, equipment_type))
    return rows


if __name__ == "__main__":
    unittest.main()
