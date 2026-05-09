"""Tests for Spark backend packaging without requiring a Spark runtime."""

from __future__ import annotations

import json
import os
import sys
import unittest
import warnings


class SparkBackendTests(unittest.TestCase):
    def test_top_level_spark_rate_glm_import_is_lazy(self) -> None:
        import glm_factor_optimizer as public_api

        self.assertNotIn("SparkRateGLM", public_api.__dict__)

        from glm_factor_optimizer import SparkRateGLM

        self.assertIs(SparkRateGLM, public_api.SparkRateGLM)
        self.assertEqual(SparkRateGLM.__name__, "RateGLM")

    def test_spark_backend_imports_without_pyspark(self) -> None:
        from glm_factor_optimizer.spark import (
            SparkGLM,
            SparkGLMWorkflow,
            RateGLM,
            aggregate_rate_table,
            category_target_order,
        )

        self.assertEqual(SparkGLM.__name__, "SparkGLM")
        self.assertEqual(SparkGLMWorkflow.__name__, "SparkGLMWorkflow")
        self.assertEqual(RateGLM.__name__, "RateGLM")
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
            from pyspark.sql import DataFrame as SparkDataFrame
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F
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
        from glm_factor_optimizer import GLM as PublicGLM
        from glm_factor_optimizer import RateGLM as PublicRateGLM
        from glm_factor_optimizer import split as public_split
        from glm_factor_optimizer.spark.model import FittedSparkGLM

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

            public_glm = PublicGLM(target="events", exposure="hours", prediction="predicted_events")
            public_model = public_glm.fit(train, ["machine_age", "site_region"])
            self.assertIsInstance(public_model, FittedSparkGLM)
            public_scored_validation = public_glm.predict(validation, public_model)
            self.assertIsInstance(public_scored_validation, SparkDataFrame)
            public_report = public_glm.report(public_scored_validation)
            self.assertIsInstance(public_report["summary"], SparkDataFrame)
            self.assertIsInstance(public_report["calibration"], SparkDataFrame)
            self.assertNotIn("lift", public_report)
            public_spec = public_glm.bins(train, "machine_age", bins=4)
            public_applied = public_glm.apply(train, public_spec)
            self.assertIsInstance(public_applied, SparkDataFrame)
            self.assertIn("machine_age_bin", public_applied.columns)
            public_optimized = public_glm.optimize(
                train,
                validation,
                "machine_age",
                trials=1,
                n_prebins=4,
                min_bin_size=1.0,
            )
            json.dumps(public_optimized.spec)

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

            public_train, public_validation, public_holdout = public_split(df, seed=42)
            manual_rate_glm = PublicRateGLM(
                target_col="events",
                exposure_col="hours",
                family="poisson",
                prediction_col="predicted_events",
            )
            rate_model = manual_rate_glm.fit(public_train, factors=["machine_age", "site_region"])
            self.assertIsInstance(rate_model, FittedSparkGLM)
            rate_scored = manual_rate_glm.predict(public_validation, rate_model)
            self.assertIsInstance(rate_scored, SparkDataFrame)
            self.assertIn("predicted_events", rate_scored.columns)

            weighted_train = public_train.withColumn("row_weight", F.lit(1.0))
            missing_weight_glm = PublicRateGLM(
                candidate_factors=["machine_age"],
                target_col="events",
                exposure_col="hours",
                weight_col="row_weight",
                family="poisson",
            )
            with self.assertRaisesRegex(KeyError, "row_weight"):
                missing_weight_glm.fit(weighted_train, validation_df=public_validation)

            auto_glm = PublicRateGLM(
                candidate_factors=["machine_age", "site_region", "equipment_type"],
                target_col="events",
                exposure_col="hours",
                family="poisson",
                prediction_col="predicted_events",
                top_n=2,
            )
            fitted = auto_glm.fit(public_train, validation_df=public_validation)
            self.assertIs(fitted, auto_glm)
            self.assertIsInstance(auto_glm.model_, FittedSparkGLM)
            self.assertIsInstance(auto_glm.identified_factors_, SparkDataFrame)
            self.assertGreater(auto_glm.identified_factors_.count(), 0)
            self.assertIn("selected", auto_glm.identified_factors_.columns)
            self.assertLessEqual(len(auto_glm.selected_factors_), 2)
            scored_holdout = auto_glm.predict(public_holdout)
            self.assertIsInstance(scored_holdout, SparkDataFrame)
            self.assertIn("predicted_events", scored_holdout.columns)
            self.assertGreaterEqual(scored_holdout.count(), 0)
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
