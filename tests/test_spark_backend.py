"""Tests for Spark backend packaging without requiring a Spark runtime."""

from __future__ import annotations

import unittest


class SparkBackendTests(unittest.TestCase):
    def test_spark_backend_imports_without_pyspark(self) -> None:
        from rate_glm_optimizer.spark import SparkGLM, SparkGLMWorkflow, aggregate_rate_table

        self.assertEqual(SparkGLM.__name__, "SparkGLM")
        self.assertEqual(SparkGLMWorkflow.__name__, "SparkGLMWorkflow")
        self.assertEqual(aggregate_rate_table.__name__, "aggregate_rate_table")

    def test_lazy_pyspark_error_is_actionable_when_missing(self) -> None:
        from rate_glm_optimizer.spark._deps import require_pyspark

        try:
            require_pyspark()
        except ImportError as exc:
            self.assertIn("rate-glm-optimizer[spark]", str(exc))


if __name__ == "__main__":
    unittest.main()
