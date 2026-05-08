"""Self-contained Spark backend example.

Run inside a PySpark environment, or install locally with:

    pip install "glm-factor-optimizer[spark]"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_factor_optimizer.spark import SparkGLMWorkflow


def make_sample_rows() -> list[tuple[int, float, float, str, str]]:
    """Create a small neutral event-rate dataset for local Spark examples."""

    rows: list[tuple[int, float, float, str, str]] = []
    regions = ["north", "south", "east", "west"]
    equipment = ["pump", "press", "lathe"]
    for index in range(120):
        hours = float(1 + (index % 5))
        machine_age = float(1 + (index % 24))
        region = regions[index % len(regions)]
        equipment_type = equipment[index % len(equipment)]
        base_level = 0.14 * hours
        age_effect = 0.04 * (machine_age > 12)
        region_effect = {"north": 0.02, "south": 0.08, "east": 0.04, "west": 0.01}[region]
        equipment_effect = {"pump": 0.03, "press": 0.06, "lathe": 0.02}[equipment_type]
        events = int((base_level + age_effect + region_effect + equipment_effect) > 0.25) + int(index % 19 == 0)
        rows.append((events, hours, machine_age, region, equipment_type))
    return rows


def main() -> None:
    """Run a local Spark workflow on generated sample data."""

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("glm-factor-optimizer-example")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        sdf = spark.createDataFrame(
            make_sample_rows(),
            ["events", "hours", "machine_age", "site_region", "equipment_type"],
        )
        workflow = SparkGLMWorkflow(
            target="events",
            family="poisson",
            exposure="hours",
            prediction="predicted_events",
            trials=3,
            n_prebins=5,
            min_bin_size=1.0,
            factor_kinds={
                "machine_age": "numeric",
                "site_region": "categorical",
            },
        )
        result = workflow.fit(sdf, ["machine_age", "site_region"])
        print("Selected factors:", result.selected_factors)
        result.validation_report["summary"].show(truncate=False)
        result.holdout_report["summary"].show(truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
