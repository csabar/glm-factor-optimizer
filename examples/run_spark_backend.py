"""Databricks/Spark backend sketch.

Run this inside a PySpark environment, for example a Databricks notebook or a
local environment with `pip install "rate-glm-optimizer[spark]"`.
"""

from __future__ import annotations

from rate_glm_optimizer.spark import SparkGLM, split


def main(sdf) -> None:
    train_sdf, valid_sdf, holdout_sdf = split(sdf, seed=42)
    glm = SparkGLM(
        target="events",
        family="poisson",
        exposure="hours",
        prediction="predicted_count",
    )

    result = glm.optimize(
        train_sdf,
        valid_sdf,
        "score",
        fixed=["segment"],
        trials=30,
        n_prebins=8,
        min_bin_size=1_000.0,
        cache_input=True,
    )

    train_sdf = glm.apply(train_sdf, result.spec)
    valid_sdf = glm.apply(valid_sdf, result.spec)
    holdout_sdf = glm.apply(holdout_sdf, result.spec)

    model = glm.fit(train_sdf, factors=[result.output, "segment"])
    scored_holdout = glm.predict(holdout_sdf, model)
    glm.report(scored_holdout)["summary"].show()
