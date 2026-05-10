import os

from databricks.connect import DatabricksSession
from pyspark.sql.functions import abs, col, lit, when

from glm_factor_optimizer import GLMStudy, RateGLM, split
from glm_factor_optimizer.spark import SparkGLMStudy


def main() -> None:
    print("1. Connecting to Databricks Serverless")
    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    print("   spark.range preflight:", spark.range(3).count())

    print("2. Loading sample table and preparing positive exposure")
    df = spark.table("samples.healthverity.claims_sample_synthetic")
    df = df.withColumn("event", when(col("line_charge").cast("double") > 0, 1).otherwise(0))
    units = abs(col("procedure_units_billed").cast("double"))
    df = df.withColumn("exposure", when(units > 0, units).otherwise(lit(1.0)))
    df = df.select(
        "event",
        "exposure",
        "claim_type",
        "patient_gender",
        "patient_state",
        "payer_type",
    )
    rows = df.count()
    print("   rows:", rows)
    exposure_issues = df.where((col("exposure").isNull()) | (col("exposure") <= 0)).count()
    assert exposure_issues == 0, f"exposure must be positive, found {exposure_issues} bad rows"

    factors = ["claim_type", "patient_gender", "patient_state", "payer_type"]

    print("3. Testing split aliases")
    train, valid, holdout = split(
        df,
        train_fraction=0.6,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=42,
    )
    print("   split counts:", train.count(), valid.count(), holdout.count())

    print("4. Testing RateGLM automatic Spark screening")
    glm = RateGLM(
        candidate_factors=factors,
        target_col="event",
        exposure_col="exposure",
        family="poisson",
        prediction_col="predicted_events",
        top_n=2,
    )
    glm.fit(train, validation_df=valid)
    assert glm.selected_factors_, "RateGLM should select at least one factor"
    print("   selected:", glm.selected_factors_)
    glm.identified_factors_.show()
    if glm.model_ is not None:
        glm.model_.release()

    print("5. Testing GLMStudy Spark dispatch and candidate ranking")
    study = GLMStudy(
        df,
        target="event",
        exposure="exposure",
        family="poisson",
        prediction="predicted_events",
        factor_kinds={factor: "categorical" for factor in factors},
        min_bin_size=100.0,
        seed=42,
    )
    assert isinstance(study, SparkGLMStudy), type(study)
    study.split(seed=42)
    ranking = study.rank_candidates(factors, bins=3, max_groups=3)
    print(ranking[["factor", "deviance_improvement", "error"]])
    assert ranking["error"].isna().all(), ranking[["factor", "error"]]

    print("6. Testing FactorBlock compare/accept")
    block = study.factor("claim_type")
    block.coarse_bins()
    comparison = block.compare()
    print(comparison)
    assert "candidate_validation_deviance" in comparison.columns
    block.accept()

    if os.environ.get("DBX_SMOKE_OPTIMIZE") == "1":
        print("7. Testing one-trial Spark optimization with Serverless cache fallback")
        optimized = study.factor("patient_state", kind="categorical")
        result = optimized.optimize(trials=1, max_bins=3, cache_input=True)
        print("   optimized:", result.output, result.validation_deviance)
    else:
        print("7. Skipping Optuna optimization; set DBX_SMOKE_OPTIMIZE=1 to test it")

    print("8. Testing fit/report/finalize")
    study.fit_main_effects()
    validation_summary = study.validation_report()["summary"]
    holdout_summary = study.finalize()["summary"]
    print(validation_summary)
    print(holdout_summary)
    assert len(validation_summary) == 1
    assert len(holdout_summary) == 1

    print("Databricks Spark Connect smoke passed.")
    spark.stop()


if __name__ == "__main__":
    main()
