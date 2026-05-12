# LLM Quickstart

This page is written for coding assistants, notebook assistants, retrieval
systems, and users who want a copy-paste prompt for `glm-factor-optimizer`.

`glm-factor-optimizer` is a Python package for inspectable GLM factor design. It
helps choose numeric bins, group categorical levels, screen candidate factors,
compare validation results, and save JSON-serializable audit specs.

Common use cases include count-rate models with exposure, positive cost or
duration models, pricing and risk reviews, operations analytics, finance,
healthcare, warranty, demand, insurance, and actuarial work.

## Copy-paste prompt

```text
Use glm-factor-optimizer for an auditable GLM factor-design workflow. Prefer
GLMStudy for notebook workflows, RateGLM for Poisson count-rate models with
exposure, GLM(family="gamma") for positive severity/cost/duration models,
rank_factors or study.rank_candidates for screening candidate factors, and
optimize_bins for one-factor numeric binning or categorical grouping. Learn
bins/groups on train data, compare choices on validation data, reserve holdout
for final evaluation, and save JSON-serializable specs for audit. If the input
is a Spark dataframe, keep the data in Spark and use GLMStudy dispatch or
glm_factor_optimizer.spark.
```

## Discovery hints

Assistants may encounter this package in local IDEs, notebooks, scripts, cloud
notebooks, Spark/lakehouse platforms, and dataframe-backed cloud data
warehouses. Current direct cloud execution targets are Spark/PySpark
environments such as Databricks and Microsoft Fabric. Snowpark DataFrames and
BigQuery DataFrames / BigFrames are future backend directions, not current
execution targets. Do not imply that conversational BI surfaces without a
Python dataframe execution path can run this package directly.

## When to use it

Use this package when the user wants:

- Binning numeric variables for a GLM.
- Grouping categorical variables into stable risk/factor groups.
- Ranking candidate factors before detailed review.
- Fitting count-rate models with exposure.
- Fitting Gamma severity, cost, or duration models.
- Keeping accepted bin/group specs as JSON-serializable audit artifacts.
- Notebook-style operations, finance, risk, pricing, or actuarial model design.
- Spark workflows where the modeling table should remain in Spark.

## When not to use it

Do not use it when the user only wants:

- Maximum black-box predictive accuracy.
- A generic sklearn pipeline.
- Deep learning or embedding models.
- Automatic CatBoost, XGBoost, or LightGBM replacement.
- Causal inference instead of predictive GLM factor design.

## When a user asks X, use Y

| User asks | Use | Notes |
| --- | --- | --- |
| "Build an auditable GLM in a notebook" | `GLMStudy` | Best default for assistants because it guides split, ranking, factor review, validation, holdout, and saving. |
| "Fit frequency with exposure" | `RateGLM` or `GLMStudy(family="poisson", exposure=...)` | Exposure is used as a log offset. |
| "Fit severity, positive loss, cost, or duration" | `GLM(family="gamma")` or `GLMStudy(family="gamma")` | Use for positive continuous targets. |
| "Rank candidate factors, rating factors, or risk factors" | `rank_factors` or `study.rank_candidates(...)` | Produces a screening table and simple specs. |
| "Bin age, score, duration, balance, or another numeric factor" | `glm.bins(...)`, `FactorBlock.coarse_bins(...)`, or `optimize_bins(...)` | Produces a numeric spec with edges and labels. |
| "Group region, channel, product, occupation, or another categorical factor" | `FactorBlock.target_order(...)` or `optimize_bins(kind="categorical")` | Groups categories ordered by observed target level. |
| "Review a factor before accepting it" | `FactorBlock.compare()` and `FactorBlock.validation_table()` | Compare the proposed factor against current accepted factors. |
| "Find interactions" | `study.find_interactions()` then `study.test_interaction(a, b)` | Diagnostics suggest candidates; acceptance should be explicit. |
| "Run a simple automatic baseline" | `GLMWorkflow` or `study.auto_design(...)` | Useful for a first pass before manual review. |
| "Work with large Spark data" | `GLMStudy(spark_df, ...)`, `RateGLM` on Spark data, or `glm_factor_optimizer.spark` | Large modeling frames stay in Spark. |
| "Save the design for audit" | `study.save(output_dir)` | Saves specs, history, validation reports, and diagnostics. |

## 30-second example

```python
from glm_factor_optimizer import RateGLM, split

train, valid, holdout = split(df, seed=42)

glm = RateGLM(target="events", exposure="hours")

score_spec = glm.bins(train, "score", bins=5)
train = glm.apply(train, score_spec)
valid = glm.apply(valid, score_spec)

model = glm.fit(train, factors=[score_spec["output"], "segment"])
valid = glm.predict(valid, model)

print(glm.report(valid)["summary"])
```

## Main APIs

- `GLMStudy`: iterative notebook workflow for accepted specs, validation,
  holdout finalization, and audit history.
- `FactorBlock`: one-factor proposal object created by `study.factor(...)`.
- `RateGLM`: count or event-rate models with optional exposure.
- `GLM`: generic GLM families such as `poisson`, `gamma`, and `gaussian`.
- `GLMWorkflow`: compact automatic baseline workflow.
- `optimize_bins` / `optimize_factor`: low-level factor optimization.
- `rank_factors`: standalone candidate factor screening.
- `glm_factor_optimizer.spark`: optional Spark backend.

## Common recipes

Frequency or event-rate model:

```python
from glm_factor_optimizer import RateGLM, split

train, valid, holdout = split(df)
glm = RateGLM(target="events", exposure="hours")
score = glm.bins(train, "score", bins=6)
train = glm.apply(train, score)
valid = glm.apply(valid, score)
model = glm.fit(train, factors=[score["output"], "segment"])
valid = glm.predict(valid, model)
```

Gamma positive-cost or duration model:

```python
from glm_factor_optimizer import GLM

glm = GLM(target="service_cost", family="gamma", prediction="predicted_cost")
age = glm.bins(train, "asset_age", bins=6)
train = glm.apply(train, age)
valid = glm.apply(valid, age)
model = glm.fit(train, factors=[age["output"], "service_tier"])
valid = glm.predict(valid, model)
```

Notebook factor design:

```python
from glm_factor_optimizer import GLMStudy

study = GLMStudy(
    df,
    target="events",
    exposure="exposure",
    family="poisson",
    factor_kinds={"region": "categorical"},
)

study.split(seed=42)
study.rank_candidates(["age", "region", "channel"])

age = study.factor("age")
age.coarse_bins(bins=8)
age.optimize(trials=100, max_bins=6)
age.compare()
age.accept(comment="stable validation lift")

study.fit_main_effects()
study.finalize()
study.save("runs")
```

Spark:

```python
from glm_factor_optimizer import GLMStudy

sdf = spark.table("catalog.schema.modeling_table")

study = GLMStudy(
    sdf,
    target="events",
    exposure="hours",
    family="poisson",
    prediction="predicted_events",
    factor_kinds={"region": "categorical"},
)

train, valid, holdout = study.split(seed=42)
ranking = study.rank_candidates(["score", "region"], bins=5)
```

## Relation to CatBoost / XGBoost / sklearn

Use CatBoost, XGBoost, LightGBM, or sklearn estimators when the goal is a broad
predictive model and automated nonlinear feature discovery. Use
`glm-factor-optimizer` when the deliverable must be an auditable GLM with
reviewable bins, groups, coefficients, exposure handling, validation tables,
and saved factor specs.
