# glm-factor-optimizer

[![CI](https://github.com/csabar/glm-factor-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/csabar/glm-factor-optimizer/actions/workflows/ci.yml)
[![Security](https://github.com/csabar/glm-factor-optimizer/actions/workflows/security.yml/badge.svg)](https://github.com/csabar/glm-factor-optimizer/actions/workflows/security.yml)
[![License](https://img.shields.io/github/license/csabar/glm-factor-optimizer)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Simple GLM tools for factor binning, grouping, model screening, and workflow
automation. The package is domain-free: it works for count-rate, positive
continuous, and other small GLM modeling problems.

Full project documentation is included in the source distribution under `docs/`
and published at
[csabar.github.io/glm-factor-optimizer](https://csabar.github.io/glm-factor-optimizer/).
The docs are organized as tutorials, how-to guides, reference, and explanation.
Release notes are maintained in [CHANGELOG.md](CHANGELOG.md).

Minimal runnable example:

```python
import pandas as pd

from glm_factor_optimizer import RateGLM, split

df = pd.DataFrame(
    [
        {
            "events": int((index % 7 == 0) + (index % 10 > 6)),
            "hours": 1.0 + 0.5 * (index % 4),
            "score": (index % 10) / 10,
            "segment": ["standard", "plus", "premium"][index % 3],
        }
        for index in range(60)
    ]
)

train, valid, _ = split(df, seed=42)

glm = RateGLM(target="events", exposure="hours")
score_spec = glm.bins(train, "score", bins=4)

train = glm.apply(train, score_spec)
valid = glm.apply(valid, score_spec)

model = glm.fit(train, factors=[score_spec["output"], "segment"])
valid = glm.predict(valid, model)

print(glm.report(valid)["summary"])
```

Use `RateGLM` for count-rate models:

- count target, like `events`
- exposure column, like `hours`
- numeric or categorical factors

```python
from glm_factor_optimizer import RateGLM, split

train, valid, holdout = split(df)

glm = RateGLM(target="events", exposure="hours")

score_spec = glm.bins(train, "score", bins=5)
train = glm.apply(train, score_spec)
valid = glm.apply(valid, score_spec)

model = glm.fit(train, factors=[score_spec["output"], "segment"])
valid = glm.predict(valid, model)

report = glm.report(valid)
print(report["summary"])
```

Use `GLM` for other families, such as Gamma cost or duration models:

```python
from glm_factor_optimizer import GLM

glm = GLM(target="severity", family="gamma", prediction="predicted_severity")

age_spec = glm.bins(train, "machine_age", bins=6)
train = glm.apply(train, age_spec)
valid = glm.apply(valid, age_spec)

model = glm.fit(train, factors=[age_spec["output"], "equipment_type"])
valid = glm.predict(valid, model)
```

Optimize one factor manually:

```python
result = glm.optimize(
    train,
    valid,
    "score",
    fixed_factors=["segment"],
    trials=50,
)

train = glm.apply(train, result.spec)
valid = glm.apply(valid, result.spec)
model = glm.fit(train, factors=[result.output, "segment"])
```

The same optimizer is also exposed as `optimize_bins`:

```python
from glm_factor_optimizer import optimize_bins

result = optimize_bins(
    train,
    valid,
    target="events",
    exposure="hours",
    factor="score",
)
```

Add custom penalties with lambdas or named functions:

```python
from glm_factor_optimizer import small_bin_size_penalty, small_count_penalty

result = glm.optimize(
    train,
    valid,
    "score",
    penalties={
        "small_count": small_count_penalty(min_count=5, penalty=0.02),
        "many_bins": lambda c: 0.01 * max(c["bin_count"] - 6, 0),
        "gap": lambda c: max(c["validation_deviance"] - c["train_deviance"], 0),
    },
)
```

Penalty callables receive a context dictionary with the selected `spec`, the
training `bin_table`, train/validation deviance, predictions, transformed
dataframes, factor name, kind, and fixed factors.

Rank candidate factors before detailed optimization:

```python
ranking = glm.rank(
    train,
    valid,
    ["score", "segment", "region"],
    factor_kinds={"segment": "categorical", "region": "categorical"},
)
print(ranking[["factor", "deviance_improvement"]])
```

Run a higher-level sequential workflow with optional ranking, logging, and
interaction diagnostics:

```python
from glm_factor_optimizer import GLMWorkflow

workflow = GLMWorkflow(
    target="events",
    family="poisson",
    exposure="hours",
    factor_kinds={"segment": "categorical"},
    trials=50,
    rank_candidates=True,
    top_n=5,
    interaction_diagnostics=True,
    output_dir="runs",
)

result = workflow.fit(df, factors=["score", "segment"])
print(result.validation_report["summary"])
print(result.coefficients)
```

For notebook-style iterative model design, use `GLMStudy`:

```python
from glm_factor_optimizer import GLMStudy

study = GLMStudy(
    df,
    target="events",
    exposure="hours",
    prediction="predicted_count",
    factor_kinds={"segment": "categorical"},
)

study.split(seed=42)
ranking = study.rank_candidates(["score", "segment", "region"])

score = study.factor("score")
score.coarse_bins(bins=10)
score.optimize(trials=100, max_bins=6)
score.compare()
score.accept(comment="score bins looked consistent")

study.fit_main_effects()
study.validation_report()

refined = study.refine_factor("score", trials=200)
refined.accept(comment="full-model refinement")

study.find_interactions()
study.finalize()
study.save("runs")
```

Useful helper modules are available for manual workflows:

```python
from glm_factor_optimizer.aggregation import aggregate_rate_table
from glm_factor_optimizer.diagnostics import find_interactions
from glm_factor_optimizer.runs import RunLogger
from glm_factor_optimizer.sampling import stratified_sample
```

Example synthetic datasets live under `examples/` and are not part of the
installable package API. The examples cover general event-rate, severity, and
Spark-style workflows across operational and service settings.

Use the optional Spark backend in PySpark environments:

```python
from glm_factor_optimizer.spark import SparkGLM, SparkGLMWorkflow

glm = SparkGLM(
    target="events",
    family="poisson",
    exposure="hours",
    prediction="predicted_count",
)

score_spec = glm.bins(train_sdf, "score", bins=8)
train_sdf = glm.apply(train_sdf, score_spec)
valid_sdf = glm.apply(valid_sdf, score_spec)

model = glm.fit(train_sdf, factors=[score_spec["output"], "segment"])
valid_sdf = glm.predict(valid_sdf, model)
```

The top-level notebook helpers also dispatch on Spark dataframes:

```python
from glm_factor_optimizer import RateGLM

glm = RateGLM(target_col="events", exposure_col="hours")
model = glm.fit(train_sdf, factors=["score", "segment"])
valid_sdf = glm.predict(valid_sdf, model)
```

For long Spark Connect sessions, call `model.release()` when a fitted model is
no longer needed. Study and optimization workflows release their temporary Spark
ML models automatically.

For notebook-style Spark study workflows, use the same `GLMStudy` entry point:

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

train, valid, holdout = study.split(
    train_fraction=0.6,
    validation_fraction=0.2,
    holdout_fraction=0.2,
    seed=42,
)
ranking = study.rank_candidates(["score", "region"], bins=5)

score = study.factor("score")
score.coarse_bins(bins=5)
score.compare()
score.accept(comment="Spark study baseline")

study.fit_main_effects()
study.validation_report()
study.finalize()
```

This keeps raw, split, and scored modeling tables in Spark. Study rankings,
comparisons, reports, and saved artifacts are bounded aggregate metadata for
notebook inspection.

Spark Optuna optimization runs Optuna on the driver and Spark GLM jobs inside
each trial:

```python
result = glm.optimize(
    train_sdf,
    valid_sdf,
    "score",
    fixed_factors=["segment"],
    trials=30,
    cache_input=True,
    cache_trials=False,
)
```

For a Spark-native factor screening workflow, keep the data in Spark:

```python
from glm_factor_optimizer import RateGLM, split

sdf = spark.table("catalog.schema.modeling_table")
train, valid, holdout = split(sdf)

candidate_factors = [
    "event_type",
    "region",
    "service_channel",
    "equipment_type",
]

glm = RateGLM(
    candidate_factors=candidate_factors,
    target_col="events",
    exposure_col="hours",
    family="poisson",
    prediction_col="predicted_events",
    top_n=10,
)
glm.fit(train, validation_df=valid)

display(glm.identified_factors_)
scored_holdout = glm.predict(holdout)
```

This path keeps the modeling table as a Spark dataframe and fits with Spark
ML generalized linear regression.

Install locally with Spark support using:

```bash
pip install "glm-factor-optimizer[spark]"
```

All binning and grouping specs are plain JSON-serializable dictionaries.

## Contributing

Development setup, test commands, coverage, and release notes are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
