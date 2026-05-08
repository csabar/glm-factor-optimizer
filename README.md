# glm-factor-optimizer

Simple GLM tools for factor binning, grouping, model screening, and workflow
automation. The package is domain-free: it works for count-rate, positive
continuous, and other small GLM modeling problems.

Full project documentation is included in the source distribution under `docs/`.
The docs are organized as tutorials, how-to guides, reference, and explanation.

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
    fixed=["segment"],
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
score.accept(comment="stable score shape")

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

Spark Optuna optimization runs Optuna on the driver and Spark GLM jobs inside
each trial:

```python
result = glm.optimize(
    train_sdf,
    valid_sdf,
    "score",
    fixed=["segment"],
    trials=30,
    cache_input=True,
    cache_trials=False,
)
```

Install locally with Spark support using:

```bash
pip install "glm-factor-optimizer[spark]"
```

All binning and grouping specs are plain JSON-serializable dictionaries.

## Contributing

Development setup, test commands, coverage, and release notes are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
