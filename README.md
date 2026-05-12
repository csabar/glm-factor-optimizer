# glm-factor-optimizer

[![CI](https://github.com/csabar/glm-factor-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/csabar/glm-factor-optimizer/actions/workflows/ci.yml)
[![Security](https://github.com/csabar/glm-factor-optimizer/actions/workflows/security.yml/badge.svg)](https://github.com/csabar/glm-factor-optimizer/actions/workflows/security.yml)
[![License](https://img.shields.io/github/license/csabar/glm-factor-optimizer)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Design auditable GLM factors in Python: bin numeric variables, group categorical
levels, rank candidate factors, fit compact GLMs, and save the specs/reports
that explain the model. The package is domain-free: it works for count-rate,
positive continuous, and other tabular GLM problems where the final factor
design must be inspectable, reproducible, and easy to audit.

Full project documentation is included in the source distribution under `docs/`
and published at
[csabar.github.io/glm-factor-optimizer](https://csabar.github.io/glm-factor-optimizer/).
The docs are organized as tutorials, how-to guides, reference, and explanation.
Release notes are maintained in [CHANGELOG.md](CHANGELOG.md). See
[ROADMAP.md](ROADMAP.md) for planned development themes.

## Install

```bash
pip install glm-factor-optimizer
```

## What problem does this solve?

Many practical GLM projects are not only about fitting coefficients. Analysts
also need to decide how numeric factors should be binned, how categorical
levels should be grouped, which candidate variables are worth reviewing, and
how to preserve those choices for peer review, deployment, monitoring, and
audit.

`glm-factor-optimizer` focuses on that factor-design layer. It learns
JSON-serializable binning and grouping specs from train data, evaluates them on
validation data, and keeps the fitted model small enough to inspect. It is most
useful when the model has to be reviewed by people as well as scored by code.

## What you get

- Reusable factor specs with source columns, output columns, bin edges, category
  mappings, labels, and defaults for missing or unseen values.
- Validation reports for summary fit, calibration, lift, factor-level results,
  train/validation comparison, and holdout finalization.
- Audit artifacts such as `specs.json`, `history.json`, coefficients,
  validation CSVs, holdout CSVs, and optimization trial tables.
- Pandas workflows for local analysis and optional Spark workflows for large
  PySpark modeling tables.

## When to use it

- You need a GLM with explicit numeric bins, categorical groups, or interaction
  candidates.
- You want to model counts, rates, severities, costs, durations, balances, or
  other tabular outcomes with a transparent factor structure.
- You need train/validation/holdout discipline for accepted model-design
  decisions.
- You want to compare and save factor specs as plain dictionaries rather than
  opaque model objects.
- You work in notebooks, local IDEs, scripts, Spark, or lakehouse-style
  analytics environments and want a repeatable workflow that an analyst can
  inspect step by step.

Examples range from incident rates and service costs to demand, utilization,
credit risk, warranty, insurance pricing, and actuarial loss-cost work.

## When not to use it

- You only need maximum black-box predictive accuracy.
- You want an automatic replacement for CatBoost, XGBoost, LightGBM, or a deep
  learning model.
- Your data is unstructured text, images, audio, or high-dimensional embeddings.
- You need causal identification rather than predictive GLM factor design.
- Your workflow does not require inspectable bins, groups, coefficients, or
  saved audit artifacts.

## 30-second example

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

## Main APIs

| API | Use it for |
| --- | --- |
| `GLMStudy` | Notebook-style iterative factor design with accepted/rejected specs, validation reports, holdout finalization, and saved audit history. |
| `FactorBlock` | One-factor work inside a study: coarse bins, target-ordered groups, optimization, comparison, acceptance, and rejection. |
| `RateGLM` | Low-level Poisson count-rate models with optional exposure and convenience methods for bins, optimization, ranking, fitting, prediction, and reports. |
| `GLM` | Low-level GLM families such as `poisson`, `gamma`, and `gaussian`. |
| `GLMWorkflow` | Automatic baseline workflow for ranking, selecting, fitting, diagnostics, and logging. |
| `optimize_bins` / `optimize_factor` | Direct numeric binning or categorical grouping optimization for one factor. |
| `rank_factors` | Candidate factor screening before deeper review. |
| `glm_factor_optimizer.spark` | Optional Spark backend that keeps large modeling tables in Spark. |

## Choosing an API

| Task | Start with | Why |
| --- | --- | --- |
| Build an auditable notebook workflow | `GLMStudy` | Records accepted specs, comments, validation reports, holdout results, and history. |
| Fit frequency or event counts with exposure | `RateGLM` or `GLMStudy(family="poisson", exposure=...)` | Uses exposure as a log offset for Poisson count-rate models. |
| Fit positive cost, duration, or severity | `GLM(family="gamma")` or `GLMStudy(family="gamma")` | Supports positive continuous targets. |
| Find useful candidate factors | `rank_factors` or `study.rank_candidates(...)` | Screens candidates before deeper manual review. |
| Bin a numeric factor | `glm.bins(...)`, `FactorBlock.coarse_bins(...)`, or `optimize_bins(...)` | Produces reusable numeric binning specs. |
| Group high-cardinality categories | `FactorBlock.target_order(...)` or `optimize_bins(kind="categorical")` | Orders categories by observed target level and groups them into stable bands. |
| Compare a proposed factor to the current model | `FactorBlock.compare()` | Evaluates the proposed spec against accepted factors. |
| Keep large data in Spark | `GLMStudy` on a Spark dataframe or `glm_factor_optimizer.spark` | Avoids converting large modeling tables to pandas. |
| Save model design for audit | `study.save(...)` | Writes specs, history, reports, and diagnostics. |

## Relation to CatBoost / XGBoost / sklearn

`glm-factor-optimizer` is not a gradient boosting library and is not a generic
`sklearn` pipeline replacement. CatBoost, XGBoost, LightGBM, and many sklearn
estimators are strong choices when predictive accuracy and automated nonlinear
feature discovery matter most.

Use this package when the deliverable is an inspectable GLM factor design:
reviewable numeric bins, grouped categorical factors, explicit exposure
handling, GLM coefficients, validation tables, and JSON-serializable audit
specs. It can also complement boosting models: use a booster to benchmark
predictive headroom, then use `glm-factor-optimizer` to build a simpler model
that operations, finance, risk, pricing, or analytics teams can review and
maintain.

## LLM discovery

This repository includes LLM-oriented documentation for coding assistants,
notebook assistants, IDE chat, and retrieval systems that understand the
[`/llms.txt`](https://llmstxt.org/) convention:

- [docs/llms.txt](docs/llms.txt): concise project map for LLMs.
- [docs/llms-full.txt](docs/llms-full.txt): expanded assistant context.
- [docs/llm-quickstart.md](docs/llm-quickstart.md): problem-to-API mapping,
  copy-paste prompts, and common recipes.
- [docs/sitemap.md](docs/sitemap.md): compact map of human and LLM-readable
  documentation.

Copy-paste prompt for AI assistants:

```text
Use the Python package glm-factor-optimizer to design an auditable GLM factor
model. Prefer GLMStudy for notebook workflows, RateGLM for Poisson count-rate
models with exposure, GLM(family="gamma") for positive severity or cost
models, optimize_bins for one-factor binning/grouping, and rank_factors for
candidate screening. Keep bins/groups as JSON-serializable specs learned on
train data, compare on validation data, reserve holdout for final evaluation,
and use Spark APIs when the input is a Spark dataframe.
```

## Common recipes

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

glm = GLM(target="service_cost", family="gamma", prediction="predicted_cost")

age_spec = glm.bins(train, "asset_age", bins=6)
train = glm.apply(train, age_spec)
valid = glm.apply(valid, age_spec)

model = glm.fit(train, factors=[age_spec["output"], "service_tier"])
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

For large Spark tables with an ordered time split, exact splitting remains the
default. Use `time_split="approximate"` to avoid a global row-number window when
approximate fraction boundaries are acceptable.

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
