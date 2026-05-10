# Public API Reference

The tables below cover the public API that is meant for notebooks, scripts, and
workflows. Private helper functions are not listed.

## `GLMStudy`

Import:

```python
from glm_factor_optimizer import GLMStudy
```

Constructor:

```python
GLMStudy(
    df,
    target,
    family="poisson",
    exposure=None,
    weight=None,
    prediction="predicted",
    factor_kinds=None,
    min_bin_size=100.0,
    seed=42,
)
```

`GLMStudy` accepts pandas and Spark DataFrames. Pandas inputs use the pandas
study implementation. Spark inputs dispatch to a Spark-backed study that keeps
raw, split, and scored modeling tables as Spark DataFrames while returning small
aggregate metadata tables for notebook inspection.

Primary methods:

| Method | Purpose |
| --- | --- |
| `split(...)` | Create train, validation, and holdout samples. |
| `rank_candidates(factors, ...)` | Screen candidate raw factors. |
| `factor(name, kind=None)` | Return a `FactorBlock`. |
| `accept(block_or_name, spec=None, comment=None)` | Accept a factor spec. |
| `reject(block_or_name, comment=None)` | Record a rejected proposal. |
| `fit_main_effects()` | Fit the current accepted model. |
| `validation_report()` | Return validation tables for the current model. |
| `holdout_report()` | Explicitly score holdout. |
| `refine_factor(name, ...)` | Re-optimize one accepted factor with other factors fixed. |
| `refine_all(...)` | Propose refinements for all accepted factors. |
| `find_interactions(...)` | Diagnose candidate pair interactions. |
| `test_interaction(a, b)` | Evaluate one coarse interaction. |
| `accept_interaction(a, b=None, comment=None)` | Accept a tested interaction or test and accept a pair. |
| `auto_design(factors, ...)` | Build an automatic baseline through the same acceptance path. |
| `finalize()` | Fit current model and score holdout. |
| `save(output_dir)` | Save specs, history, reports, and diagnostics. |

Important attributes:

| Attribute | Meaning |
| --- | --- |
| `specs` | Accepted raw factor specs keyed by raw factor name. |
| `interaction_specs` | Accepted interaction specs keyed by interaction output. |
| `accepted_raw_factors` | Raw factors accepted as main effects. |
| `selected_factors` | Transformed model columns used by the fitted model. |
| `ranking` | Latest candidate ranking dataframe. |
| `history` | Audit events. |
| `model_versions` | Stored scored model-version summaries. |
| `current_model` | Latest fitted `FittedGLM`. |
| `train_scored` | Latest scored train dataframe. |
| `validation_scored` | Latest scored validation dataframe. |
| `holdout_scored` | Holdout scored by `finalize()` or `holdout_report()`. |

For Spark studies, `train`, `validation`, `holdout`, and scored frame
attributes are Spark DataFrames. Ranking, comparison, report, and audit tables
are bounded pandas metadata.

## `FactorBlock`

Import:

```python
from glm_factor_optimizer import FactorBlock
```

Usually created by:

```python
block = study.factor("machine_age")
```

Spark studies return `SparkFactorBlock`, which mirrors the same notebook
methods while running factor operations on Spark.

Methods:

| Method | Purpose |
| --- | --- |
| `coarse_bins(bins=10, method="quantile")` | Create simple train-derived bins or groups. |
| `target_order(max_groups=None)` | Calculate categorical target order. |
| `set_spec(spec)` | Use a manual JSON-serializable spec. |
| `optimize(...)` | Run Optuna for this factor with accepted factors fixed. |
| `bin_table(sample="train")` | Inspect train, validation, or holdout bin sizes and actuals. |
| `validation_table()` | Return validation diagnostics by proposed output. |
| `compare()` | Compare current model against this proposed spec. |
| `accept(comment=None)` | Accept into the owning study. |
| `reject(comment=None)` | Record rejection in the owning study. |

## Low-Level Modeling

```python
from glm_factor_optimizer import GLM, RateGLM
```

`GLM` supports:

- `family="poisson"`
- `family="gamma"`
- `family="gaussian"`

`RateGLM` is a convenience wrapper for Poisson-style count/exposure models and
simple candidate-factor screening. Exposure is optional; when supplied, it is
used as a log offset.

Core methods:

- `fit(df, factors)`
- `predict(df, model)`
- `report(df, bins=10)`
- `bins(df, factor, bins=10, method="quantile")`
- `apply(df, spec)`
- `optimize(train_df, validation_df, factor, ...)`
- `rank(train_df, validation_df, factors, ...)`

## Optimization

```python
from glm_factor_optimizer import optimize_bins, optimize_factor
```

`optimize_bins` is an alias for `optimize_factor`.

Important arguments:

- `train_df`
- `validation_df`
- `target`
- `exposure`
- `factor`
- `kind`: `"numeric"` or `"categorical"`
- `family`
- `fixed_factors`
- `weight`
- `trials`
- `max_bins`
- `n_prebins`
- `min_bin_size`
- `penalties`

The objective uses validation deviance plus complexity and bin-size penalties.

## Screening

```python
from glm_factor_optimizer import rank_factors
```

Returns a dataframe with validation deviance improvement, missing rates,
coverage, screening p-value, bin counts, and the simple screening spec.

## Validation

```python
from glm_factor_optimizer import (
    validation_report,
    by_factor_report,
    train_validation_comparison,
    weighted_mae,
    weighted_rmse,
)
```

Use these for custom reports outside `GLMStudy`.

## Spark Backend

```python
from glm_factor_optimizer.spark import SparkGLM, SparkGLMStudy, SparkGLMWorkflow
```

The Spark backend is optional and imports PySpark lazily. Install with:

```bash
pip install "glm-factor-optimizer[spark]"
```

Top-level `GLM`, `RateGLM`, and `GLMStudy` dispatch on the input dataframe. With
Spark dataframes, manual fitting uses Spark ML generalized linear regression:

```python
from glm_factor_optimizer import RateGLM

glm = RateGLM(target_col="events", exposure_col="hours")
model = glm.fit(train_sdf, factors=["event_type", "region"])
scored = glm.predict(valid_sdf, model)
```

In long Spark Connect notebooks, call `model.release()` after a fitted model is
no longer needed. Study and optimization workflows release temporary Spark ML
models automatically.

Spark users can use the same study workflow without converting the modeling
table to pandas:

```python
from glm_factor_optimizer import GLMStudy

study = GLMStudy(
    spark.table("catalog.schema.modeling_table"),
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
ranking = study.rank_candidates(["machine_age", "region"], bins=5)

age = study.factor("machine_age")
age.coarse_bins(bins=5)
age.compare()
age.accept()

study.fit_main_effects()
report = study.validation_report()
final = study.finalize()
```

Spark study reports are aggregate metadata collected for display and saved run
artifacts; the large modeling frames remain Spark DataFrames.

For Spark-native factor screening, keep the data in Spark:

```python
from glm_factor_optimizer import RateGLM, split

train, valid, holdout = split(spark.table("catalog.schema.modeling_table"))

glm = RateGLM(
    candidate_factors=["event_type", "region", "service_channel"],
    target_col="events",
    exposure_col="hours",
    family="poisson",
    prediction_col="predicted_events",
)
glm.fit(train, validation_df=valid)

display(glm.identified_factors_)
scored_holdout = glm.predict(holdout)
```
