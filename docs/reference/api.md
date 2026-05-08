# Public API Reference

This page documents the intended public surface. Private helper functions are
not listed.

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

## `FactorBlock`

Import:

```python
from glm_factor_optimizer import FactorBlock
```

Usually created by:

```python
block = study.factor("machine_age")
```

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

`RateGLM` is a convenience Poisson model with required exposure.

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
- `fixed` or `fixed_factors`
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
from glm_factor_optimizer.spark import SparkGLM, SparkGLMWorkflow
```

The Spark backend is optional and imports PySpark lazily. Install with:

```bash
pip install "glm-factor-optimizer[spark]"
```
