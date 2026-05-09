# Architecture

`glm-factor-optimizer` keeps the low-level pieces visible. You can use the
automatic workflows, or call the same binning, fitting, scoring, and logging
functions yourself.

## Layer 1: Specs and Model Primitives

Core modules:

- `bins.py`
- `model.py`
- `metrics.py`
- `optimize.py`
- `penalties.py`

Responsibilities:

- create JSON-serializable numeric and categorical specs
- apply specs to dataframes
- fit GLMs
- score predictions
- optimize one factor at a time
- define reusable optimization penalties

Core mechanics live here.

## Layer 2: Analysis Helpers

Helper modules:

- `screening.py`
- `aggregation.py`
- `sampling.py`
- `diagnostics.py`
- `validation.py`
- `runs.py`

Responsibilities:

- rank candidate factors
- create representative samples
- aggregate model development tables
- find candidate interactions
- produce validation reports
- save run artifacts

Use them in notebooks or as building blocks for automatic workflows.

## Layer 3: Notebook Study

Main modules:

- `study.py`
- `factor.py`

Responsibilities:

- own split train, validation, and holdout samples
- manage accepted and rejected factor specs
- keep audit history
- compare candidate bins to the current model
- refine factors with the current full model fixed
- test interactions explicitly
- finalize and save model artifacts

Use `GLMStudy` when you want to design factors step by step and keep a record of
the choices you made.

## Layer 4: Automatic Workflow

Main module:

- `workflow.py`

`GLMWorkflow` is a compact automatic wrapper for baselines, experiments, and
examples. For close review, use `GLMStudy` instead.

## Layer 5: Spark Backend

Spark modules live under:

- `glm_factor_optimizer.spark`

Spark support lives in its own package namespace, so pandas users can install
the core package without PySpark. Spark imports are lazy, and the same JSON
specs can be used with Spark dataframes and Spark GLM jobs.

## Data Flow

A common flow is:

```text
raw dataframe
  -> split
  -> rank candidate raw factors
  -> propose factor spec
  -> apply spec to train and validation
  -> fit model with accepted factors fixed
  -> score validation
  -> accept or reject
  -> fit main effects
  -> refine accepted factors
  -> test interactions
  -> finalize on holdout
  -> save artifacts
```

## Why Specs Are JSON Dictionaries

Specs are simple dictionaries so they are:

- easy to inspect in notebooks
- easy to write to JSON
- independent of Python class versions
- portable to Spark or other execution engines
- suitable for audit review

That keeps saved specs readable and portable.
