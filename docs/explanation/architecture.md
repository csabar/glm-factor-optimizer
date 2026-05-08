# Architecture

`glm-factor-optimizer` is organized in layers. Each layer is useful on its own,
but the preferred professional workflow is built from the lower layers instead
of hiding them.

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

This layer is intentionally small and domain-free.

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

These modules support both notebook modeling and automatic workflows.

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

`GLMStudy` is the recommended interface for iterative GLM factor design because it
keeps the process interactive and reviewable.

## Layer 4: Automatic Workflow

Main module:

- `workflow.py`

`GLMWorkflow` is a compact automatic wrapper. It is useful for baselines,
experiments, and examples. It is not meant to replace the notebook workbench for
professional model design.

## Layer 5: Spark Backend

Spark modules live under:

- `glm_factor_optimizer.spark`

The Spark backend is separate so pandas users do not need PySpark installed.
Spark imports are lazy. The long-term design is to keep the same conceptual
objects and specs while allowing Spark dataframes and Spark GLM jobs in
Spark-compatible environments.

## Data Flow

The standard flow is:

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

Specs are simple dictionaries because they need to be:

- easy to inspect in notebooks
- easy to write to JSON
- independent of Python class versions
- portable to Spark or other execution engines
- suitable for audit review

This is more durable than storing opaque transformer objects.
