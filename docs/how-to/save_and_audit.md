# How To Save and Audit a Study

Professional pricing workflows need reproducible decisions, not only fitted
models. `GLMStudy` records an audit history and saves all major artifacts.

## Record Comments During Modeling

```python
age = study.factor("driver_age")
age.optimize(trials=100)
age.accept(comment="Accepted 5 bins; stable validation shape")

vehicle = study.factor("vehicle_type", kind="categorical")
vehicle.optimize(trials=100)
vehicle.reject(comment="Too sparse after grouping")
```

History rows include:

- timestamp
- action
- factor
- output column
- bin count
- score before
- score after
- comment
- JSON-serializable spec when available

## Finalize Before Saving Holdout Artifacts

```python
study.finalize()
```

This scores the holdout set and records a final audit event.

## Save

```python
run_path = study.save("runs")
```

The run directory contains:

- `params.json`
- `specs.json`
- `interaction_specs.json`
- `history.json`
- `factor_ranking.csv`
- `factor_ranking_specs.json`
- `model_versions.csv`
- `coefficients.csv`
- validation reports
- holdout reports
- Optuna trial tables
- interaction diagnostics when available

## Optional MLflow

The current `RunLogger` supports optional MLflow logging when installed. The
core package does not require MLflow.

```python
from rate_glm_optimizer import RunLogger

logger = RunLogger("runs", name="frequency_model", mlflow=True)
```

Use filesystem logging as the default in local notebooks. Use MLflow in
Databricks or managed experiment-tracking environments.

