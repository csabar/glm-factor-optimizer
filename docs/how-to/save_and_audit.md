# How To Save and Audit a Study

Reviewed modeling work depends on reproducible decisions, not only fitted
models. `GLMStudy` records an audit history and saves the main artifacts.

## Record Comments During Modeling

```python
age = study.factor("machine_age")
age.optimize(trials=100)
age.accept(comment="Accepted five bins after validation check")

equipment = study.factor("equipment_type", kind="categorical")
equipment.optimize(trials=100)
equipment.reject(comment="Too sparse after grouping")
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

`RunLogger` can also mirror artifacts to MLflow when MLflow is installed. The
core package does not require it.

```python
from glm_factor_optimizer import RunLogger

logger = RunLogger("runs", name="count_rate_model", mlflow=True)
```

Use filesystem logging as the default in local notebooks. Use MLflow in
shared notebook or experiment-tracking environments.
