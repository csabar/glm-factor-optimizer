# Tutorial: Notebook GLM Factor Study Workflow

In this notebook workflow, you rank candidate factors, optimize bins/groups,
accept a small model, test an interaction, and save the artifacts.

The example models incident counts per site operating hour. The same pattern
also works for defects per machine hour, orders per visit, support tickets per
active account, or positive continuous outcomes with a Gamma GLM.

## 1. Create Example Data

The example data is generated in the notebook.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = 800
hours = rng.uniform(0.5, 2.5, size=rows)
machine_age = rng.normal(6.0, 2.5, size=rows).clip(0.2, 15.0)
usage_score = rng.normal(size=rows)
site_region = rng.choice(["north", "south", "east", "west"], size=rows)
equipment_type = rng.choice(["standard", "compact", "heavy"], size=rows)

region_effect = pd.Series(site_region).map(
    {"north": -0.10, "south": 0.15, "east": 0.0, "west": 0.08}
).to_numpy()
equipment_effect = pd.Series(equipment_type).map(
    {"standard": 0.0, "compact": -0.15, "heavy": 0.25}
).to_numpy()
mean_rate = np.exp(
    -1.0
    + 0.25 * (machine_age > 8.0)
    + 0.35 * (usage_score > 0.7)
    + region_effect
    + equipment_effect
)
events = rng.poisson(mean_rate * hours)

df = pd.DataFrame(
    {
        "events": events,
        "hours": hours,
        "machine_age": machine_age,
        "usage_score": usage_score,
        "site_region": site_region,
        "equipment_type": equipment_type,
    }
)
```

## 2. Create the Study

```python
from glm_factor_optimizer import GLMStudy

study = GLMStudy(
    df,
    target="events",
    exposure="hours",
    family="poisson",
    prediction="predicted_count",
    factor_kinds={
        "site_region": "categorical",
        "equipment_type": "categorical",
    },
    min_bin_size=20.0,
    seed=42,
)
```

`GLMStudy` keeps track of the modeling state:

- raw data
- train, validation, and holdout splits
- accepted factor specs
- accepted interaction specs
- model versions
- ranking results
- validation reports
- audit history

## 3. Split Once

```python
train, validation, holdout = study.split(
    train_fraction=0.6,
    validation_fraction=0.2,
    holdout_fraction=0.2,
    seed=42,
)
```

Use validation for design decisions. Keep holdout for final evaluation; ordinary
ranking and factor optimization do not score it.

## 4. Rank Candidate Factors

```python
candidate_factors = [
    "machine_age",
    "usage_score",
    "site_region",
    "equipment_type",
]

ranking = study.rank_candidates(candidate_factors, bins=5, max_groups=4)
ranking[
    [
        "factor",
        "kind",
        "deviance_improvement",
        "p_value",
        "validation_missing_rate",
        "validation_measure_coverage",
        "bins",
        "min_bin_size",
    ]
].head(20)
```

Ranking is only screening, not final variable selection. Before accepting a
factor, check stability, interpretation, and whether the grouping will be usable
in practice.

## 5. Build One Factor Block

```python
age = study.factor("machine_age", kind="numeric")

age.coarse_bins(bins=5)
age.bin_table()
```

For a categorical factor:

```python
region = study.factor("site_region", kind="categorical")
region.target_order(max_groups=4)
region.bin_table()
```

## 6. Optimize and Compare

```python
age_result = age.optimize(
    trials=5,  # use 100+ for a real model review
    max_bins=5,
    n_prebins=8,
    min_bin_size=20.0,
)

age.compare()
age.validation_table()
```

`compare()` fits a candidate model using:

- all currently accepted factors fixed
- the proposed factor spec added or replacing the old spec
- validation deviance as the primary comparison score

## 7. Accept or Reject

```python
age.accept(comment="Five-bin machine_age proposal looked consistent")
```

Rejecting also records an audit event:

```python
region.reject(comment="Validation improvement too small")
```

Accepted specs become part of the current model design. Rejected proposals stay
in the study history.

## 8. Add Another Factor

```python
equipment = study.factor("equipment_type", kind="categorical")
equipment.optimize(trials=5, min_bin_size=20.0)  # use more trials in production
equipment.compare()
equipment.accept(comment="Equipment grouping looked consistent")
```

## 9. Fit the Current Main-Effects Model

```python
model = study.fit_main_effects()

report = study.validation_report()
report["summary"]
report["train_validation"]
report["model_versions"]
```

The validation report includes:

- overall summary
- calibration table
- lift table
- by-factor reports for accepted transformed factors
- train-vs-validation comparison
- model version comparison

## 10. Refine With the Full Model Fixed

Once a baseline main-effects model exists, refine one accepted factor while all
other accepted factors remain fixed:

```python
refined_age = study.refine_factor(
    "machine_age",
    trials=5,  # use 100+ for a real model review
    max_bins=5,
    n_prebins=8,
)

refined_age.compare()
refined_age.accept(comment="Full-model refinement accepted")
```

Or propose refinements for all accepted factors:

```python
proposals = study.refine_all(trials=5, accept=False)
```

You can set `accept=True`, but review proposals first for any model you intend
to keep.

## 11. Search for Interactions

```python
interactions = study.find_interactions(min_bin_size=20.0)
interactions.head(20)
```

Interactions are diagnostic candidates. They are not added automatically.

Test one interaction:

```python
test = study.test_interaction("machine_age", "equipment_type")
test
```

Accept only after checking that the pattern is consistent and explainable:

```python
study.accept_interaction(
    "machine_age",
    "equipment_type",
    comment="Accepted after checking interaction cells",
)
study.fit_main_effects()
```

## 12. Finalize on Holdout

```python
holdout_report = study.finalize()
holdout_report["summary"]
```

`finalize()` scores holdout and records a final audit event. Use it when the
model design is ready for final evaluation.

## 13. Save the Study

```python
run_path = study.save("runs")
run_path
```

Saved artifacts include:

- params
- accepted specs
- interaction specs
- factor ranking
- trial tables
- validation reports
- holdout reports
- coefficient table
- model versions
- audit history
