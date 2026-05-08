# Tutorial: Notebook AGLM Study Workflow

This tutorial shows the preferred workflow for professional model design:
interactive, auditable, and validation-driven.

The example uses count data with exposure, but the same pattern works for other
supported GLM families such as Gamma severity.

## 1. Create the Study

```python
from rate_glm_optimizer import GLMStudy

study = GLMStudy(
    df,
    target="claim_count",
    exposure="exposure",
    family="poisson",
    prediction="predicted_count",
    factor_kinds={
        "region": "categorical",
        "vehicle_type": "categorical",
    },
    min_bin_size=500.0,
    seed=42,
)
```

`GLMStudy` owns the modeling state:

- raw data
- train, validation, and holdout splits
- accepted factor specs
- accepted interaction specs
- model versions
- ranking results
- validation reports
- audit history

## 2. Split Once

```python
train, validation, holdout = study.split(
    train_fraction=0.6,
    validation_fraction=0.2,
    holdout_fraction=0.2,
    seed=42,
)
```

The package uses the validation sample for model design decisions. Holdout is
reserved for final evaluation and is not scored during ordinary ranking or
factor optimization.

## 3. Rank Candidate Factors

```python
candidate_factors = [
    "driver_age",
    "vehicle_age",
    "region",
    "vehicle_type",
    "bonus_malus",
]

ranking = study.rank_candidates(candidate_factors, bins=6, max_groups=6)
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

Ranking is only screening. It is not final variable selection. A good factor
should still pass human review for stability, interpretability, and business
reasonableness.

## 4. Build One Factor Block

```python
age = study.factor("driver_age", kind="numeric")

age.coarse_bins(bins=10)
age.bin_table()
```

For a categorical factor:

```python
region = study.factor("region", kind="categorical")
region.risk_order(max_groups=6)
region.bin_table()
```

## 5. Optimize and Compare

```python
age_result = age.optimize(
    trials=100,
    max_bins=6,
    n_prebins=12,
    min_bin_size=500.0,
)

age.compare()
age.validation_table()
```

`compare()` fits a candidate model using:

- all currently accepted factors fixed
- the proposed factor spec added or replacing the old spec
- validation deviance as the primary comparison score

## 6. Accept or Reject

```python
age.accept(comment="Stable monotone shape; accepted 5-bin driver_age factor")
```

Rejecting also records an audit event:

```python
region.reject(comment="Validation improvement too small")
```

Accepted specs become part of the current model design. Rejected proposals stay
in the study history.

## 7. Fit the Current Main-Effects Model

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

## 8. Refine With the Full Model Fixed

Once a baseline main-effects model exists, refine one accepted factor while all
other accepted factors remain fixed:

```python
refined_age = study.refine_factor(
    "driver_age",
    trials=200,
    max_bins=6,
    n_prebins=16,
)

refined_age.compare()
refined_age.accept(comment="Improved validation deviance with stable bins")
```

Or propose refinements for all accepted factors:

```python
proposals = study.refine_all(trials=50, accept=False)
```

Automatic acceptance is available, but notebook review is recommended.

## 9. Search for Interactions

```python
interactions = study.find_interactions(min_bin_size=500.0)
interactions.head(20)
```

Interactions are diagnostic candidates. They are not added automatically.

Test one interaction:

```python
test = study.test_interaction("driver_age", "vehicle_type")
test
```

Accept only if it is stable and explainable:

```python
study.accept_interaction("driver_age", "vehicle_type", comment="Known age x vehicle effect")
study.fit_main_effects()
```

## 10. Finalize on Holdout

```python
holdout_report = study.finalize()
holdout_report["summary"]
```

`finalize()` scores holdout and records a final audit event. Use it once the
model design is complete.

## 11. Save the Study

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

