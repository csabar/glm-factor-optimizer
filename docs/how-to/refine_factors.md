# How To Refine Factors With the Full Model Fixed

After you have a main-effects model, re-check important factors inside that
model. `GLMStudy.refine_factor()` re-optimizes one accepted factor while keeping
all other accepted factors fixed.

## Accept an Initial Factor

```python
age = study.factor("machine_age", kind="numeric")
age.optimize(trials=100, max_bins=6, n_prebins=12)
age.accept(comment="Initial machine_age factor")
```

Add more factors:

```python
equipment = study.factor("equipment_type", kind="categorical")
equipment.optimize(trials=100)
equipment.accept(comment="Equipment type grouping")

study.fit_main_effects()
```

## Refine One Factor

```python
refined_age = study.refine_factor(
    "machine_age",
    trials=200,
    max_bins=6,
    n_prebins=16,
)

refined_age.compare()
refined_age.bin_table()
refined_age.validation_table()
```

The refinement keeps every other accepted factor fixed. If the current accepted
factor is already in the model, the old version is removed from the fixed set
while the proposed replacement is evaluated.

## Accept or Leave as Proposal

```python
refined_age.accept(comment="Accepted full-model machine_age refinement")
```

If the proposal is not better or not stable:

```python
refined_age.reject(comment="Rejected because validation gain was too small")
```

## Refine All Factors

```python
proposals = study.refine_all(trials=50, accept=False)
```

This returns one `FactorBlock` per accepted raw factor. Review each block before
accepting. Automatic acceptance is available:

```python
study.refine_all(trials=50, accept=True)
```

Use automatic acceptance only for exploratory baselines or controlled batch
experiments.
