# How To Test and Accept Interactions

Interaction diagnostics identify factor pairs where actual-vs-predicted
patterns may not be captured by the main-effects model.

## Fit a Main-Effects Model First

```python
study.fit_main_effects()
```

Interactions use accepted transformed factors, such as `machine_age_bin` or
`equipment_type_group`.

## Find Candidates

```python
candidates = study.find_interactions(min_bin_size=500.0)
candidates.head(20)
```

Important columns:

- `pair`
- `train_mean_abs_deviation`
- `validation_mean_abs_deviation`
- `score`
- `cells_train`
- `cells_validation`
- `total_bin_size_train`
- `total_bin_size_validation`

The diagnostic requires a pattern to appear in both train and validation.

## Test One Interaction

You can use raw factor names if they have accepted specs:

```python
result = study.test_interaction("machine_age", "equipment_type")
result
```

Or pass transformed factor names directly:

```python
result = study.test_interaction("machine_age_bin", "equipment_type_group")
```

The interaction is built as a coarse categorical cross of the accepted
transformed factor columns.

## Accept Explicitly

```python
study.accept_interaction(
    "machine_age",
    "equipment_type",
    comment="Known machine_age x equipment_type effect",
)

study.fit_main_effects()
```

Do not accept an interaction only because the validation metric improves. Check:

- exposure or weight per interaction cell
- train/validation consistency
- coefficient plausibility
- operational interpretation
- stability over time if a period column is available

## Automatic Interaction Acceptance

`auto_design(..., accept_interactions=True)` can accept top diagnostics, but the
safer path for reviewed models is manual acceptance in a notebook.

