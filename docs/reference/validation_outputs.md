# Validation Outputs

Validation functions return pandas dataframes for notebook inspection, CSV
export, and saved run artifacts. Spark study workflows keep modeling tables in
Spark and collect only bounded aggregate report metadata as pandas dataframes.
Low-level `SparkGLM.report(...)` returns Spark DataFrames.

## Summary

Produced by:

- `summary(...)`
- `scored_summary(...)`
- `study.validation_report()["summary"]`
- `study.finalize()["summary"]`

Typical columns:

| Column | Meaning |
| --- | --- |
| `rows` | Number of rows. |
| `actual` | Sum of target. |
| `predicted` | Sum of prediction. |
| `actual_to_predicted` | Actual divided by predicted. |
| `deviance` | Family-specific mean deviance. |
| `exposure` | Total exposure when exposure is configured. |
| `actual_rate` | Actual divided by exposure. |
| `predicted_rate` | Predicted divided by exposure. |
| `actual_mean` | Actual mean when no exposure is configured. |
| `predicted_mean` | Predicted mean when no exposure is configured. |
| `mae` | Weighted mean absolute error. |
| `rmse` | Weighted root mean squared error. |

## Calibration

Produced by:

- `calibration(...)`
- `study.validation_report()["calibration"]`

Rows are grouped by prediction-level quantiles.

Typical columns:

- `bin`
- `actual`
- `predicted`
- `exposure` when configured
- `actual_rate` or `actual_mean`
- `predicted_rate` or `predicted_mean`
- `actual_to_predicted`

## Lift

Produced by:

- `lift_table(...)`
- `study.validation_report()["lift"]`

Contains calibration columns plus `lift`, comparing each bin's observed level
against the overall observed level.

## By-Factor Reports

Produced by:

- `by_factor_report(...)`
- `study.validation_report()["by_<factor>"]`

Typical columns:

- transformed factor value
- `rows`
- `bin_size`
- `actual`
- `predicted`
- rate or mean columns
- `actual_to_predicted`

## Train vs Validation

Produced by:

- `train_validation_comparison(...)`
- `study.validation_report()["train_validation"]`

Use this report to spot overfit binning or unstable factors.

## Model Versions

Produced by:

- `model_version_comparison(...)`
- `study.validation_report()["model_versions"]`

Typical columns:

- `version`
- `factors`
- `train_deviance`
- `validation_deviance`
- `validation_mae`
- `validation_rmse`

## Holdout

Holdout reports are produced by:

- `study.finalize()`
- `study.holdout_report()`

For reviewed models, treat holdout as final evaluation. Avoid using holdout
metrics during ordinary factor ranking or bin refinement.

