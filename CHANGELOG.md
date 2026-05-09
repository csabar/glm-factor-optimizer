# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

No unreleased changes yet.

## [0.1.1] - 2026-05-10

### Added

- Added `__version__` to the top-level package API.
- Added Spark-native dispatch for top-level `GLM` and `RateGLM` helpers:
  Spark DataFrame inputs now use Spark ML generalized linear regression for
  fitting and return Spark DataFrames for scoring/reporting workflows.
- Added Spark-native candidate factor screening through `RateGLM` and
  `SparkRateGLM`, including Spark DataFrame `identified_factors_` output.
- Added top-level `split(...)` support for Spark DataFrames.
- Added public documentation examples for manual Spark fitting and Spark-native
  factor screening.
- Added documentation guards so public examples do not recommend converting
  large Spark modeling tables to pandas.

### Changed

- `RateGLM` now accepts `target_col`, `exposure_col`, `weight_col`, and
  `prediction_col` aliases for notebook workflows that commonly use column
  suffix naming.
- Spark helper internals may use pandas only for bounded aggregate metadata,
  such as bin summaries or category-order tables; large modeling tables stay in
  Spark.
- Package metadata now targets version `0.1.1`.

### Fixed

- Fixed top-level `RateGLM.fit(spark_df, factors=[...])` so it no longer falls
  through to the pandas/statsmodels path.
- Fixed Spark screening validation so missing exposure or weight columns in
  validation data are caught before model fitting starts.

## [0.1.0] - 2026-05-09

### Added

- Initial PyPI release.
- Added pandas-first GLM factor binning, grouping, model fitting, Optuna-based
  one-factor bin optimization, factor screening, validation tables, diagnostics,
  run logging, and notebook-oriented study workflows.
- Added optional Spark backend primitives for Spark DataFrame fitting,
  binning/grouping, aggregation, optimization, and workflow experiments.
- Added project documentation, examples, CI, release workflow, and MkDocs
  publishing setup.
