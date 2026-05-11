# Roadmap

`glm-factor-optimizer` is an audit-first GLM factor design toolkit. The project
helps users discover useful signal, turn it into transparent factor structures,
compare alternatives repeatedly, and save artifacts that can be reviewed later.

This roadmap is intentionally schema-neutral. Public examples and documentation
should use synthetic data, generic column names, and aggregate outputs only.

## Guiding Principles

- Keep the core package domain-free and focused on GLM factor design.
- Prefer readable JSON specs, notebook-friendly tables, and bounded aggregate
  artifacts over opaque model objects.
- Use challenger models as advisors for signal discovery, not as replacements
  for auditable GLM workflows.
- Keep Spark support native where possible, without requiring large modeling
  tables to be converted to pandas.
- Treat examples and tutorials as public artifacts: they should be synthetic,
  reusable, and free of private schema details.

## v0.2: Workflow Clarity and Repeatability

Planned focus:

- Add a synthetic two-stage count and severity tutorial.
- Document exposure, weights, filtered positive-target samples, and combined
  expected-value scoring.
- Add repeated benchmark utilities for multi-seed workflow comparison.
- Add missingness, coverage, cardinality, and split-drift diagnostics for
  candidate factors.
- Continue Spark and Databricks hardening around unsupported caching, temporary
  model cleanup, and aggregate-only reporting.

Expected outcome:

- Users can run a clear end-to-end public example, compare close model choices
  across repeated splits, and see whether candidate features are safe enough to
  test before optimization.

## v0.3: Challenger-Assisted Model Design

Planned focus:

- Add an optional `catboost` extra for signal discovery.
- Add challenger fit and comparison helpers.
- Add feature-importance, interaction-suggestion, and advisor reports that can
  be exported as Markdown and JSON.
- Keep challenger APIs outside the core optimizer path and gracefully unavailable
  when optional dependencies are not installed.

Expected outcome:

- Users can use stronger black-box challengers to find signal, then translate
  the useful parts into auditable GLM factor structures.

## v0.4+: Richer Transparent Modeling Tools

Planned focus:

- Add structured interaction search with cardinality-aware pruning.
- Add numeric split-candidate and monotonicity diagnostics.
- Explore optional spline or semi-parametric numeric-effect experiments.
- Add safer controls for high-cardinality categorical factors.
- Bring repeated benchmarking and diagnostics to Spark where feasible with
  bounded metadata outputs.

Expected outcome:

- Users get richer design assistance while keeping the final model structure
  inspectable, portable, and suitable for review.

## Non-Goals

- Do not add project-specific pricing workflows to the core package.
- Do not bundle country-specific external datasets.
- Do not add raw geospatial ingestion to the optimizer core.
- Do not automatically adopt external feature bundles without coverage and drift
  diagnostics.
- Do not turn tree-based challengers into the default production modeling path.
