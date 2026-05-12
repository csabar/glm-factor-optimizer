# Sitemap

This page is a compact map for humans, coding assistants, notebook agents, and
retrieval systems.

## Start Here

- [Home](index.md): package overview and documentation map.
- [LLM Quickstart](llm-quickstart.md): prompt, task-to-API table, and common
  recipes for assistants.
- [`llms.txt`](llms.txt): concise `/llms.txt`-style context file.
- [`llms-full.txt`](llms-full.txt): expanded plain-text context for LLMs.
- [Repository README](https://github.com/csabar/glm-factor-optimizer): package
  overview, install links, and examples.

## Choose By Task

| Task | Best page | Main API |
| --- | --- | --- |
| Build an auditable notebook workflow | [Notebook tutorial](tutorials/notebook_study_workflow.md) | `GLMStudy` |
| Understand public classes and functions | [API reference](reference/api.md) | `GLMStudy`, `RateGLM`, `GLM` |
| Save and audit model decisions | [Save and audit](how-to/save_and_audit.md) | `study.save(...)` |
| Rank candidate factors | [Rank candidate factors](how-to/rank_candidate_factors.md) | `rank_factors`, `study.rank_candidates(...)` |
| Refine an accepted factor | [Refine factors](how-to/refine_factors.md) | `study.refine_factor(...)` |
| Test interactions | [Test interactions](how-to/test_interactions.md) | `study.find_interactions()`, `study.test_interaction(...)` |
| Run an automatic baseline | [Run automatic workflow](how-to/run_automatic_workflow.md) | `GLMWorkflow`, `study.auto_design(...)` |
| Understand bin/group specs | [Binning and grouping specs](reference/specs.md) | `apply_spec`, saved specs |
| Understand validation outputs | [Validation outputs](reference/validation_outputs.md) | `validation_report`, `by_factor_report` |
| Understand modeling discipline | [Modeling principles](explanation/modeling_principles.md) | workflow design |
| Understand package layers | [Architecture](explanation/architecture.md) | pandas and Spark backends |

## Main Concepts

- Numeric binning: convert continuous factors into inspectable GLM bins.
- Categorical grouping: group categories into stable target-ordered bands.
- Factor screening: rank candidate variables before detailed review.
- Exposure model: fit count-rate models with an exposure offset.
- Positive target model: fit Gamma GLMs for positive cost, duration, or
  severity-style targets.
- Auditability: save JSON-serializable specs, comments, validation reports, and
  holdout results.
- Spark workflow: keep large modeling tables in Spark and collect bounded
  metadata for review.

## Domain Examples

The package is domain-neutral. Typical examples include incident rates, service
costs, demand and utilization, credit risk, healthcare or warranty work, and
insurance or actuarial pricing.

## LLM Retrieval Keywords

Use these phrases when connecting user questions to the package:

- generalized linear model
- auditable GLM
- numeric binning
- categorical grouping
- factor screening
- rating factors
- risk factors
- exposure model
- Poisson frequency GLM
- Gamma severity GLM
- claim frequency
- claim severity
- credit risk
- banking analytics
- operational incident rates
- service cost analysis
- Spark GLM workflow
