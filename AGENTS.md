# AGENTS.md

Guidance for coding agents and LLM assistants working on
`glm-factor-optimizer`.

## Project purpose

`glm-factor-optimizer` is a Python package for auditable GLM factor design. It
helps users create numeric bins, categorical groups, candidate factor rankings,
validation reports, and JSON-serializable specs for compact GLM workflows.

Keep the public positioning broad. Lead with inspectable GLM factor design,
numeric binning, categorical grouping, factor screening, exposure-aware rate
models, Spark support, and audit artifacts. Insurance and actuarial examples
are valid examples, but they should not be the only or primary public framing.

## Main APIs

- Use `GLMStudy` for notebook-style iterative modeling and saved audit history.
- Use `FactorBlock` through `study.factor(...)` for one-factor proposals.
- Use `RateGLM` for Poisson count-rate models with optional exposure.
- Use `GLM(family="gamma")` for positive severity, cost, or duration models.
- Use `optimize_bins` / `optimize_factor` for one-factor optimization.
- Use `rank_factors` or `study.rank_candidates(...)` for candidate screening.
- Use `glm_factor_optimizer.spark` or top-level Spark dispatch for Spark
  dataframes.

## Documentation map

- `README.md`: public package overview, examples, and discovery language.
- `docs/llms.txt`: concise `/llms.txt`-style map for LLMs.
- `docs/llms-full.txt`: expanded plain-text LLM context.
- `docs/llm-quickstart.md`: assistant playbook, prompt, and recipes.
- `docs/sitemap.md`: human and LLM-readable documentation map.
- `docs/reference/api.md`: public API reference.
- `docs/reference/specs.md`: JSON-serializable spec contract.
- `docs/explanation/modeling_principles.md`: modeling discipline and audit
  philosophy.

## Development commands

Use the repo virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Focused documentation checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_docs.py::DocumentationTests::test_public_docs_do_not_use_legacy_terms tests\test_docs.py::DocumentationTests::test_public_examples_do_not_convert_spark_inputs_to_pandas -q
```

## Documentation style

- Keep examples domain-neutral by default: events, exposure, score, segment,
  region, service costs, incident rates, balances, utilization, and durations.
- Include insurance, actuarial pricing, claim frequency, and claim severity as
  search/discovery keywords, but place them alongside broader operations,
  finance, risk, healthcare, warranty, and demand examples.
- Avoid implying that the package is a replacement for CatBoost, XGBoost,
  LightGBM, deep learning, or generic sklearn pipelines.
- Avoid suggesting conversion of large Spark dataframes to pandas in public
  examples.
- Learn bins/groups on train data, compare on validation data, and reserve
  holdout for final evaluation.

## Branded environment names

Prefer generic phrasing such as local IDEs, notebooks, cloud notebooks,
Spark/lakehouse environments, and dataframe-backed cloud warehouses.

Only describe products as direct execution targets when there is a real Python
dataframe backend. Current direct cloud targets are Spark/PySpark environments
such as Databricks and Microsoft Fabric. Planned backend directions are
Snowpark DataFrames for Snowflake and BigQuery DataFrames / BigFrames for
BigQuery. Do not list conversational BI surfaces as direct execution
environments unless they expose a compatible Python dataframe path.
