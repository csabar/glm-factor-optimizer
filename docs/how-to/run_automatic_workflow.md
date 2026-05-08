# How To Run an Automatic Baseline Workflow

The package supports automatic workflows for baselines, benchmarks, and batch
experiments. The notebook `GLMStudy` workflow remains the recommended path for
production model design.

## Automatic Study Design

```python
from glm_factor_optimizer import GLMStudy

study = GLMStudy(
    df,
    target="events",
    exposure="hours",
    prediction="predicted_count",
    factor_kinds={"segment": "categorical", "region": "categorical"},
)

study.split(seed=42)
study.auto_design(
    ["score", "segment", "region", "age"],
    top_n=5,
    trials=50,
    accept_interactions=False,
)

study.validation_report()["summary"]
```

This calls the same ranking, factor optimization, and acceptance path that a
manual notebook would use.

## Sequential `GLMWorkflow`

For a smaller automatic wrapper:

```python
from glm_factor_optimizer import GLMWorkflow

workflow = GLMWorkflow(
    target="events",
    family="poisson",
    exposure="hours",
    prediction="predicted_count",
    factor_kinds={"segment": "categorical"},
    rank_candidates=True,
    top_n=5,
    trials=50,
)

result = workflow.fit(df, ["score", "segment", "region", "age"])
```

Use `GLMWorkflow` when you want a compact, stateless automatic result. Use
`GLMStudy` when you want notebook iteration, audit history, refines, and manual
accept/reject decisions.

## Saving Automatic Runs

```python
study.finalize()
study.save("runs")
```

or:

```python
workflow = GLMWorkflow(..., output_dir="runs")
result = workflow.fit(df, factors)
```

