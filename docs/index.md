# glm-factor-optimizer Documentation

`glm-factor-optimizer` helps you design GLM factors in a way that is easy to
inspect, rerun, and save. Start with the tutorial if you want the full notebook
flow, or jump to a how-to guide when you need a specific task.

A typical modeling session looks like this:

1. Split data into train, validation, and holdout.
2. Rank coarse candidate factors.
3. Work factor by factor with `FactorBlock`.
4. Accept bins/groups deliberately after inspection.
5. Fit the current main-effects model.
6. Refine accepted factors with the full model fixed.
7. Search and test candidate interactions.
8. Finalize once on holdout.
9. Save specs, history, metrics, and validation tables.

## Documentation Map

Tutorial:

- [Notebook GLM Factor Study Workflow](tutorials/notebook_study_workflow.md)

How-to guides:

- [Rank and Screen Candidate Factors](how-to/rank_candidate_factors.md)
- [Refine Factors With the Full Model Fixed](how-to/refine_factors.md)
- [Test and Accept Interactions](how-to/test_interactions.md)
- [Run an Automatic Baseline Workflow](how-to/run_automatic_workflow.md)
- [Save and Audit a Study](how-to/save_and_audit.md)

Reference:

- [Public API Reference](reference/api.md)
- [Binning and Grouping Specs](reference/specs.md)
- [Validation Outputs](reference/validation_outputs.md)

Explanation:

- [Architecture](explanation/architecture.md)
- [Modeling Principles](explanation/modeling_principles.md)

## Main Interfaces

Use `GLMStudy` for iterative notebook modeling:

```python
from glm_factor_optimizer import GLMStudy

study = GLMStudy(
    df,
    target="events",
    exposure="hours",
    prediction="predicted_count",
)
```

Use `RateGLM` or `GLM` for low-level manual work:

```python
from glm_factor_optimizer import RateGLM

glm = RateGLM(target="events", exposure="hours")
```

Use `GLMWorkflow` for compact batch baselines:

```python
from glm_factor_optimizer import GLMWorkflow
```

Use `glm_factor_optimizer.spark` for Spark-oriented primitives.
