# glm-factor-optimizer Documentation

This documentation uses a Diataxis-style structure:

- **Tutorials** teach the main workflow from beginning to end.
- **How-to guides** solve focused modeling tasks.
- **Reference** documents public APIs and data structures.
- **Explanation** describes architecture and modeling principles.

The recommended professional workflow for this package is notebook-first:

1. Split data into train, validation, and holdout.
2. Rank coarse candidate factors.
3. Work factor by factor with `FactorBlock`.
4. Accept stable bins/groups explicitly.
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

Use `GLMWorkflow` when a simple automatic sequential workflow is enough:

```python
from glm_factor_optimizer import GLMWorkflow
```

Use `glm_factor_optimizer.spark` for Spark-oriented primitives.
