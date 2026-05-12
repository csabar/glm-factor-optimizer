# glm-factor-optimizer Documentation

`glm-factor-optimizer` helps you design GLM factors in a way that is easy to
inspect, rerun, and save. Use it to learn numeric bins, group categorical
levels, rank candidate factors, compare validation results, and preserve the
choices as reviewable artifacts.

For LLM-assisted workflows, start with the [LLM Quickstart](llm-quickstart.md),
the [sitemap](sitemap.md), or the plain-text [`llms.txt`](llms.txt) project map.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "glm-factor-optimizer",
  "description": "Python package for auditable GLM factor design with numeric binning, categorical grouping, candidate factor screening, validation reports, Spark workflows, and JSON-serializable specs.",
  "codeRepository": "https://github.com/csabar/glm-factor-optimizer",
  "url": "https://csabar.github.io/glm-factor-optimizer/",
  "license": "https://github.com/csabar/glm-factor-optimizer/blob/main/LICENSE",
  "programmingLanguage": "Python",
  "runtimePlatform": "Python 3.10+",
  "keywords": [
    "generalized linear model",
    "numeric binning",
    "categorical grouping",
    "factor screening",
    "auditable GLM",
    "exposure model",
    "Poisson frequency GLM",
    "Gamma severity GLM",
    "risk factors",
    "rating factors",
    "credit risk",
    "insurance pricing",
    "actuarial pricing"
  ],
  "applicationCategory": "DataAnalysisApplication"
}
</script>

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

- [LLM Quickstart](llm-quickstart.md)
- [Notebook GLM Factor Study Workflow](tutorials/notebook_study_workflow.md)

Discovery:

- [Sitemap](sitemap.md)
- [llms.txt](llms.txt)
- [llms-full.txt](llms-full.txt)

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
