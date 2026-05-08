# How To Rank and Screen Candidate Factors

Use ranking to identify factors worth deeper review. Ranking is a screening
step, not final variable selection.

## With `GLMStudy`

```python
study.split(seed=42)

ranking = study.rank_candidates(
    ["score", "age", "segment", "region"],
    bins=6,
    max_groups=6,
)

ranking.sort_values("deviance_improvement", ascending=False).head(20)
```

Useful columns:

- `deviance_improvement`: validation deviance improvement versus intercept-only
- `relative_improvement`: improvement divided by baseline validation deviance
- `p_value`: chi-square style screening p-value when SciPy is available
- `train_missing_rate`
- `validation_missing_rate`
- `train_measure_coverage`
- `validation_measure_coverage`
- `bins`
- `min_bin_size`
- `small_bins`

## With the Low-Level API

```python
from rate_glm_optimizer import rank_factors

ranking = rank_factors(
    train,
    valid,
    target="events",
    exposure="hours",
    factors=["score", "segment", "region"],
    factor_kinds={"segment": "categorical", "region": "categorical"},
)
```

## Recommended Review Rules

Prefer factors that:

- improve validation deviance materially
- have stable train and validation behavior
- cover enough exposure, weight, or rows
- have low missing rate or a meaningful missing group
- can be explained to model reviewers
- avoid too many bins for a tiny improvement

Do not accept factors only because the screening p-value is small. In large
datasets, tiny effects can be statistically significant but not useful.

