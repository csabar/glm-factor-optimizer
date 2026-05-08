# Modeling Principles

This package is designed for controlled GLM model design. The goal is not to
maximize a validation metric at any cost. The goal is a stable, explainable,
auditable model.

## Train, Validation, Holdout

Use train data to learn bins, groupings, and model coefficients.

Use validation data to compare design choices:

- factor inclusion
- binning choices
- grouping choices
- interaction candidates
- penalty settings

Use holdout only for final evaluation. Avoid repeated holdout checks during
ordinary design work.

## One Factor at a Time

Iterative GLM factor-design workflows are useful because they keep model decisions inspectable.
Optimizing one factor at a time allows the modeler to ask:

- Does this factor improve validation performance?
- Are the bins stable?
- Are sparse bins penalized?
- Is the shape explainable?
- Does the result still make sense after other factors are included?

`GLMStudy.refine_factor()` supports this by re-optimizing a factor while all
other accepted factors remain fixed.

## Coarse First, Refine Later

A strong workflow is:

1. Create coarse bins/groups for screening.
2. Accept a simple stable factor.
3. Fit the model with accepted factors.
4. Refine factors inside the current full model.
5. Accept only improvements that are stable and explainable.

This avoids over-investing in a single factor before the rest of the model is
known.

## Validation Performance Is Not the Only Criterion

Optuna can find small metric improvements that are not worth accepting.
Professional review should also consider:

- number of bins
- minimum exposure, weight, or row count
- minimum target count
- train-validation gap
- stability of relativities
- monotonicity or shape reasonableness
- missing value behavior
- unseen category behavior
- operational simplicity

The package supports custom penalties so modelers can encode project-specific
stability rules.

## Categorical Grouping by Risk Ordering

High-cardinality categorical variables are converted into an ordered grouping
problem:

1. Calculate category risk using train data only.
2. Sort categories by observed risk.
3. Let cutpoints define groups along that order.
4. Apply the saved mapping to validation, holdout, and future data.

This makes categorical grouping searchable while keeping the final spec
inspectable.

## Interactions Are Candidates, Not Automatic Truth

Interaction diagnostics identify where the main-effects model may be missing a
pattern. They do not prove that an interaction should be added.

Accept interactions only after checking:

- sufficient exposure or weight in cells
- train and validation consistency
- coefficient stability
- practical interpretability
- whether the interaction duplicates an already accepted business rule

## Auditability

Every accepted modeling decision should be reproducible. `GLMStudy` records:

- accepted specs
- rejected proposals
- score before and after
- comments
- model versions
- validation reports
- holdout reports after finalization

This audit trail matters for model review, peer review, and future model
maintenance.
