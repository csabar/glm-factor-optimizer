# Modeling Principles

Good GLM factor design is not just a search for the lowest validation score.
Accepted bins, groups, and interactions work best when they are easy to inspect,
reproduce, and explain.

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

Optimizing one factor at a time keeps the review manageable. For each proposal,
ask:

- Does this factor improve validation performance?
- Are the bins stable?
- Are sparse bins penalized?
- Is the shape explainable?
- Does the result still make sense after other factors are included?

`GLMStudy.refine_factor()` re-optimizes one factor while all other accepted
factors remain fixed.

## Coarse First, Refine Later

A practical workflow is:

1. Create coarse bins/groups for screening.
2. Accept a simple factor that behaves consistently.
3. Fit the model with accepted factors.
4. Refine factors inside the current full model.
5. Accept only improvements that are consistent and explainable.

This avoids over-investing in a single factor before the rest of the model is
known.

## Validation Performance Is Not the Only Criterion

Optuna can find small metric improvements that are not worth accepting.
Also check:

- number of bins
- minimum exposure, weight, or row count
- minimum target count
- train-validation gap
- stability of fitted level changes
- monotonicity or shape plausibility
- missing value behavior
- unseen category behavior
- operational simplicity

Use custom penalties to encode project-specific stability rules.

## Categorical Grouping by Observed Target Level

High-cardinality categorical variables are converted into an ordered grouping
problem:

1. Calculate each category's observed target level using train data only.
2. For exposure models, use actual target divided by exposure.
3. For weighted models, use weighted mean target.
4. Otherwise, use the category mean target.
5. Sort categories by that observed target level.
6. Let cutpoints define groups along that order.
7. Apply the saved mapping to validation, holdout, and future data.

The optimizer can then search over group cutpoints while the saved spec remains
readable.

## Interactions Are Candidates, Not Decisions

Interaction diagnostics show where the main-effects model may be missing a
pattern. Treat them as leads, not decisions.

Accept interactions only after checking:

- sufficient exposure or weight in cells
- train and validation consistency
- coefficient stability
- practical interpretability
- whether the interaction duplicates an already accepted modeling rule

## Auditability

`GLMStudy` records the pieces used to reproduce accepted modeling decisions:

- accepted specs
- rejected proposals
- score before and after
- comments
- model versions
- validation reports
- holdout reports after finalization

The saved history makes later review and model maintenance much easier.
