"""Simple GLM factor-binning and modeling tools."""

from .aggregation import aggregate_rate_table, aggregate_table
from .bins import apply_spec, category_risk_order, make_numeric_bins
from .core import GLM, RateGLM
from .diagnostics import find_interactions, pair_diagnostics
from .metrics import (
    calibration,
    gaussian_deviance,
    gamma_deviance,
    lift_table,
    model_deviance,
    poisson_deviance,
    summary,
)
from .model import FittedGLM, FittedRateGLM, fit_glm, fit_rate_glm
from .optimize import (
    OptimizationResult,
    optimize_factor,
    small_bin_size_penalty,
    small_count_penalty,
    train_validation_gap_penalty,
)
from .penalties import bin_count_penalty, small_target_penalty, unstable_relativity_penalty
from .runs import RunLogger
from .sampling import missing_strata, stratified_sample
from .screening import rank_factors
from .split import split
from .workflow import GLMWorkflow, WorkflowResult, run_workflow

optimize_bins = optimize_factor

__all__ = [
    "FittedRateGLM",
    "FittedGLM",
    "GLM",
    "GLMWorkflow",
    "OptimizationResult",
    "RateGLM",
    "RunLogger",
    "aggregate_rate_table",
    "aggregate_table",
    "apply_spec",
    "bin_count_penalty",
    "calibration",
    "category_risk_order",
    "fit_rate_glm",
    "fit_glm",
    "find_interactions",
    "gaussian_deviance",
    "gamma_deviance",
    "lift_table",
    "make_numeric_bins",
    "missing_strata",
    "model_deviance",
    "optimize_bins",
    "optimize_factor",
    "pair_diagnostics",
    "poisson_deviance",
    "rank_factors",
    "split",
    "small_bin_size_penalty",
    "small_count_penalty",
    "small_target_penalty",
    "summary",
    "train_validation_gap_penalty",
    "unstable_relativity_penalty",
    "stratified_sample",
    "WorkflowResult",
    "run_workflow",
]
