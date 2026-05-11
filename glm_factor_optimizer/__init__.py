"""Simple GLM factor-binning and modeling tools."""

from importlib.metadata import PackageNotFoundError, version as _package_version
from typing import TYPE_CHECKING

from .aggregation import aggregate_rate_table, aggregate_table
from .bins import apply_spec, category_target_order, make_numeric_bins
from .core import GLM, RateGLM
from .diagnostics import find_interactions, pair_diagnostics
from .factor import FactorBlock
from .metrics import (
    calibration,
    gaussian_deviance,
    gamma_deviance,
    lift_table,
    model_deviance,
    poisson_deviance,
    summary,
    weighted_mae,
    weighted_rmse,
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
from .study import GLMStudy
from .validation import by_factor_report, train_validation_comparison, validation_report
from .workflow import GLMWorkflow, WorkflowResult, run_workflow

if TYPE_CHECKING:
    from .spark.auto import SparkRateGLM as SparkRateGLM

optimize_bins = optimize_factor


def __getattr__(name: str):
    if name == "SparkRateGLM":
        from .spark.auto import SparkRateGLM as _SparkRateGLM

        globals()[name] = _SparkRateGLM
        return _SparkRateGLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    __version__ = _package_version("glm-factor-optimizer")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata.
    __version__ = "0.1.2"

__all__ = [
    "FittedRateGLM",
    "FittedGLM",
    "FactorBlock",
    "GLM",
    "GLMStudy",
    "GLMWorkflow",
    "OptimizationResult",
    "RateGLM",
    "RunLogger",
    "SparkRateGLM",
    "aggregate_rate_table",
    "aggregate_table",
    "apply_spec",
    "bin_count_penalty",
    "calibration",
    "category_target_order",
    "by_factor_report",
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
    "train_validation_comparison",
    "unstable_relativity_penalty",
    "stratified_sample",
    "validation_report",
    "weighted_mae",
    "weighted_rmse",
    "WorkflowResult",
    "__version__",
    "run_workflow",
]
