"""Optional PySpark backend for Spark environments."""

from .aggregation import aggregate_rate_table, aggregate_table
from .auto import RateGLM, SparkRateGLM
from .bins import apply_spec, category_target_order, make_categorical_groups, make_numeric_bins
from .core import SparkGLM
from .metrics import calibration, model_deviance, summary
from .model import FittedSparkGLM, fit_glm
from .optimize import SparkOptimizationResult, optimize_factor
from .split import split
from .workflow import SparkGLMWorkflow, SparkWorkflowResult

__all__ = [
    "FittedSparkGLM",
    "SparkGLM",
    "SparkGLMWorkflow",
    "SparkOptimizationResult",
    "SparkRateGLM",
    "SparkWorkflowResult",
    "RateGLM",
    "aggregate_rate_table",
    "aggregate_table",
    "apply_spec",
    "calibration",
    "category_target_order",
    "fit_glm",
    "make_categorical_groups",
    "make_numeric_bins",
    "model_deviance",
    "optimize_factor",
    "split",
    "summary",
]
