"""Nyra Speech Timing Benchmark.

Model-agnostic evaluation of word-level timing accuracy: load ground-truth word
timings for a dataset, then normalize, align and score predicted word timings
with MAE, F1@xms and mIoU.
"""

from .datasets import (
    DATASETS,
    GroundTruthSample,
    list_datasets,
    load_ground_truth,
)
from .evaluate import (
    DEFAULT_COLLARS,
    BenchmarkResult,
    evaluate_dataset,
    evaluate_sample,
)
from .schema import (
    PredictionFile,
    ValidationReport,
    load_predictions,
    validate_predictions,
)

__version__ = "0.1.0"

__all__ = [
    "DATASETS",
    "GroundTruthSample",
    "list_datasets",
    "load_ground_truth",
    "DEFAULT_COLLARS",
    "BenchmarkResult",
    "evaluate_dataset",
    "evaluate_sample",
    "PredictionFile",
    "ValidationReport",
    "load_predictions",
    "validate_predictions",
]
