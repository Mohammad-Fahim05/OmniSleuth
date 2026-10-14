"""Benchmark evaluation module."""

from omnisleuth.evaluation.benchmark_suite import (
    BENCHMARK_QUESTIONS,
    BenchmarkQuestion,
)
from omnisleuth.evaluation.dataset_generator import (
    generate_benchmark_sales_dataset,
)
from omnisleuth.evaluation.evaluator import (
    ExperimentRecord,
    OmniSleuthEvaluator,
    PerQuestionResult,
    SystemBenchmarkResult,
)

__all__ = [
    "generate_benchmark_sales_dataset",
    "BenchmarkQuestion",
    "BENCHMARK_QUESTIONS",
    "OmniSleuthEvaluator",
    "SystemBenchmarkResult",
    "PerQuestionResult",
    "ExperimentRecord",
]
