"""
Shared Components

This domain contains shared utilities, types, constants,
and common functionality used across all domains.
"""

from .test_framework import TestFramework, TestResult, TestSuite, BenchmarkRunner, test_decorator
from .lazy_imports import LazyLoader, get_torch, get_numpy

__all__ = [
    "TestFramework",
    "TestResult",
    "TestSuite",
    "BenchmarkRunner",
    "test_decorator",
    "LazyLoader",
    "get_torch",
    "get_numpy",
]
