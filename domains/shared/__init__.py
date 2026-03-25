"""
Shared Components

This domain contains shared utilities, types, constants,
and common functionality used across all domains.
"""

from .test_framework import TestFramework, TestResult, TestSuite, BenchmarkRunner, test_decorator

__all__ = [
    "TestFramework",
    "TestResult",
    "TestSuite",
    "BenchmarkRunner",
    "test_decorator",
]
