"""
Shared Components

This domain contains shared utilities, types, constants,
and common functionality used across all domains.
"""

from .test_framework import TestFramework, TestResult, TestSuite, BenchmarkRunner, test_decorator
from .lazy_imports import LazyLoader, lazy_import, get_torch, safe_import
from . import stub_torch

__all__ = [
    "TestFramework",
    "TestResult",
    "TestSuite",
    "BenchmarkRunner",
    "test_decorator",
    "LazyLoader",
    "lazy_import",
    "get_torch",
    "safe_import",
    "stub_torch",
]


def get_compat_torch():
    """
    Get torch or stub torch if unavailable.
    
    Returns:
        Module: Real torch or stub_torch
    """
    torch = get_torch()
    if torch is None:
        return stub_torch
    return torch
