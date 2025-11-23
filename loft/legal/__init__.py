"""
Legal domain module for contract law reasoning.

This module implements contract law rules, focusing on the Statute of Frauds
as the initial test domain for the symbolic-neural architecture.
"""

from .statute_of_frauds import (
    StatuteOfFraudsTestCase,
    StatuteOfFraudsDemo,
    StatuteOfFraudsSystem,
)
from .test_cases import ALL_TEST_CASES

__all__ = [
    "StatuteOfFraudsTestCase",
    "StatuteOfFraudsDemo",
    "StatuteOfFraudsSystem",
    "ALL_TEST_CASES",
]
