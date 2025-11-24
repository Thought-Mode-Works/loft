"""
Core: ASP-based Symbolic Reasoning and Rule Incorporation

This module provides the symbolic core functionality using Answer Set Programming,
including safe rule incorporation with stratified modification policies.
"""

from loft.core.incorporation import (
    IncorporationResult,
    RuleIncorporationEngine,
    SimpleASPCore,
    SimpleTestSuite,
    SimpleVersionControl,
)
from loft.core.modification_session import (
    ModificationSession,
    SessionReport,
)

__all__ = [
    # Incorporation
    "IncorporationResult",
    "RuleIncorporationEngine",
    "SimpleASPCore",
    "SimpleTestSuite",
    "SimpleVersionControl",
    # Sessions
    "ModificationSession",
    "SessionReport",
]
