"""
Dialectical reasoning components for Phase 4.

This module implements dialectical validation where rules are refined through
criticism and synthesis rather than simple binary acceptance/rejection.

Phase 4.1: Critic LLM System
Phase 4.2: Multi-Agent Debate Framework
"""

from loft.dialectical.critique_schemas import (
    Contradiction,
    CritiqueReport,
    CritiqueSeverity,
    EdgeCase,
)
from loft.dialectical.debate_schemas import (
    DebateArgument,
    DebateContext,
    DebatePhase,
    DebateRound,
    DialecticalCycleResult,
)
from loft.dialectical.debate_framework import DebateFramework
from loft.dialectical.synthesizer import Synthesizer
from loft.dialectical.critic import CriticSystem

__all__ = [
    # Phase 4.1: Critique
    "CritiqueReport",
    "EdgeCase",
    "Contradiction",
    "CritiqueSeverity",
    "CriticSystem",
    # Phase 4.2: Debate
    "DebateFramework",
    "Synthesizer",
    "DebateArgument",
    "DebateContext",
    "DebatePhase",
    "DebateRound",
    "DialecticalCycleResult",
]
