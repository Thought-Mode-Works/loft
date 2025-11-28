"""
Rule evolution tracking system (Phase 4.3 + 4.5 enhancements).

Maintains version history, dialectical lineage, and performance metrics
across rule iterations.

Phase 4.5 additions:
- Text-based visualization of rule genealogy
- Query interface for evolution analytics
"""

from loft.evolution.evolution_schemas import (
    RuleVersion,
    RuleLineage,
    EvolutionContext,
    PerformanceSnapshot,
)
from loft.evolution.evolution_tracker import RuleEvolutionTracker
from loft.evolution.evolution_store import RuleEvolutionStore
from loft.evolution.visualization import EvolutionVisualizer, print_rule_genealogy
from loft.evolution.queries import RuleEvolutionDB

__all__ = [
    "RuleVersion",
    "RuleLineage",
    "EvolutionContext",
    "PerformanceSnapshot",
    "RuleEvolutionTracker",
    "RuleEvolutionStore",
    "EvolutionVisualizer",
    "print_rule_genealogy",
    "RuleEvolutionDB",
]
