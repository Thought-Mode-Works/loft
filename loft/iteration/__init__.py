"""
Iterative rule building module for LOFT.

Provides iterative ASP rule building with:
- Coverage tracking and expansion monitoring
- Monotonicity enforcement
- Living document generation
- Contradiction and redundancy detection
"""

from loft.iteration.coverage_tracker import CoverageTracker, CoverageMetrics
from loft.iteration.living_document import (
    LivingDocumentManager,
    RuleAdjustment,
    CycleSummary,
)
from loft.iteration.monotonicity import (
    MonotonicityEnforcer,
    OperationType,
    CoverageImpact,
)
from loft.iteration.rule_builder import (
    IterativeRuleBuilder,
    ContradictionCheck,
    RedundancyCheck,
    AdditionResult,
)

__all__ = [
    # Coverage tracking
    "CoverageTracker",
    "CoverageMetrics",
    # Living document
    "LivingDocumentManager",
    "RuleAdjustment",
    "CycleSummary",
    # Monotonicity
    "MonotonicityEnforcer",
    "OperationType",
    "CoverageImpact",
    # Rule building
    "IterativeRuleBuilder",
    "ContradictionCheck",
    "RedundancyCheck",
    "AdditionResult",
]
