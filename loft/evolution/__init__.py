"""
Rule evolution tracking and visualization.

Provides infrastructure to track how rules evolve over time through
dialectical refinement, A/B testing, and performance-based selection.

Example usage:
    >>> from loft.evolution import RuleEvolutionTracker, RuleMetadata
    >>>
    >>> # Create tracker
    >>> tracker = RuleEvolutionTracker()
    >>>
    >>> # Track a new rule
    >>> metadata = tracker.create_rule(
    ...     rule_text="enforceable(C) :- has_consideration(C).",
    ...     natural_language="A contract requires consideration",
    ...     created_by="llm_generator",
    ... )
    >>>
    >>> # Record validation result
    >>> tracker.record_validation(metadata.rule_id, passed=True, accuracy=0.85)
    >>>
    >>> # Create new version
    >>> new_metadata = tracker.create_version(
    ...     parent_id=metadata.rule_id,
    ...     new_rule_text="enforceable(C) :- has_consideration(C), not void(C).",
    ...     change_reason="Added void contract check",
    ... )
"""

from .tracking import (
    RuleMetadata,
    ValidationResult,
    ABTestResult,
    StratificationLayer,
    RuleStatus,
    RuleEvolutionTracker,
)

from .storage import (
    RuleEvolutionStorage,
    StorageConfig,
)

from .visualization import (
    format_rule_history,
    format_genealogy_tree,
    format_performance_chart,
    format_ab_test_dashboard,
)

from .queries import (
    RuleEvolutionDB,
    EvolutionQuery,
)

__all__ = [
    # Core tracking
    "RuleMetadata",
    "ValidationResult",
    "ABTestResult",
    "StratificationLayer",
    "RuleStatus",
    "RuleEvolutionTracker",
    # Storage
    "RuleEvolutionStorage",
    "StorageConfig",
    # Visualization
    "format_rule_history",
    "format_genealogy_tree",
    "format_performance_chart",
    "format_ab_test_dashboard",
    # Queries
    "RuleEvolutionDB",
    "EvolutionQuery",
]
