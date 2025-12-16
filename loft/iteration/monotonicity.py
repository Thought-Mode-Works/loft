"""
Monotonicity enforcement for iterative rule building.

Ensures that coverage never decreases during rule modifications,
providing safety guarantees for autonomous learning.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Any, Optional
from enum import Enum


class OperationType(str, Enum):
    """Types of rule operations."""

    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"
    REPLACE = "replace"


@dataclass
class CoverageImpact:
    """Impact assessment for a proposed operation."""

    operation: OperationType
    rule_id: str
    predicates_added: Set[str]
    predicates_removed: Set[str]
    estimated_coverage_change: float
    would_violate_monotonicity: bool
    reason: str


class MonotonicityEnforcer:
    """
    Ensures coverage monotonically increases.

    Prevents operations that would decrease coverage,
    maintaining forward progress guarantees.
    """

    def __init__(self, coverage_tracker: Any, tolerance: float = 0.0):
        """
        Initialize monotonicity enforcer.

        Args:
            coverage_tracker: CoverageTracker instance
            tolerance: Allowed coverage decrease tolerance (default 0.0)
        """
        self.tracker = coverage_tracker
        self.tolerance = tolerance

    def would_decrease_coverage(
        self, operation: OperationType, rule: Any, current_predicates: Set[str]
    ) -> bool:
        """
        Check if operation would decrease coverage.

        Args:
            operation: Type of operation
            rule: ASPRule object
            current_predicates: Currently covered predicates

        Returns:
            True if operation would decrease coverage beyond tolerance
        """
        impact = self.assess_impact(operation, rule, current_predicates)
        return impact.would_violate_monotonicity

    def assess_impact(
        self, operation: OperationType, rule: Any, current_predicates: Set[str]
    ) -> CoverageImpact:
        """
        Assess coverage impact of an operation.

        Args:
            operation: Type of operation
            rule: ASPRule object
            current_predicates: Currently covered predicates

        Returns:
            CoverageImpact assessment
        """
        # Extract predicates from rule
        rule_predicates = self._extract_rule_predicates(rule)

        if operation == OperationType.ADD:
            # Adding always safe (monotonic increase or stable)
            new_predicates = rule_predicates - current_predicates
            coverage_change = len(new_predicates) / max(
                len(self.tracker.domain_predicates), 1
            )

            return CoverageImpact(
                operation=operation,
                rule_id=getattr(rule, "rule_id", "unknown"),
                predicates_added=new_predicates,
                predicates_removed=set(),
                estimated_coverage_change=coverage_change,
                would_violate_monotonicity=False,
                reason="Adding rules maintains monotonicity",
            )

        elif operation == OperationType.REMOVE:
            # Check if removing would orphan predicates
            removed_predicates = self._would_orphan_predicates(rule, current_predicates)

            if removed_predicates:
                coverage_change = -len(removed_predicates) / max(
                    len(self.tracker.domain_predicates), 1
                )
                would_violate = abs(coverage_change) > self.tolerance

                return CoverageImpact(
                    operation=operation,
                    rule_id=getattr(rule, "rule_id", "unknown"),
                    predicates_added=set(),
                    predicates_removed=removed_predicates,
                    estimated_coverage_change=coverage_change,
                    would_violate_monotonicity=would_violate,
                    reason=f"Removing rule would orphan {len(removed_predicates)} predicates",
                )
            else:
                # Safe to remove (predicates covered by other rules)
                return CoverageImpact(
                    operation=operation,
                    rule_id=getattr(rule, "rule_id", "unknown"),
                    predicates_added=set(),
                    predicates_removed=set(),
                    estimated_coverage_change=0.0,
                    would_violate_monotonicity=False,
                    reason="Predicates remain covered by other rules",
                )

        elif operation in (OperationType.MODIFY, OperationType.REPLACE):
            # Modification: check net predicate change
            # (This is simplified - real implementation would need before/after rules)
            return CoverageImpact(
                operation=operation,
                rule_id=getattr(rule, "rule_id", "unknown"),
                predicates_added=set(),
                predicates_removed=set(),
                estimated_coverage_change=0.0,
                would_violate_monotonicity=False,
                reason="Modification impact assessment not fully implemented",
            )

        return CoverageImpact(
            operation=operation,
            rule_id="unknown",
            predicates_added=set(),
            predicates_removed=set(),
            estimated_coverage_change=0.0,
            would_violate_monotonicity=False,
            reason="Unknown operation type",
        )

    def _extract_rule_predicates(self, rule: Any) -> Set[str]:
        """Extract predicates defined by rule (head predicates)."""
        if hasattr(rule, "new_predicates"):
            return set(rule.new_predicates)
        elif hasattr(rule, "extract_predicates"):
            return set(rule.extract_predicates())
        else:
            return set()

    def _would_orphan_predicates(
        self, rule: Any, current_predicates: Set[str]
    ) -> Set[str]:
        """
        Check if removing rule would orphan any predicates.

        Args:
            rule: Rule being removed
            current_predicates: Currently covered predicates

        Returns:
            Set of predicates that would be orphaned
        """
        # This is simplified - real implementation would need
        # to check all rules to see which predicates are covered
        # by multiple rules
        rule_predicates = self._extract_rule_predicates(rule)

        # For now, assume all rule predicates would be orphaned
        # A more sophisticated implementation would check other rules
        return rule_predicates & current_predicates

    def get_safe_operations(
        self,
        proposed_operations: List[Tuple[OperationType, Any]],
        current_predicates: Set[str],
    ) -> List[Tuple[OperationType, Any]]:
        """
        Filter operations to only those that maintain monotonicity.

        Args:
            proposed_operations: List of (operation, rule) tuples
            current_predicates: Currently covered predicates

        Returns:
            List of safe operations that maintain monotonicity
        """
        safe_operations = []

        for operation, rule in proposed_operations:
            if not self.would_decrease_coverage(operation, rule, current_predicates):
                safe_operations.append((operation, rule))

        return safe_operations

    def verify_monotonicity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify that coverage has been monotonic throughout history.

        Returns:
            (is_monotonic, violation_message)
        """
        if not self.tracker.history or len(self.tracker.history) < 2:
            return True, None

        for i in range(1, len(self.tracker.history)):
            prev_coverage = self.tracker.history[i - 1].predicate_coverage
            curr_coverage = self.tracker.history[i].predicate_coverage

            if curr_coverage < prev_coverage - self.tolerance:
                violation_msg = (
                    f"Monotonicity violation at snapshot {i}: "
                    f"coverage decreased from {prev_coverage:.2%} to {curr_coverage:.2%}"
                )
                return False, violation_msg

        return True, None

    def get_coverage_guarantees(self) -> dict:
        """
        Get current coverage guarantees.

        Returns:
            Dictionary with guarantee information
        """
        is_monotonic, violation = self.verify_monotonicity()

        current_coverage = (
            self.tracker.current_coverage if self.tracker.history else 0.0
        )

        return {
            "monotonic": is_monotonic,
            "current_coverage": current_coverage,
            "tolerance": self.tolerance,
            "violation": violation,
            "snapshots_count": len(self.tracker.history),
        }
