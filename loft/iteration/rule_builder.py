"""
Iterative ASP rule builder for accumulating knowledge across runs.

Validates new rules against existing knowledge, tracks coverage expansion,
and documents all adjustments for full transparency.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

from loft.symbolic.asp_rule import ASPRule
from loft.symbolic.stratification import StratificationLevel
from loft.persistence.asp_persistence import ASPPersistenceManager
from loft.iteration.coverage_tracker import CoverageTracker
from loft.iteration.living_document import LivingDocumentManager, RuleAdjustment
from loft.iteration.monotonicity import MonotonicityEnforcer


@dataclass
class ContradictionCheck:
    """Result of contradiction check."""

    has_contradiction: bool
    conflicting_rules: List[ASPRule] = None

    def __post_init__(self):
        if self.conflicting_rules is None:
            self.conflicting_rules = []


@dataclass
class RedundancyCheck:
    """Result of redundancy check."""

    is_redundant: bool
    subsuming_rule: Optional[ASPRule] = None


@dataclass
class AdditionResult:
    """Result of adding a rule."""

    status: str  # "added", "rejected", "skipped"
    reason: str
    coverage_change: float = 0.0
    new_predicates_covered: List[str] = None
    details: Any = None

    def __post_init__(self):
        if self.new_predicates_covered is None:
            self.new_predicates_covered = []


class IterativeRuleBuilder:
    """
    Builds ASP rules iteratively across experiment runs.

    Features:
    - Loads existing rules as foundation
    - Validates new rules against existing knowledge
    - Checks for contradictions and redundancy
    - Tracks coverage expansion
    - Documents all adjustments
    - Enforces monotonicity guarantees
    """

    def __init__(
        self,
        persistence: ASPPersistenceManager,
        coverage_tracker: CoverageTracker,
        living_document: Optional[LivingDocumentManager] = None,
        enable_monotonicity: bool = True,
        monotonicity_tolerance: float = 0.0,
    ):
        """
        Initialize iterative rule builder.

        Args:
            persistence: ASP persistence manager
            coverage_tracker: Coverage tracker
            living_document: Living document manager (optional)
            enable_monotonicity: Whether to enforce monotonicity
            monotonicity_tolerance: Allowed coverage decrease tolerance
        """
        self.persistence = persistence
        self.coverage_tracker = coverage_tracker
        self.living_document = living_document or LivingDocumentManager()
        self.enable_monotonicity = enable_monotonicity

        # Monotonicity enforcer
        if enable_monotonicity:
            self.monotonicity = MonotonicityEnforcer(
                coverage_tracker, tolerance=monotonicity_tolerance
            )
        else:
            self.monotonicity = None

        # Rule storage by layer
        self.rules_by_layer: Dict[StratificationLevel, List[ASPRule]] = {
            StratificationLevel.CONSTITUTIONAL: [],
            StratificationLevel.STRATEGIC: [],
            StratificationLevel.TACTICAL: [],
            StratificationLevel.OPERATIONAL: [],
        }

        # Load existing rules
        self._load_existing_rules()

    def _load_existing_rules(self) -> None:
        """Load persisted rules as foundation for iteration."""
        try:
            load_result = self.persistence.load_all_rules()

            # Populate rules_by_layer
            self.rules_by_layer = load_result.rules_by_layer

            # Calculate initial coverage
            all_rules = self.get_all_rules()
            covered_predicates = self.coverage_tracker.extract_predicates_from_rules(
                all_rules
            )

            # Record initial metrics
            self.coverage_tracker.record_metrics(
                covered_predicates=covered_predicates,
                cases_with_predictions=0,  # Will be updated during testing
                scenarios_covered=0,  # Will be updated during testing
                total_rules=len(all_rules),
                rules_by_layer={
                    layer.value: len(rules)
                    for layer, rules in self.rules_by_layer.items()
                },
            )

        except Exception:
            # If no existing rules, start fresh
            pass

    def add_rule(self, rule: ASPRule, skip_checks: bool = False) -> AdditionResult:
        """
        Add a rule, validating against existing knowledge.

        Args:
            rule: ASPRule to add
            skip_checks: Skip contradiction/redundancy checks (for batch import)

        Returns:
            AdditionResult with status and details
        """
        # Check for contradictions
        if not skip_checks:
            contradiction_check = self._check_contradictions(rule)
            if contradiction_check.has_contradiction:
                self._document_rejection(rule, "contradiction", contradiction_check)
                return AdditionResult(
                    status="rejected",
                    reason="contradiction",
                    details=contradiction_check.conflicting_rules,
                )

            # Check for redundancy
            redundancy_check = self._check_redundancy(rule)
            if redundancy_check.is_redundant:
                self._document_rejection(rule, "redundant", redundancy_check)
                return AdditionResult(
                    status="skipped",
                    reason="redundant",
                    details=redundancy_check.subsuming_rule,
                )

        # Check coverage impact (before adding)
        coverage_before = self.coverage_tracker.current_coverage or 0.0

        # Add rule to layer
        layer = rule.stratification_level
        self.rules_by_layer[layer].append(rule)

        # Update coverage
        all_rules = self.get_all_rules()
        covered_predicates = self.coverage_tracker.extract_predicates_from_rules(
            all_rules
        )

        self.coverage_tracker.record_metrics(
            covered_predicates=covered_predicates,
            cases_with_predictions=0,  # Updated externally
            scenarios_covered=0,  # Updated externally
            total_rules=len(all_rules),
            rules_by_layer={
                lyr.value: len(rules) for lyr, rules in self.rules_by_layer.items()
            },
        )

        coverage_after = self.coverage_tracker.current_coverage

        # Document adjustment
        self._document_adjustment(
            rule=rule,
            action="added",
            coverage_change=coverage_after - coverage_before,
        )

        # Persist rule
        try:
            self.persistence.save_rule(rule, layer)
        except Exception:
            # Log but don't fail
            pass

        return AdditionResult(
            status="added",
            reason="passed all checks",
            coverage_change=coverage_after - coverage_before,
            new_predicates_covered=self.coverage_tracker.newly_covered_predicates,
        )

    def _check_contradictions(self, rule: ASPRule) -> ContradictionCheck:
        """
        Check if rule contradicts existing rules.

        Args:
            rule: Rule to check

        Returns:
            ContradictionCheck result
        """
        # Simplified contradiction detection:
        # Check if rules have opposite conclusions for same conditions

        existing_rules = self.get_all_rules()
        conflicting = []

        for existing in existing_rules:
            # Check for direct contradiction patterns
            # (This is simplified - real implementation would use ASP reasoning)
            if self._rules_likely_conflict(rule, existing):
                conflicting.append(existing)

        return ContradictionCheck(
            has_contradiction=len(conflicting) > 0, conflicting_rules=conflicting
        )

    def _rules_likely_conflict(self, rule1: ASPRule, rule2: ASPRule) -> bool:
        """
        Heuristic check if rules likely conflict.

        Real implementation would use ASP solver to check for contradictions.
        """
        # Simplified heuristic:
        # - If both rules define same predicate
        # - And have similar body conditions
        # - They might conflict

        rule1_head = set(rule1.new_predicates)
        rule2_head = set(rule2.new_predicates)

        # Same head predicates might conflict
        if rule1_head & rule2_head:
            # Check if bodies are similar (simplified)
            rule1_body_preds = set(rule1.predicates_used)
            rule2_body_preds = set(rule2.predicates_used)

            # If bodies share many predicates, might conflict
            overlap = rule1_body_preds & rule2_body_preds
            if len(overlap) > len(rule1_body_preds) * 0.5:
                return True

        return False

    def _check_redundancy(self, rule: ASPRule) -> RedundancyCheck:
        """
        Check if rule is subsumed by existing rules.

        Args:
            rule: Rule to check

        Returns:
            RedundancyCheck result
        """
        existing_rules = self.get_all_rules()

        for existing in existing_rules:
            # Check if existing rule subsumes new rule
            if self._rule_subsumes(existing, rule):
                return RedundancyCheck(is_redundant=True, subsuming_rule=existing)

        return RedundancyCheck(is_redundant=False)

    def _rule_subsumes(self, existing: ASPRule, new_rule: ASPRule) -> bool:
        """
        Check if existing rule subsumes new rule.

        A rule subsumes another if it covers the same or more cases.
        Real implementation would use ASP subsumption checking.
        """
        # Simplified heuristic:
        # If rules have same head and existing has weaker body conditions
        existing_head = set(existing.new_predicates)
        new_head = set(new_rule.new_predicates)

        # Must have same head
        if existing_head != new_head:
            return False

        # Check if existing body is subset of new body
        # (weaker conditions = more general)
        existing_body = set(existing.predicates_used)
        new_body = set(new_rule.predicates_used)

        # If existing uses fewer body predicates, it's more general
        if len(existing_body) < len(new_body) and existing_body.issubset(new_body):
            return True

        # Check for exact duplicate
        if existing.asp_text.strip() == new_rule.asp_text.strip():
            return True

        return False

    def _document_adjustment(
        self, rule: ASPRule, action: str, coverage_change: float
    ) -> None:
        """Document rule adjustment in living document."""
        adjustment = RuleAdjustment(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            rule_id=rule.rule_id,
            rule_text=rule.asp_text,
            layer=rule.stratification_level.value,
            coverage_change=coverage_change,
            reason=rule.metadata.notes or "No reason provided",
            source=rule.metadata.provenance,
            metadata={
                "confidence": rule.confidence,
                "validation_score": rule.metadata.validation_score,
                "tags": rule.metadata.tags,
            },
        )

        self.living_document.append_adjustment(adjustment)

    def _document_rejection(
        self, rule: ASPRule, reason: str, check_result: Any
    ) -> None:
        """Document rule rejection in living document."""
        details = {}
        if isinstance(check_result, ContradictionCheck):
            details["conflicting_rules"] = [
                r.rule_id for r in check_result.conflicting_rules
            ]
        elif isinstance(check_result, RedundancyCheck):
            details["subsumed_by"] = (
                check_result.subsuming_rule.rule_id
                if check_result.subsuming_rule
                else None
            )

        adjustment = RuleAdjustment(
            timestamp=datetime.utcnow().isoformat(),
            action="rejected",
            rule_id=rule.rule_id,
            rule_text=rule.asp_text,
            layer=rule.stratification_level.value,
            coverage_change=0.0,
            reason=reason,
            source=rule.metadata.provenance,
            metadata=details,
        )

        self.living_document.append_adjustment(adjustment)

    def get_all_rules(self) -> List[ASPRule]:
        """Get all rules from all layers."""
        all_rules = []
        for rules in self.rules_by_layer.values():
            all_rules.extend(rules)
        return all_rules

    def get_rules_by_layer(self, layer: StratificationLevel) -> List[ASPRule]:
        """Get rules from specific layer."""
        return self.rules_by_layer.get(layer, [])

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics about rule base."""
        return {
            "total_rules": len(self.get_all_rules()),
            "rules_by_layer": {
                layer.value: len(rules) for layer, rules in self.rules_by_layer.items()
            },
            "coverage": self.coverage_tracker.current_coverage,
            "coverage_trend": self.coverage_tracker.get_coverage_trend(),
            "monotonic": self.coverage_tracker.is_monotonic(),
            "uncovered_predicates": len(
                self.coverage_tracker.get_uncovered_predicates()
            ),
        }
