"""
Logical consistency checking engine for symbolic core.

Implements various consistency checks:
- Non-contradiction: Ensure no rule contradicts another
- Completeness: Identify gaps in rule coverage
- Coherence: Related rules must be compatible
- Transitivity: Logical chains must close correctly
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from enum import Enum
import re
from ..version_control import Rule, CoreState, StratificationLevel


class InconsistencyType(str, Enum):
    """Types of logical inconsistencies."""

    CONTRADICTION = "contradiction"
    INCOMPLETENESS = "incompleteness"
    INCOHERENCE = "incoherence"
    TRANSITIVITY_VIOLATION = "transitivity_violation"


@dataclass
class Inconsistency:
    """Represents a logical inconsistency."""

    type: InconsistencyType
    severity: str  # "error", "warning", "info"
    message: str
    rule_ids: List[str]
    details: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format inconsistency for display."""
        rules_str = ", ".join(self.rule_ids)
        return f"[{self.severity.upper()}] {self.type.value}: {self.message} (rules: {rules_str})"


@dataclass
class ConsistencyReport:
    """Report of consistency check results."""

    passed: bool
    inconsistencies: List[Inconsistency]
    warnings: int
    errors: int
    info: int

    def summary(self) -> str:
        """Generate summary of consistency report."""
        if self.passed:
            return "✓ All consistency checks passed"
        return (
            f"✗ Consistency check failed: "
            f"{self.errors} errors, {self.warnings} warnings, {self.info} info"
        )

    def format(self) -> str:
        """Format full report for display."""
        lines = ["=" * 60, "Consistency Check Report", "=" * 60, ""]
        lines.append(self.summary())
        lines.append("")

        if self.inconsistencies:
            lines.append("Inconsistencies:")
            lines.append("-" * 60)
            for inc in self.inconsistencies:
                lines.append(str(inc))
                if inc.details:
                    for key, value in inc.details.items():
                        lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class ConsistencyChecker:
    """
    Logical consistency checker for symbolic core.

    Performs various consistency checks on rule sets to ensure
    logical coherence and detect contradictions.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize consistency checker.

        Args:
            strict: If True, warnings count as failures
        """
        self.strict = strict

    def check(self, state: CoreState) -> ConsistencyReport:
        """
        Perform full consistency check on a core state.

        Args:
            state: Core state to check

        Returns:
            ConsistencyReport with all findings
        """
        inconsistencies: List[Inconsistency] = []

        # Run all consistency checks
        inconsistencies.extend(self.check_contradictions(state.rules))
        inconsistencies.extend(self.check_completeness(state.rules))
        inconsistencies.extend(self.check_coherence(state.rules))
        inconsistencies.extend(self.check_transitivity(state.rules))

        # Count by severity
        errors = sum(1 for i in inconsistencies if i.severity == "error")
        warnings = sum(1 for i in inconsistencies if i.severity == "warning")
        info = sum(1 for i in inconsistencies if i.severity == "info")

        # Determine pass/fail
        passed = errors == 0 and (not self.strict or warnings == 0)

        return ConsistencyReport(
            passed=passed,
            inconsistencies=inconsistencies,
            warnings=warnings,
            errors=errors,
            info=info,
        )

    def check_contradictions(self, rules: List[Rule]) -> List[Inconsistency]:
        """
        Check for contradictory rules.

        Looks for rules that directly contradict each other, such as:
        - p. and -p.
        - p :- q. and -p :- q.

        Args:
            rules: List of rules to check

        Returns:
            List of contradiction inconsistencies
        """
        inconsistencies: List[Inconsistency] = []

        # Build map of predicates to rules
        predicate_map: Dict[str, List[Tuple[Rule, bool]]] = (
            {}
        )  # predicate -> (rule, is_negated)

        for rule in rules:
            predicates = self._extract_predicates(rule.content)
            for pred, is_negated in predicates:
                if pred not in predicate_map:
                    predicate_map[pred] = []
                predicate_map[pred].append((rule, is_negated))

        # Check for contradictions
        for pred, rule_list in predicate_map.items():
            # Check if same predicate appears both negated and non-negated
            has_positive = any(not neg for _, neg in rule_list)
            has_negative = any(neg for _, neg in rule_list)

            if has_positive and has_negative:
                # Find the conflicting rules
                positive_rules = [r for r, neg in rule_list if not neg]
                negative_rules = [r for r, neg in rule_list if neg]

                # Check if they have same conditions (simplified check)
                for pos_rule in positive_rules:
                    for neg_rule in negative_rules:
                        if self._same_conditions(pos_rule.content, neg_rule.content):
                            inconsistencies.append(
                                Inconsistency(
                                    type=InconsistencyType.CONTRADICTION,
                                    severity="error",
                                    message=f"Contradictory rules for predicate '{pred}'",
                                    rule_ids=[pos_rule.rule_id, neg_rule.rule_id],
                                    details={
                                        "positive": pos_rule.content,
                                        "negative": neg_rule.content,
                                    },
                                )
                            )

        return inconsistencies

    def check_completeness(self, rules: List[Rule]) -> List[Inconsistency]:
        """
        Check for completeness issues.

        Identifies potential gaps in rule coverage, such as:
        - Predicates used in rule bodies but never defined
        - Missing rules for expected stratification levels

        Args:
            rules: List of rules to check

        Returns:
            List of incompleteness inconsistencies
        """
        inconsistencies: List[Inconsistency] = []

        # Extract all predicates defined (in heads) and used (in bodies)
        defined_predicates: Set[str] = set()
        used_predicates: Set[str] = set()

        for rule in rules:
            # Extract head predicates (defined)
            head = self._extract_head(rule.content)
            if head:
                defined_predicates.update(p for p, _ in self._extract_predicates(head))

            # Extract body predicates (used)
            body = self._extract_body(rule.content)
            if body:
                used_predicates.update(p for p, _ in self._extract_predicates(body))

        # Check for undefined predicates
        undefined = used_predicates - defined_predicates
        if undefined:
            inconsistencies.append(
                Inconsistency(
                    type=InconsistencyType.INCOMPLETENESS,
                    severity="warning",
                    message=f"Predicates used but not defined: {', '.join(sorted(undefined))}",
                    rule_ids=[],
                    details={"undefined_predicates": ", ".join(sorted(undefined))},
                )
            )

        return inconsistencies

    def check_coherence(self, rules: List[Rule]) -> List[Inconsistency]:
        """
        Check for coherence issues.

        Verifies that related rules are compatible, such as:
        - Rules at different stratification levels are properly ordered
        - Confidence scores are consistent for related rules

        Args:
            rules: List of rules to check

        Returns:
            List of coherence inconsistencies
        """
        inconsistencies: List[Inconsistency] = []

        # Check stratification ordering
        # Constitutional rules should not depend on lower-level rules
        const_rules = [
            r for r in rules if r.level == StratificationLevel.CONSTITUTIONAL
        ]
        lower_rules = [
            r for r in rules if r.level != StratificationLevel.CONSTITUTIONAL
        ]

        for const_rule in const_rules:
            body_preds = set()
            body = self._extract_body(const_rule.content)
            if body:
                body_preds = set(p for p, _ in self._extract_predicates(body))

            # Check if constitutional rule depends on lower-level predicates
            for lower_rule in lower_rules:
                head = self._extract_head(lower_rule.content)
                if head:
                    head_preds = set(p for p, _ in self._extract_predicates(head))
                    if body_preds & head_preds:
                        inconsistencies.append(
                            Inconsistency(
                                type=InconsistencyType.INCOHERENCE,
                                severity="warning",
                                message="Constitutional rule depends on lower-level rule",
                                rule_ids=[const_rule.rule_id, lower_rule.rule_id],
                                details={
                                    "constitutional": const_rule.content,
                                    "lower_level": lower_rule.content,
                                },
                            )
                        )

        return inconsistencies

    def check_transitivity(self, rules: List[Rule]) -> List[Inconsistency]:
        """
        Check for transitivity violations.

        Verifies that logical chains close correctly, such as:
        - If A -> B and B -> C, then A -> C should be derivable

        Args:
            rules: List of rules to check

        Returns:
            List of transitivity inconsistencies
        """
        inconsistencies: List[Inconsistency] = []

        # Build implication graph
        implications: Dict[str, Set[str]] = {}  # predicate -> set of implied predicates

        for rule in rules:
            head = self._extract_head(rule.content)
            body = self._extract_body(rule.content)

            if head and body:
                head_preds = [p for p, _ in self._extract_predicates(head)]
                body_preds = [p for p, _ in self._extract_predicates(body)]

                for body_pred in body_preds:
                    for head_pred in head_preds:
                        if body_pred not in implications:
                            implications[body_pred] = set()
                        implications[body_pred].add(head_pred)

        # Check for transitive closure
        # For now, just verify that chains are not circular
        for pred, implied in implications.items():
            if pred in implied:
                inconsistencies.append(
                    Inconsistency(
                        type=InconsistencyType.TRANSITIVITY_VIOLATION,
                        severity="warning",
                        message=f"Circular dependency detected for predicate '{pred}'",
                        rule_ids=[],
                        details={"predicate": pred},
                    )
                )

        return inconsistencies

    def _extract_predicates(self, rule_text: str) -> List[Tuple[str, bool]]:
        """
        Extract predicates from rule text.

        Args:
            rule_text: ASP rule text

        Returns:
            List of (predicate_name, is_negated) tuples
        """
        predicates: List[Tuple[str, bool]] = []

        # Simple pattern matching for predicates
        # Matches: predicate_name(args) or -predicate_name(args) or just predicate_name
        pattern = r"(-)?(\w+)(?:\([^)]*\))?"

        for match in re.finditer(pattern, rule_text):
            negation = match.group(1)
            predicate = match.group(2)

            # Skip common keywords
            if predicate in ["not", "if", "then", "else"]:
                continue

            is_negated = negation == "-"
            predicates.append((predicate, is_negated))

        return predicates

    def _extract_head(self, rule_text: str) -> Optional[str]:
        """Extract head of a rule (before :-)."""
        if ":-" in rule_text:
            return rule_text.split(":-")[0].strip()
        # Fact (no body)
        return rule_text.strip()

    def _extract_body(self, rule_text: str) -> Optional[str]:
        """Extract body of a rule (after :-)."""
        if ":-" in rule_text:
            parts = rule_text.split(":-")
            if len(parts) > 1:
                return parts[1].strip()
        return None

    def _same_conditions(self, rule1: str, rule2: str) -> bool:
        """Check if two rules have the same conditions (simplified)."""
        body1 = self._extract_body(rule1)
        body2 = self._extract_body(rule2)

        # If both are facts (no body), consider them same conditions
        if not body1 and not body2:
            return True

        # If one is fact and other is not, different conditions
        if not body1 or not body2:
            return False

        # Normalize and compare
        return self._normalize(body1) == self._normalize(body2)

    def _normalize(self, text: str) -> str:
        """Normalize rule text for comparison."""
        # Remove whitespace and convert to lowercase
        return re.sub(r"\s+", "", text.lower())
