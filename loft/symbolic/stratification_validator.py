"""
Stratification validator for checking integrity and detecting violations.

Validates:
- Dependency graph integrity
- Circular dependency detection
- Constitutional layer immutability
- Policy compliance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from loguru import logger

from loft.symbolic.asp_rule import ASPRule, StratificationLevel
from loft.symbolic.stratified_core import StratifiedASPCore


@dataclass
class StratificationViolation:
    """A single stratification integrity violation."""

    severity: str  # "critical", "high", "medium", "low"
    layer: Optional[StratificationLevel]
    violation_type: str
    description: str
    affected_rules: List[str] = field(default_factory=list)


@dataclass
class StratificationReport:
    """Comprehensive stratification integrity report."""

    valid: bool
    violations: List[StratificationViolation]
    stats: Dict[str, any]
    cycles_detected: List[List[str]] = field(default_factory=list)


class StratificationValidator:
    """Validates stratification integrity across the ASP core."""

    def __init__(self, initial_constitutional_rules: Optional[List[ASPRule]] = None):
        """
        Initialize validator.

        Args:
            initial_constitutional_rules: Expected constitutional rules (for immutability check)
        """
        self.initial_constitutional_rules = initial_constitutional_rules or []
        logger.info("Initialized StratificationValidator")

    def validate_core(self, core: StratifiedASPCore) -> StratificationReport:
        """
        Comprehensive validation of stratification integrity.

        Checks:
        1. No circular dependencies across layers
        2. Constitutional layer unchanged (if initial rules provided)
        3. Dependency graph respects hierarchy
        4. Policies enforced correctly

        Args:
            core: Stratified ASP core to validate

        Returns:
            Comprehensive validation report
        """
        violations = []

        logger.info("Running stratification integrity validation...")

        # Check 1: Constitutional layer immutability
        if self.initial_constitutional_rules:
            const_violations = self._check_constitutional_immutability(core)
            violations.extend(const_violations)

        # Check 2: Dependency graph compliance
        dep_violations = self._check_dependency_compliance(core)
        violations.extend(dep_violations)

        # Check 3: Circular dependencies
        cycles = self._detect_cycles(core)
        for cycle in cycles:
            violations.append(
                StratificationViolation(
                    severity="high",
                    layer=None,
                    violation_type="circular_dependency",
                    description=f"Circular dependency detected: {' → '.join(cycle)}",
                    affected_rules=cycle,
                )
            )

        # Get stats
        stats = core.get_modification_stats()

        logger.info(
            f"Validation complete: {len(violations)} violations, {len(cycles)} cycles detected"
        )

        return StratificationReport(
            valid=len(violations) == 0,
            violations=violations,
            stats=stats,
            cycles_detected=cycles,
        )

    def _check_constitutional_immutability(
        self, core: StratifiedASPCore
    ) -> List[StratificationViolation]:
        """
        Check that constitutional layer hasn't been modified.

        Args:
            core: Core to check

        Returns:
            List of violations found
        """
        violations = []

        constitutional_rules = core.get_rules_by_layer(
            StratificationLevel.CONSTITUTIONAL
        )

        # Check if any rules were added or removed
        initial_ids = {r.rule_id for r in self.initial_constitutional_rules}
        current_ids = {r.rule_id for r in constitutional_rules}

        added = current_ids - initial_ids
        removed = initial_ids - current_ids

        if added or removed:
            desc_parts = []
            if added:
                desc_parts.append(f"{len(added)} rules added")
            if removed:
                desc_parts.append(f"{len(removed)} rules removed")

            violations.append(
                StratificationViolation(
                    severity="critical",
                    layer=StratificationLevel.CONSTITUTIONAL,
                    violation_type="unauthorized_modification",
                    description=f"Constitutional layer modified: {', '.join(desc_parts)}",
                    affected_rules=list(added | removed),
                )
            )

        return violations

    def _check_dependency_compliance(
        self, core: StratifiedASPCore
    ) -> List[StratificationViolation]:
        """
        Check that all dependencies comply with stratification policies.

        Args:
            core: Core to check

        Returns:
            List of violations found
        """
        violations = []

        for layer in StratificationLevel:
            policy = core.policies[layer.value]
            rules = core.get_rules_by_layer(layer)

            for rule in rules:
                # Check each predicate dependency
                for pred in rule.predicates_used:
                    # Skip if predicate is defined by this rule
                    if pred in rule.new_predicates:
                        continue

                    pred_layer = core._find_predicate_layer(pred)

                    if pred_layer and pred_layer not in policy.can_depend_on:
                        violations.append(
                            StratificationViolation(
                                severity="high",
                                layer=layer,
                                violation_type="invalid_dependency",
                                description=(
                                    f"Rule {rule.rule_id} in {layer.value} depends on "
                                    f"predicate '{pred}' from {pred_layer.value}"
                                ),
                                affected_rules=[rule.rule_id],
                            )
                        )

        return violations

    def _detect_cycles(self, core: StratifiedASPCore) -> List[List[str]]:
        """
        Detect circular dependencies in predicate definitions.

        Uses DFS to detect cycles in the predicate dependency graph.

        Args:
            core: Core to check

        Returns:
            List of cycles, where each cycle is a list of predicate names
        """
        # Build dependency graph: predicate -> predicates it depends on
        graph: Dict[str, Set[str]] = {}
        all_rules = core.get_all_rules()

        for rule in all_rules:
            for new_pred in rule.new_predicates:
                if new_pred not in graph:
                    graph[new_pred] = set()

                # Add dependencies (excluding self-reference)
                graph[new_pred].update(p for p in rule.predicates_used if p != new_pred)

        # Detect cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            """
            DFS to detect cycles.

            Returns:
                True if cycle detected
            """
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check neighbors
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle!
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        # Run DFS from each node
        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles
