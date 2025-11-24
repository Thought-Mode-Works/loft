"""
Stratified ASP Core with comprehensive dependency validation.

Implements the StratifiedASPCore that enforces:
- Stratification policies
- Dependency validation
- Modification cooldowns
- Comprehensive integrity checking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from loft.symbolic.asp_rule import ASPRule, StratificationLevel
from loft.symbolic.stratification import ModificationPolicy


@dataclass
class ModificationEvent:
    """Record of a single modification to the stratified core."""

    timestamp: datetime
    layer: StratificationLevel
    action: str
    rule: ASPRule
    bypassed_checks: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AddRuleResult:
    """Result of attempting to add a rule to stratified core."""

    success: bool
    reason: str = ""
    rule_id: Optional[str] = None
    requires_human_approval: bool = False


@dataclass
class ModificationStats:
    """Statistics for modifications to a specific layer."""

    total_modifications: int
    last_modification: Optional[datetime]
    rules_current: int
    cooldown_remaining_hours: Optional[float] = None


class StratifiedASPCore:
    """
    ASP core with comprehensive stratification enforcement.

    Validates:
    1. Modification policies (confidence, autonomy, limits)
    2. Dependency constraints (can_depend_on)
    3. Modification cooldowns
    4. Stratification integrity
    """

    def __init__(
        self,
        policies: Optional[Dict[StratificationLevel, ModificationPolicy]] = None,
    ):
        """
        Initialize stratified ASP core.

        Args:
            policies: Modification policies (uses defaults if None)
        """
        from loft.symbolic.stratification import MODIFICATION_POLICIES

        self.policies = {
            level.value: policy for level, policy in (policies or MODIFICATION_POLICIES).items()
        }

        # Rules organized by layer
        self.rules_by_layer: Dict[str, List[ASPRule]] = {
            level.value: [] for level in StratificationLevel
        }

        # Predicate index: predicate name -> layer it's defined in
        self.predicate_index: Dict[str, str] = {}

        # Modification history
        self.modification_history: List[ModificationEvent] = []

        # Last modification time per layer
        self.last_modification_time: Dict[str, Optional[datetime]] = {
            level.value: None for level in StratificationLevel
        }

        logger.info("Initialized StratifiedASPCore")

    def add_rule(
        self,
        rule: ASPRule,
        target_layer: StratificationLevel,
        bypass_checks: bool = False,
        is_autonomous: bool = True,
    ) -> AddRuleResult:
        """
        Add rule to specified stratification layer.

        Validates:
        1. Policy allows modifications
        2. Confidence meets threshold
        3. Dependencies don't violate stratification
        4. Cooldown period respected

        Args:
            rule: ASP rule to add
            target_layer: Target stratification layer
            bypass_checks: Skip all checks (initialization only)
            is_autonomous: Whether this is autonomous modification

        Returns:
            AddRuleResult indicating success/failure
        """
        layer_key = target_layer.value

        if not bypass_checks:
            policy = self.policies[layer_key]

            # Check 1: Policy allows autonomous modification
            if not policy.autonomous_allowed and is_autonomous:
                logger.warning(
                    f"Autonomous modification not allowed for {layer_key} layer"
                )
                return AddRuleResult(
                    success=False,
                    reason=f"Autonomous modification not allowed for {layer_key} layer",
                    requires_human_approval=True,
                )

            # Check 2: Confidence threshold
            if rule.confidence < policy.confidence_threshold:
                logger.warning(
                    f"Confidence {rule.confidence} below threshold {policy.confidence_threshold}"
                )
                return AddRuleResult(
                    success=False,
                    reason=f"Confidence {rule.confidence:.2f} below threshold {policy.confidence_threshold}",
                )

            # Check 3: Cooldown period
            last_modification = self.last_modification_time[layer_key]
            if last_modification and policy.modification_cooldown_hours < float("inf"):
                hours_since = (datetime.now() - last_modification).total_seconds() / 3600
                if hours_since < policy.modification_cooldown_hours:
                    remaining = policy.modification_cooldown_hours - hours_since
                    logger.warning(
                        f"Cooldown period for {layer_key}: {remaining:.1f} hours remaining"
                    )
                    return AddRuleResult(
                        success=False,
                        reason=f"Cooldown period: {remaining:.1f} hours remaining",
                    )

            # Check 4: Stratification dependencies
            dependency_violation = self._check_stratification_dependencies(rule, target_layer)
            if dependency_violation:
                logger.warning(f"Stratification violation: {dependency_violation}")
                return AddRuleResult(
                    success=False,
                    reason=f"Stratification violation: {dependency_violation}",
                )

        # Add rule
        rule.stratification_level = target_layer
        self.rules_by_layer[layer_key].append(rule)

        # Update predicate index
        for predicate in rule.new_predicates:
            self.predicate_index[predicate] = layer_key

        # Update last modification time
        self.last_modification_time[layer_key] = datetime.now()

        # Record modification
        self.modification_history.append(
            ModificationEvent(
                timestamp=datetime.now(),
                layer=target_layer,
                action="add_rule",
                rule=rule,
                bypassed_checks=bypass_checks,
                metadata={"is_autonomous": is_autonomous},
            )
        )

        logger.info(f"Successfully added rule to {layer_key} layer: {rule.rule_text[:50]}...")

        return AddRuleResult(
            success=True,
            rule_id=rule.id,
            reason="Rule successfully added",
        )

    def _check_stratification_dependencies(
        self, rule: ASPRule, target_layer: StratificationLevel
    ) -> Optional[str]:
        """
        Check if rule dependencies violate stratification.

        Rules can only depend on predicates from:
        - Same layer or higher (more stable)
        - As specified in policy.can_depend_on

        Args:
            rule: Rule to check
            target_layer: Target layer for this rule

        Returns:
            Error message if violation found, None otherwise
        """
        policy = self.policies[target_layer.value]

        # Extract predicates used by this rule
        used_predicates = rule.predicates_used

        for predicate in used_predicates:
            # Skip if predicate is defined by this rule itself
            if predicate in rule.new_predicates:
                continue

            # Find which layer defines this predicate
            defining_layer_key = self.predicate_index.get(predicate)

            if defining_layer_key is None:
                # New predicate, not yet defined
                continue

            # Convert to enum
            defining_layer = StratificationLevel(defining_layer_key)

            # Check if dependency is allowed
            if defining_layer not in policy.can_depend_on:
                return (
                    f"Rule in {target_layer.value} cannot depend on predicate "
                    f"'{predicate}' from {defining_layer.value}"
                )

        return None

    def get_rules_by_layer(self, layer: StratificationLevel) -> List[ASPRule]:
        """Get all rules in a specific layer."""
        return self.rules_by_layer[layer.value].copy()

    def get_all_rules(self) -> List[ASPRule]:
        """Get all rules across all layers."""
        all_rules = []
        for layer in StratificationLevel:
            all_rules.extend(self.rules_by_layer[layer.value])
        return all_rules

    def get_modification_stats(self) -> Dict[str, ModificationStats]:
        """Get statistics on modifications by layer."""
        stats = {}
        for level in StratificationLevel:
            layer_key = level.value
            policy = self.policies[layer_key]

            # Count modifications
            events = [e for e in self.modification_history if e.layer == level]

            # Calculate cooldown remaining
            cooldown_remaining = None
            last_mod = self.last_modification_time[layer_key]
            if last_mod and policy.modification_cooldown_hours < float("inf"):
                hours_since = (datetime.now() - last_mod).total_seconds() / 3600
                if hours_since < policy.modification_cooldown_hours:
                    cooldown_remaining = policy.modification_cooldown_hours - hours_since

            stats[layer_key] = ModificationStats(
                total_modifications=len(events),
                last_modification=last_mod,
                rules_current=len(self.rules_by_layer[layer_key]),
                cooldown_remaining_hours=cooldown_remaining,
            )

        return stats

    def _find_predicate_layer(self, predicate: str) -> Optional[StratificationLevel]:
        """Find which layer defines a predicate."""
        layer_key = self.predicate_index.get(predicate)
        if layer_key:
            return StratificationLevel(layer_key)
        return None
