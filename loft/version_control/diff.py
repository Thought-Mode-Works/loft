"""
Diff computation for comparing core states.

Computes and formats differences between two versions of the symbolic core.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from .core_state import CoreState, Rule, StratificationLevel


class ChangeType(str, Enum):
    """Type of change in a diff."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class RuleChange:
    """Represents a change to a single rule."""

    change_type: ChangeType
    old_rule: Optional[Rule]
    new_rule: Optional[Rule]

    def summary(self) -> str:
        """Get a one-line summary of this change."""
        if self.change_type == ChangeType.ADDED:
            assert self.new_rule is not None
            return f"+ [{self.new_rule.level.value}] {self.new_rule.content[:50]}"
        elif self.change_type == ChangeType.REMOVED:
            assert self.old_rule is not None
            return f"- [{self.old_rule.level.value}] {self.old_rule.content[:50]}"
        elif self.change_type == ChangeType.MODIFIED:
            assert self.old_rule is not None and self.new_rule is not None
            return f"M [{self.new_rule.level.value}] {self.new_rule.content[:50]}"
        else:
            return "  (unchanged)"


@dataclass
class ConfigChange:
    """Represents a change to configuration."""

    key: str
    old_value: Any
    new_value: Any


@dataclass
class CoreStateDiff:
    """
    Complete diff between two core states.

    Includes all changes to rules, configuration, and metrics.
    """

    from_state_id: str
    to_state_id: str
    rule_changes: List[RuleChange]
    config_changes: List[ConfigChange]
    metric_changes: Dict[str, Tuple[float, float]]  # metric -> (old, new)

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (
            len(self.rule_changes) > 0
            or len(self.config_changes) > 0
            or len(self.metric_changes) > 0
        )

    def count_by_type(self) -> Dict[str, int]:
        """Count changes by type."""
        counts = {
            "added": 0,
            "removed": 0,
            "modified": 0,
        }
        for change in self.rule_changes:
            if change.change_type != ChangeType.UNCHANGED:
                counts[change.change_type.value] += 1
        return counts

    def summary(self) -> str:
        """Generate a summary of the diff."""
        if not self.has_changes():
            return "No changes"

        counts = self.count_by_type()
        parts = []
        if counts["added"] > 0:
            parts.append(f"{counts['added']} rules added")
        if counts["removed"] > 0:
            parts.append(f"{counts['removed']} rules removed")
        if counts["modified"] > 0:
            parts.append(f"{counts['modified']} rules modified")
        if self.config_changes:
            parts.append(f"{len(self.config_changes)} config changes")
        if self.metric_changes:
            parts.append(f"{len(self.metric_changes)} metric changes")

        return ", ".join(parts)

    def format(self, include_unchanged: bool = False) -> str:
        """
        Format diff for display.

        Args:
            include_unchanged: Whether to show unchanged rules

        Returns:
            Formatted diff string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Diff: {self.from_state_id} -> {self.to_state_id}")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"Summary: {self.summary()}")
        lines.append("")

        # Rule changes
        if self.rule_changes:
            lines.append("Rule Changes:")
            lines.append("-" * 60)
            for change in self.rule_changes:
                if not include_unchanged and change.change_type == ChangeType.UNCHANGED:
                    continue
                lines.append(change.summary())
            lines.append("")

        # Configuration changes
        if self.config_changes:
            lines.append("Configuration Changes:")
            lines.append("-" * 60)
            for config_change in self.config_changes:
                lines.append(
                    f"  {config_change.key}: {config_change.old_value} -> {config_change.new_value}"
                )
            lines.append("")

        # Metric changes
        if self.metric_changes:
            lines.append("Metric Changes:")
            lines.append("-" * 60)
            for metric, (old_val, new_val) in self.metric_changes.items():
                delta = new_val - old_val
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"  {metric}: {old_val:.3f} -> {new_val:.3f} ({sign}{delta:.3f})"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_diff(state1: CoreState, state2: CoreState) -> CoreStateDiff:
    """
    Compute diff between two core states.

    Args:
        state1: Old state
        state2: New state

    Returns:
        CoreStateDiff showing all changes
    """
    # Build rule maps for comparison
    rules1_map: Dict[str, Rule] = {r.rule_id: r for r in state1.rules}
    rules2_map: Dict[str, Rule] = {r.rule_id: r for r in state2.rules}

    rule_changes: List[RuleChange] = []

    # Find added and modified rules
    for rule_id, rule2 in rules2_map.items():
        if rule_id not in rules1_map:
            # Added rule
            rule_changes.append(
                RuleChange(change_type=ChangeType.ADDED, old_rule=None, new_rule=rule2)
            )
        else:
            # Check if modified
            rule1 = rules1_map[rule_id]
            if (
                rule1.content != rule2.content
                or rule1.level != rule2.level
                or rule1.confidence != rule2.confidence
            ):
                rule_changes.append(
                    RuleChange(
                        change_type=ChangeType.MODIFIED, old_rule=rule1, new_rule=rule2
                    )
                )

    # Find removed rules
    for rule_id, rule1 in rules1_map.items():
        if rule_id not in rules2_map:
            rule_changes.append(
                RuleChange(
                    change_type=ChangeType.REMOVED, old_rule=rule1, new_rule=None
                )
            )

    # Configuration changes
    config_changes: List[ConfigChange] = []
    all_keys = set(state1.configuration.keys()) | set(state2.configuration.keys())
    for key in all_keys:
        old_val = state1.configuration.get(key)
        new_val = state2.configuration.get(key)
        if old_val != new_val:
            config_changes.append(
                ConfigChange(key=key, old_value=old_val, new_value=new_val)
            )

    # Metric changes
    metric_changes: Dict[str, Tuple[float, float]] = {}
    all_metrics = set(state1.metrics.keys()) | set(state2.metrics.keys())
    for metric in all_metrics:
        old_val = state1.metrics.get(metric, 0.0)
        new_val = state2.metrics.get(metric, 0.0)
        if old_val != new_val:
            metric_changes[metric] = (old_val, new_val)

    return CoreStateDiff(
        from_state_id=state1.state_id,
        to_state_id=state2.state_id,
        rule_changes=rule_changes,
        config_changes=config_changes,
        metric_changes=metric_changes,
    )


def detect_conflicts(state1: CoreState, state2: CoreState) -> List[str]:
    """
    Detect conflicts when merging two states.

    Conflicts occur when:
    - Same rule modified in different ways
    - Constitutional rules differ
    - Critical configuration values differ

    Args:
        state1: First state
        state2: Second state

    Returns:
        List of conflict descriptions
    """
    conflicts: List[str] = []

    # Build rule maps
    rules1_map: Dict[str, Rule] = {r.rule_id: r for r in state1.rules}
    rules2_map: Dict[str, Rule] = {r.rule_id: r for r in state2.rules}

    # Check for conflicting rule modifications
    common_ids = set(rules1_map.keys()) & set(rules2_map.keys())
    for rule_id in common_ids:
        rule1 = rules1_map[rule_id]
        rule2 = rules2_map[rule_id]

        if rule1.content != rule2.content:
            conflicts.append(
                f"Rule {rule_id} has different content: "
                f"'{rule1.content[:30]}...' vs '{rule2.content[:30]}...'"
            )

        if rule1.level != rule2.level:
            conflicts.append(
                f"Rule {rule_id} has different stratification level: "
                f"{rule1.level.value} vs {rule2.level.value}"
            )

    # Check constitutional rules
    const_rules1 = state1.get_rules_by_level(StratificationLevel.CONSTITUTIONAL)
    const_rules2 = state2.get_rules_by_level(StratificationLevel.CONSTITUTIONAL)

    if len(const_rules1) != len(const_rules2):
        conflicts.append(
            f"Different number of constitutional rules: {len(const_rules1)} vs {len(const_rules2)}"
        )

    return conflicts
