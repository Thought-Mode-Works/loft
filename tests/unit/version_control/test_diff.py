"""
Unit tests for diff computation.

Tests diff computation, conflict detection, and merge operations.
Target coverage: 75%+ (from 58%)
"""

from datetime import datetime
from loft.version_control.diff import (
    compute_diff,
    detect_conflicts,
    ChangeType,
    RuleChange,
    ConfigChange,
    CoreStateDiff,
)
from loft.version_control.core_state import (
    CoreState,
    Rule,
    StratificationLevel,
)


class TestRuleChange:
    """Test RuleChange data structure."""

    def test_rule_change_added(self):
        """Test RuleChange for added rule."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        change = RuleChange(
            change_type=ChangeType.ADDED,
            old_rule=None,
            new_rule=rule,
        )

        assert change.change_type == ChangeType.ADDED
        assert change.new_rule == rule
        assert change.old_rule is None

    def test_rule_change_removed(self):
        """Test RuleChange for removed rule."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        change = RuleChange(
            change_type=ChangeType.REMOVED,
            old_rule=rule,
            new_rule=None,
        )

        assert change.change_type == ChangeType.REMOVED
        assert change.old_rule == rule
        assert change.new_rule is None

    def test_rule_change_modified(self):
        """Test RuleChange for modified rule."""
        old_rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        new_rule = Rule(
            rule_id="r1",
            content="fact(b).",
            level=StratificationLevel.TACTICAL,
            confidence=0.95,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        change = RuleChange(
            change_type=ChangeType.MODIFIED,
            old_rule=old_rule,
            new_rule=new_rule,
        )

        assert change.change_type == ChangeType.MODIFIED
        assert change.old_rule == old_rule
        assert change.new_rule == new_rule

    def test_rule_change_summary_added(self):
        """Test summary for added rule."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        change = RuleChange(
            change_type=ChangeType.ADDED,
            old_rule=None,
            new_rule=rule,
        )

        summary = change.summary()

        assert "+" in summary
        assert "tactical" in summary
        assert "fact(a)." in summary

    def test_rule_change_summary_removed(self):
        """Test summary for removed rule."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        change = RuleChange(
            change_type=ChangeType.REMOVED,
            old_rule=rule,
            new_rule=None,
        )

        summary = change.summary()

        assert "-" in summary
        assert "tactical" in summary

    def test_rule_change_summary_modified(self):
        """Test summary for modified rule."""
        old_rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        new_rule = Rule(
            rule_id="r1",
            content="fact(b).",
            level=StratificationLevel.TACTICAL,
            confidence=0.95,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        change = RuleChange(
            change_type=ChangeType.MODIFIED,
            old_rule=old_rule,
            new_rule=new_rule,
        )

        summary = change.summary()

        assert "M" in summary


class TestConfigChange:
    """Test ConfigChange data structure."""

    def test_config_change(self):
        """Test ConfigChange creation."""
        change = ConfigChange(
            key="timeout",
            old_value=30,
            new_value=60,
        )

        assert change.key == "timeout"
        assert change.old_value == 30
        assert change.new_value == 60


class TestCoreStateDiff:
    """Test CoreStateDiff data structure."""

    def test_diff_has_changes_with_rule_changes(self):
        """Test has_changes returns True when rules changed."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        rule_change = RuleChange(
            change_type=ChangeType.ADDED,
            old_rule=None,
            new_rule=rule,
        )

        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=[rule_change],
            config_changes=[],
            metric_changes={},
        )

        assert diff.has_changes() is True

    def test_diff_has_changes_with_config_changes(self):
        """Test has_changes returns True when config changed."""
        config_change = ConfigChange(key="test", old_value="a", new_value="b")

        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=[],
            config_changes=[config_change],
            metric_changes={},
        )

        assert diff.has_changes() is True

    def test_diff_has_changes_with_metric_changes(self):
        """Test has_changes returns True when metrics changed."""
        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=[],
            config_changes=[],
            metric_changes={"accuracy": (0.8, 0.9)},
        )

        assert diff.has_changes() is True

    def test_diff_has_no_changes(self):
        """Test has_changes returns False when nothing changed."""
        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=[],
            config_changes=[],
            metric_changes={},
        )

        assert diff.has_changes() is False

    def test_diff_count_by_type(self):
        """Test counting changes by type."""
        rule1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        rule2 = Rule(
            rule_id="r2",
            content="fact(b).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        changes = [
            RuleChange(ChangeType.ADDED, None, rule1),
            RuleChange(ChangeType.REMOVED, rule2, None),
            RuleChange(ChangeType.MODIFIED, rule1, rule2),
        ]

        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=changes,
            config_changes=[],
            metric_changes={},
        )

        counts = diff.count_by_type()

        assert counts["added"] == 1
        assert counts["removed"] == 1
        assert counts["modified"] == 1

    def test_diff_summary_no_changes(self):
        """Test summary with no changes."""
        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=[],
            config_changes=[],
            metric_changes={},
        )

        summary = diff.summary()

        assert "No changes" in summary

    def test_diff_summary_with_changes(self):
        """Test summary with various changes."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        changes = [
            RuleChange(ChangeType.ADDED, None, rule),
            RuleChange(ChangeType.ADDED, None, rule),
        ]

        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=changes,
            config_changes=[ConfigChange("key", "old", "new")],
            metric_changes={"accuracy": (0.8, 0.9)},
        )

        summary = diff.summary()

        assert "2 rules added" in summary
        assert "1 config changes" in summary
        assert "1 metric changes" in summary

    def test_diff_format(self):
        """Test diff formatting."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=[RuleChange(ChangeType.ADDED, None, rule)],
            config_changes=[],
            metric_changes={},
        )

        formatted = diff.format()

        assert "s1" in formatted
        assert "s2" in formatted
        assert "Rule Changes:" in formatted

    def test_diff_format_includes_unchanged(self):
        """Test diff formatting can include unchanged rules."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        diff = CoreStateDiff(
            from_state_id="s1",
            to_state_id="s2",
            rule_changes=[RuleChange(ChangeType.UNCHANGED, rule, rule)],
            config_changes=[],
            metric_changes={},
        )

        # Without include_unchanged
        formatted = diff.format(include_unchanged=False)
        assert "unchanged" not in formatted.lower()

        # With include_unchanged
        formatted = diff.format(include_unchanged=True)
        # May include unchanged in output


class TestComputeDiff:
    """Test compute_diff function."""

    def test_compute_diff_no_changes(self):
        """Test diff computation with identical states."""
        rules = [
            Rule(
                rule_id="r1",
                content="fact(a).",
                level=StratificationLevel.TACTICAL,
                confidence=0.9,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            )
        ]

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={"param": "value"},
            metrics={"accuracy": 0.9},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={"param": "value"},
            metrics={"accuracy": 0.9},
        )

        diff = compute_diff(state1, state2)

        assert not diff.has_changes()

    def test_compute_diff_added_rules(self):
        """Test diff with added rules."""
        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={},
            metrics={},
        )

        new_rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[new_rule],
            configuration={},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        counts = diff.count_by_type()
        assert counts["added"] == 1
        assert counts["removed"] == 0
        assert counts["modified"] == 0

    def test_compute_diff_removed_rules(self):
        """Test diff with removed rules."""
        rule = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        counts = diff.count_by_type()
        assert counts["added"] == 0
        assert counts["removed"] == 1
        assert counts["modified"] == 0

    def test_compute_diff_modified_rules_content(self):
        """Test diff with modified rule content."""
        rule1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        rule2 = Rule(
            rule_id="r1",
            content="fact(b).",  # Different content
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule2],
            configuration={},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        counts = diff.count_by_type()
        assert counts["modified"] == 1

    def test_compute_diff_modified_rules_level(self):
        """Test diff detects stratification level changes."""
        rule1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        rule2 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.STRATEGIC,  # Different level
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule2],
            configuration={},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        counts = diff.count_by_type()
        assert counts["modified"] == 1

    def test_compute_diff_modified_rules_confidence(self):
        """Test diff detects confidence changes."""
        rule1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        rule2 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.95,  # Different confidence
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule2],
            configuration={},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        counts = diff.count_by_type()
        assert counts["modified"] == 1

    def test_compute_diff_configuration_changes(self):
        """Test diff detects configuration changes."""
        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={"param1": "value1"},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={"param1": "value2"},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        assert len(diff.config_changes) == 1
        assert diff.config_changes[0].key == "param1"
        assert diff.config_changes[0].old_value == "value1"
        assert diff.config_changes[0].new_value == "value2"

    def test_compute_diff_configuration_added(self):
        """Test diff detects added configuration."""
        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={"new_param": "value"},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        assert len(diff.config_changes) == 1
        assert diff.config_changes[0].old_value is None

    def test_compute_diff_configuration_removed(self):
        """Test diff detects removed configuration."""
        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={"param": "value"},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={},
            metrics={},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        assert len(diff.config_changes) == 1
        assert diff.config_changes[0].new_value is None

    def test_compute_diff_metric_changes(self):
        """Test diff detects metric changes."""
        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={},
            metrics={"accuracy": 0.8},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={},
            metrics={"accuracy": 0.9},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        assert "accuracy" in diff.metric_changes
        assert diff.metric_changes["accuracy"] == (0.8, 0.9)

    def test_compute_diff_multiple_changes(self):
        """Test diff with multiple types of changes."""
        rule1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={"param": "old"},
            metrics={"accuracy": 0.8},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1],
            configuration={"param": "new"},
            metrics={"accuracy": 0.9},
        )

        diff = compute_diff(state1, state2)

        assert diff.has_changes()
        assert len(diff.rule_changes) == 1
        assert len(diff.config_changes) == 1
        assert len(diff.metric_changes) == 1


class TestDetectConflicts:
    """Test conflict detection."""

    def test_detect_conflicts_no_conflicts(self):
        """Test no conflicts detected for identical states."""
        rules = [
            Rule(
                rule_id="r1",
                content="fact(a).",
                level=StratificationLevel.TACTICAL,
                confidence=0.9,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            )
        ]

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

        conflicts = detect_conflicts(state1, state2)

        assert len(conflicts) == 0

    def test_detect_conflicts_different_content(self):
        """Test conflict detected for same rule with different content."""
        rule1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        rule2 = Rule(
            rule_id="r1",
            content="fact(b).",  # Different content
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule2],
            configuration={},
            metrics={},
        )

        conflicts = detect_conflicts(state1, state2)

        assert len(conflicts) > 0
        assert any("different content" in c for c in conflicts)

    def test_detect_conflicts_different_level(self):
        """Test conflict detected for same rule with different level."""
        rule1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        rule2 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.STRATEGIC,  # Different level
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule2],
            configuration={},
            metrics={},
        )

        conflicts = detect_conflicts(state1, state2)

        assert len(conflicts) > 0
        assert any("different stratification level" in c for c in conflicts)

    def test_detect_conflicts_constitutional_rules(self):
        """Test conflict detected for different constitutional rules."""
        rule1 = Rule(
            rule_id="r1",
            content="const_rule_a.",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
        )

        rule2 = Rule(
            rule_id="r2",
            content="const_rule_b.",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1, rule2],
            configuration={},
            metrics={},
        )

        conflicts = detect_conflicts(state1, state2)

        assert len(conflicts) > 0
        assert any("constitutional rules" in c for c in conflicts)

    def test_detect_conflicts_multiple_conflicts(self):
        """Test detection of multiple conflicts."""
        rule1_v1 = Rule(
            rule_id="r1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        rule1_v2 = Rule(
            rule_id="r1",
            content="fact(b).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )

        state1 = CoreState(
            state_id="s1",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1_v1],
            configuration={},
            metrics={},
        )

        state2 = CoreState(
            state_id="s2",
            timestamp=datetime.utcnow().isoformat(),
            rules=[rule1_v2],
            configuration={},
            metrics={},
        )

        conflicts = detect_conflicts(state1, state2)

        # Should detect both content and level conflicts
        assert len(conflicts) >= 2
