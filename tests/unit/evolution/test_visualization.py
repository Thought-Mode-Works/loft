"""Tests for rule evolution visualization."""

import pytest
from datetime import datetime, timedelta

from loft.evolution.tracking import (
    RuleMetadata,
    ValidationResult,
    ABTestResult,
    DialecticalHistory,
    StratificationLayer,
)
from loft.evolution.visualization import (
    format_rule_history,
    format_genealogy_tree,
    format_performance_chart,
    format_ab_test_dashboard,
    format_stratification_timeline,
    format_rule_diff,
)


@pytest.fixture
def sample_rule():
    """Create a sample rule for testing."""
    rule = RuleMetadata(
        rule_id="test_rule_visualization",
        rule_text="enforceable(C) :- valid(C), signed(C).",
        natural_language="A valid signed contract is enforceable",
        created_by="test",
        version="1.0",
    )
    rule.validation_results.append(
        ValidationResult(
            timestamp=datetime.now(),
            test_cases_evaluated=100,
            passed=85,
            failed=15,
            accuracy=0.85,
        )
    )
    return rule


@pytest.fixture
def sample_rules_history():
    """Create a history of rule versions."""
    rules = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(3):
        rule = RuleMetadata(
            rule_id=f"rule_v{i + 1}",
            rule_text=f"rule version {i + 1}",
            natural_language=f"Description for version {i + 1}",
            created_by="test",
            version=f"{i + 1}.0",
            created_at=base_time + timedelta(days=i * 10),
        )
        rule.validation_results.append(
            ValidationResult(
                timestamp=base_time + timedelta(days=i * 10),
                test_cases_evaluated=100,
                passed=70 + i * 10,
                failed=30 - i * 10,
                accuracy=0.7 + i * 0.1,
            )
        )
        rules.append(rule)

    return rules


class TestFormatRuleHistory:
    """Tests for format_rule_history function."""

    def test_empty_history(self):
        """Test formatting empty history."""
        result = format_rule_history([])

        assert result == "No rule history available."

    def test_single_rule_history(self, sample_rule):
        """Test formatting single rule."""
        result = format_rule_history([sample_rule])

        assert "Rule Evolution:" in result
        assert "v1.0" in result
        assert "OPERATIONAL" in result
        assert "85%" in result

    def test_multiple_rules_history(self, sample_rules_history):
        """Test formatting multiple rule versions."""
        result = format_rule_history(sample_rules_history)

        assert "v1.0" in result
        assert "v2.0" in result
        assert "v3.0" in result
        assert "├─" in result or "└─" in result

    def test_history_with_validation_disabled(self, sample_rule):
        """Test formatting without validation info."""
        result = format_rule_history([sample_rule], show_validation=False)

        assert "Accuracy:" not in result

    def test_history_with_dialectical(self):
        """Test formatting with dialectical history."""
        rule = RuleMetadata(
            rule_id="dialectical_rule",
            rule_text="test rule",
            natural_language="test",
            created_by="test",
        )
        rule.dialectical = DialecticalHistory(
            thesis_rule="original thesis rule text for testing",
            antithesis_critiques=["critique about the rule logic and assumptions"],
            synthesis_rule="improved synthesis rule incorporating feedback",
            cycles_completed=1,
        )

        result = format_rule_history([rule], show_dialectical=True)

        assert "Dialectical" in result or "dialectical" in result.lower()


class TestFormatGenealogyTree:
    """Tests for format_genealogy_tree function."""

    def test_single_node_tree(self, sample_rule):
        """Test formatting single node."""
        result = format_genealogy_tree(sample_rule, {sample_rule.rule_id: sample_rule})

        assert sample_rule.rule_id[:12] in result

    def test_tree_with_children(self):
        """Test formatting tree with children."""
        parent = RuleMetadata(
            rule_id="parent_rule",
            rule_text="parent",
            natural_language="parent rule",
            created_by="test",
        )
        parent.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=80,
                failed=20,
                accuracy=0.8,
            )
        )
        child = RuleMetadata(
            rule_id="child_rule",
            rule_text="child",
            natural_language="child rule",
            created_by="test",
        )
        child.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=85,
                failed=15,
                accuracy=0.85,
            )
        )
        parent.downstream_rules.append(child.rule_id)

        all_rules = {
            parent.rule_id: parent,
            child.rule_id: child,
        }

        result = format_genealogy_tree(parent, all_rules)

        assert "parent_rule" in result
        assert "child_rule" in result
        assert "└──" in result or "├──" in result


class TestFormatPerformanceChart:
    """Tests for format_performance_chart function."""

    def test_no_history(self):
        """Test chart with no accuracy history."""
        rule = RuleMetadata(
            rule_id="no_history",
            rule_text="test",
            natural_language="test",
            created_by="test",
        )

        result = format_performance_chart(rule)

        assert "No accuracy history" in result

    def test_chart_with_history(self):
        """Test chart with accuracy history."""
        rule = RuleMetadata(
            rule_id="with_history",
            rule_text="test",
            natural_language="test",
            created_by="test",
        )
        now = datetime.now()
        rule.accuracy_history = [
            (now - timedelta(days=3), 0.7),
            (now - timedelta(days=2), 0.75),
            (now - timedelta(days=1), 0.8),
            (now, 0.85),
        ]

        result = format_performance_chart(rule)

        assert "Accuracy Over Time" in result
        assert "│" in result
        assert "●" in result or "%" in result


class TestFormatABTestDashboard:
    """Tests for format_ab_test_dashboard function."""

    def test_no_tests(self):
        """Test dashboard with no tests."""
        result = format_ab_test_dashboard([])

        assert "No A/B tests" in result

    def test_single_test(self):
        """Test dashboard with single test."""
        test = ABTestResult(
            test_id="test_001",
            started_at=datetime.now(),
            variant_a_id="rule_v1",
            variant_b_id="rule_v2",
            variant_a_accuracy=0.75,
            variant_b_accuracy=0.82,
            cases_evaluated=100,
        )

        result = format_ab_test_dashboard([test])

        assert "Test #test_001" in result
        assert "Variant A" in result
        assert "Variant B" in result
        assert "75%" in result
        assert "82%" in result

    def test_test_with_winner(self):
        """Test dashboard showing winner."""
        test = ABTestResult(
            test_id="test_002",
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now(),
            variant_a_id="rule_v1",
            variant_b_id="rule_v2",
            variant_a_accuracy=0.70,
            variant_b_accuracy=0.90,
            cases_evaluated=200,
            p_value=0.0001,
            winner="b",
        )

        result = format_ab_test_dashboard([test])

        assert "highly significant" in result
        assert "PROMOTE" in result

    def test_test_with_rules_info(self):
        """Test dashboard with rule metadata."""
        test = ABTestResult(
            test_id="test_003",
            started_at=datetime.now(),
            variant_a_id="rule_v1",
            variant_b_id="rule_v2",
            variant_a_accuracy=0.8,
            variant_b_accuracy=0.8,
            cases_evaluated=50,
        )

        rules = {
            "rule_v1": RuleMetadata(
                rule_id="rule_v1",
                rule_text="r1",
                natural_language="r1",
                created_by="test",
                version="1.0",
            ),
            "rule_v2": RuleMetadata(
                rule_id="rule_v2",
                rule_text="r2",
                natural_language="r2",
                created_by="test",
                version="2.0",
            ),
        }

        result = format_ab_test_dashboard([test], rules)

        assert "v1.0" in result
        assert "v2.0" in result


class TestFormatStratificationTimeline:
    """Tests for format_stratification_timeline function."""

    def test_no_history(self):
        """Test timeline with no history."""
        rule = RuleMetadata(
            rule_id="no_layer_history",
            rule_text="test",
            natural_language="test",
            created_by="test",
        )
        rule.layer_history = []

        result = format_stratification_timeline(rule)

        assert "No stratification history" in result

    def test_single_layer(self):
        """Test timeline with single layer."""
        rule = RuleMetadata(
            rule_id="single_layer",
            rule_text="test",
            natural_language="test",
            created_by="test",
            current_layer=StratificationLayer.TACTICAL,
        )
        rule.layer_history = [(datetime.now(), StratificationLayer.TACTICAL)]

        result = format_stratification_timeline(rule)

        assert "TACTICAL" in result
        assert "█" in result

    def test_layer_progression(self):
        """Test timeline with layer progression."""
        rule = RuleMetadata(
            rule_id="progression",
            rule_text="test",
            natural_language="test",
            created_by="test",
        )
        now = datetime.now()
        rule.layer_history = [
            (now - timedelta(days=20), StratificationLayer.OPERATIONAL),
            (now - timedelta(days=10), StratificationLayer.TACTICAL),
            (now, StratificationLayer.STRATEGIC),
        ]
        rule.current_layer = StratificationLayer.STRATEGIC

        result = format_stratification_timeline(rule)

        assert "OPERATIONAL" in result
        assert "TACTICAL" in result
        assert "STRATEGIC" in result


class TestFormatRuleDiff:
    """Tests for format_rule_diff function."""

    def test_basic_diff(self):
        """Test basic rule diff."""
        rule_a = RuleMetadata(
            rule_id="rule_a",
            rule_text="enforceable(C) :- valid(C).",
            natural_language="v1 description",
            created_by="test",
            version="1.0",
        )
        rule_b = RuleMetadata(
            rule_id="rule_b",
            rule_text="enforceable(C) :- valid(C), signed(C).",
            natural_language="v2 description",
            created_by="test",
            version="2.0",
        )

        result = format_rule_diff(rule_a, rule_b)

        assert "v1.0" in result
        assert "v2.0" in result
        assert "Comparing:" in result

    def test_diff_with_accuracy_change(self):
        """Test diff showing accuracy impact."""
        rule_a = RuleMetadata(
            rule_id="rule_a",
            rule_text="rule v1",
            natural_language="v1",
            created_by="test",
            version="1.0",
        )
        rule_a.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=70,
                failed=30,
                accuracy=0.7,
            )
        )

        rule_b = RuleMetadata(
            rule_id="rule_b",
            rule_text="rule v2",
            natural_language="v2",
            created_by="test",
            version="2.0",
        )
        rule_b.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=85,
                failed=15,
                accuracy=0.85,
            )
        )

        result = format_rule_diff(rule_a, rule_b)

        assert "Impact Analysis" in result
        assert "Accuracy" in result
        assert "70%" in result
        assert "85%" in result
        assert "ACCEPT" in result or "improvement" in result
