"""
Unit tests for A/B testing framework.

Tests variant selection, winner determination, and strategy analysis.
"""

from loft.core.ab_testing import ABTestingFramework, SimpleTestSuite
from loft.core.ab_testing_schemas import (
    RuleVariant,
    SelectionCriteria,
    VariantPerformance,
)
from loft.neural.rule_schemas import GeneratedRule


class TestABTestingSchemas:
    """Test A/B testing schemas."""

    def test_rule_variant_creation(self):
        """Test creating a rule variant."""
        rule = GeneratedRule(
            asp_rule="test(X) :- condition(X).",
            confidence=0.85,
            reasoning="Test rule",
            source_type="gap_fill",
            source_text="Test",
            predicates_used=["test", "condition"],
            new_predicates=["test"],
        )

        variant = RuleVariant(
            variant_id="test_variant",
            rule=rule,
            strategy="balanced",
            description="Test variant",
        )

        assert variant.variant_id == "test_variant"
        assert variant.strategy == "balanced"
        assert variant.rule == rule

    def test_variant_performance_to_dict(self):
        """Test converting variant performance to dict."""
        rule = GeneratedRule(
            asp_rule="test(X) :- condition(X).",
            confidence=0.85,
            reasoning="Test",
            source_type="gap_fill",
            source_text="Test",
            predicates_used=["test", "condition"],
            new_predicates=["test"],
        )

        variant = RuleVariant(variant_id="v1", rule=rule, strategy="balanced", description="Test")

        performance = VariantPerformance(
            variant=variant,
            accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            avg_confidence=0.85,
            num_predicates=2,
        )

        data = performance.to_dict()

        assert data["variant_id"] == "v1"
        assert data["strategy"] == "balanced"
        assert data["accuracy"] == 0.90
        assert data["f1_score"] == 0.865


class TestSimpleTestSuite:
    """Test simple test suite functionality."""

    def test_split_empty_suite(self):
        """Test splitting an empty test suite."""
        suite = SimpleTestSuite([])
        suite1, suite2 = suite.split(0.5)

        assert len(suite1.test_cases) == 0
        assert len(suite2.test_cases) == 0

    def test_split_with_test_cases(self):
        """Test splitting test suite with cases."""
        test_cases = [{"id": i} for i in range(10)]
        suite = SimpleTestSuite(test_cases)

        suite1, suite2 = suite.split(0.2)

        assert len(suite1.test_cases) == 2
        assert len(suite2.test_cases) == 8

    def test_run_on_variant_conservative(self):
        """Test running tests on conservative variant."""
        rule = GeneratedRule(
            asp_rule="test(X) :- condition(X).",
            confidence=0.90,
            reasoning="Test",
            source_type="gap_fill",
            source_text="Test",
            predicates_used=["test", "condition"],
            new_predicates=["test"],
        )

        variant = RuleVariant(
            variant_id="conservative",
            rule=rule,
            strategy="conservative",
            description="Conservative variant",
        )

        suite = SimpleTestSuite()
        results = suite.run_on_variant(variant)

        # Conservative should have higher precision, lower recall
        assert results["precision"] > 0.80
        assert results["accuracy"] > 0.70

    def test_run_on_variant_permissive(self):
        """Test running tests on permissive variant."""
        rule = GeneratedRule(
            asp_rule="test(X) :- condition(X).",
            confidence=0.90,
            reasoning="Test",
            source_type="gap_fill",
            source_text="Test",
            predicates_used=["test", "condition"],
            new_predicates=["test"],
        )

        variant = RuleVariant(
            variant_id="permissive",
            rule=rule,
            strategy="permissive",
            description="Permissive variant",
        )

        suite = SimpleTestSuite()
        results = suite.run_on_variant(variant)

        # Permissive should have higher recall
        assert results["recall"] > 0.85


class TestABTestingFramework:
    """Test A/B testing framework."""

    def create_test_variant(
        self, variant_id: str, strategy: str, confidence: float = 0.85
    ) -> RuleVariant:
        """Create a test rule variant."""
        rule = GeneratedRule(
            asp_rule=f"{variant_id}(X) :- condition(X).",
            confidence=confidence,
            reasoning=f"Test {variant_id}",
            source_type="gap_fill",
            source_text=f"Test {variant_id}",
            predicates_used=[variant_id, "condition"],
            new_predicates=[variant_id],
        )

        return RuleVariant(
            variant_id=variant_id, rule=rule, strategy=strategy, description=f"Test {variant_id}"
        )

    def test_framework_initialization(self):
        """Test framework initializes correctly."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.F1_SCORE
        )

        assert framework.selection_criterion == SelectionCriteria.F1_SCORE
        assert len(framework.test_history) == 0

    def test_single_variant_test(self):
        """Test A/B testing with single variant."""
        framework = ABTestingFramework(test_suite=SimpleTestSuite())

        variant = self.create_test_variant("v1", "balanced", 0.90)

        result = framework.test_variants([variant])

        assert result.winner.variant_id == "v1"
        assert result.confidence_in_winner == 1.0  # Only one variant
        assert result.performance_gap == 0.0

    def test_two_variant_test(self):
        """Test A/B testing with two variants."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.F1_SCORE
        )

        v1 = self.create_test_variant("v1", "conservative", 0.90)
        v2 = self.create_test_variant("v2", "permissive", 0.85)

        result = framework.test_variants([v1, v2])

        assert result.winner.variant_id in ["v1", "v2"]
        assert 0.5 <= result.confidence_in_winner <= 1.0
        assert result.performance_gap >= 0.0

    def test_three_variant_test(self):
        """Test A/B testing with three variants."""
        framework = ABTestingFramework(test_suite=SimpleTestSuite())

        v1 = self.create_test_variant("v1", "conservative", 0.90)
        v2 = self.create_test_variant("v2", "balanced", 0.88)
        v3 = self.create_test_variant("v3", "permissive", 0.85)

        result = framework.test_variants([v1, v2, v3])

        assert result.winner.variant_id in ["v1", "v2", "v3"]
        assert len(result.all_results) == 3

    def test_selection_criterion_accuracy(self):
        """Test winner selection based on accuracy."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.ACCURACY
        )

        v1 = self.create_test_variant("v1", "conservative", 0.95)  # Higher confidence
        v2 = self.create_test_variant("v2", "balanced", 0.80)

        result = framework.test_variants([v1, v2])

        # Higher confidence should win with accuracy criterion
        assert result.winner.variant_id == "v1"

    def test_selection_criterion_simplicity(self):
        """Test winner selection based on simplicity."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.SIMPLICITY
        )

        # Create variants with different predicate counts
        rule1 = GeneratedRule(
            asp_rule="simple(X) :- cond(X).",
            confidence=0.85,
            reasoning="Simple",
            source_type="gap_fill",
            source_text="Test",
            predicates_used=["simple", "cond"],  # 2 predicates
            new_predicates=["simple"],
        )

        rule2 = GeneratedRule(
            asp_rule="complex(X) :- a(X), b(X), c(X).",
            confidence=0.85,
            reasoning="Complex",
            source_type="gap_fill",
            source_text="Test",
            predicates_used=["complex", "a", "b", "c"],  # 4 predicates
            new_predicates=["complex"],
        )

        v1 = RuleVariant(variant_id="simple", rule=rule1, strategy="balanced", description="Simple")
        v2 = RuleVariant(
            variant_id="complex", rule=rule2, strategy="balanced", description="Complex"
        )

        result = framework.test_variants([v1, v2])

        # Simpler rule should win
        assert result.winner.variant_id == "simple"

    def test_test_history_tracking(self):
        """Test that test history is tracked."""
        framework = ABTestingFramework(test_suite=SimpleTestSuite())

        v1 = self.create_test_variant("v1", "balanced", 0.85)

        # Run first test
        framework.test_variants([v1])
        assert len(framework.test_history) == 1

        # Run second test
        framework.test_variants([v1])
        assert len(framework.test_history) == 2

    def test_strategy_performance_analysis(self):
        """Test strategy performance analysis."""
        framework = ABTestingFramework(test_suite=SimpleTestSuite())

        # Run multiple tests with different strategies
        for i in range(3):
            v1 = self.create_test_variant(f"conservative_{i}", "conservative", 0.90)
            v2 = self.create_test_variant(f"permissive_{i}", "permissive", 0.85)
            framework.test_variants([v1, v2])

        stats = framework.analyze_strategy_performance()

        assert "conservative" in stats
        assert "permissive" in stats
        assert stats["conservative"].total_tests == 3
        assert stats["permissive"].total_tests == 3
        assert 0.0 <= stats["conservative"].win_rate <= 1.0

    def test_result_summary(self):
        """Test A/B test result summary generation."""
        framework = ABTestingFramework(test_suite=SimpleTestSuite())

        v1 = self.create_test_variant("v1", "conservative", 0.90)
        v2 = self.create_test_variant("v2", "balanced", 0.85)

        result = framework.test_variants([v1, v2])

        summary = result.summary()

        assert "Winner:" in summary
        assert "Selection criterion:" in summary
        assert "Confidence in winner:" in summary

    def test_get_test_history(self):
        """Test getting test history."""
        framework = ABTestingFramework(test_suite=SimpleTestSuite())

        v1 = self.create_test_variant("v1", "balanced", 0.85)
        framework.test_variants([v1])

        history = framework.get_test_history()

        assert len(history) == 1
        assert isinstance(history[0].winner, RuleVariant)


class TestSelectionCriteria:
    """Test different selection criteria."""

    def test_all_criteria_values(self):
        """Test all selection criteria enum values."""
        assert SelectionCriteria.ACCURACY.value == "accuracy"
        assert SelectionCriteria.PRECISION.value == "precision"
        assert SelectionCriteria.RECALL.value == "recall"
        assert SelectionCriteria.F1_SCORE.value == "f1"
        assert SelectionCriteria.CONFIDENCE.value == "confidence"
        assert SelectionCriteria.SIMPLICITY.value == "simplicity"
