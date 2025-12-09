"""
Integration tests for A/B testing variant selection.

Tests realistic scenarios with multiple competing rule variants.
"""

from loft.core.ab_testing import ABTestingFramework, SimpleTestSuite
from loft.core.ab_testing_schemas import RuleVariant, SelectionCriteria
from loft.neural.rule_schemas import GeneratedRule


class TestVariantSelectionIntegration:
    """Integration tests for variant selection."""

    def create_realistic_variant(
        self, variant_id: str, strategy: str, confidence: float, complexity: int
    ) -> RuleVariant:
        """
        Create a realistic rule variant.

        Args:
            variant_id: Unique identifier
            strategy: Strategy name
            confidence: Rule confidence
            complexity: Number of predicates (complexity)

        Returns:
            RuleVariant for testing
        """
        # Create predicates based on complexity
        predicates = [f"pred_{i}" for i in range(complexity)]

        rule = GeneratedRule(
            asp_rule=f"{variant_id}(X) :- "
            + ", ".join(f"{p}(X)" for p in predicates)
            + ".",
            confidence=confidence,
            reasoning=f"{strategy} strategy with {complexity} predicates",
            source_type="gap_fill",
            source_text=f"Generated using {strategy} approach",
            predicates_used=[variant_id] + predicates,
            new_predicates=[variant_id],
        )

        return RuleVariant(
            variant_id=variant_id,
            rule=rule,
            strategy=strategy,
            description=f"{strategy} formulation with {complexity} conditions",
        )

    def test_realistic_three_strategy_comparison(self):
        """Test comparing conservative, balanced, and permissive strategies."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.F1_SCORE
        )

        # Conservative: high confidence, more conditions (more specific)
        conservative = self.create_realistic_variant(
            "conservative_rule", "conservative", confidence=0.95, complexity=5
        )

        # Balanced: medium confidence, medium conditions
        balanced = self.create_realistic_variant(
            "balanced_rule", "balanced", confidence=0.88, complexity=3
        )

        # Permissive: lower confidence, fewer conditions (more general)
        permissive = self.create_realistic_variant(
            "permissive_rule", "permissive", confidence=0.82, complexity=2
        )

        result = framework.test_variants([conservative, balanced, permissive])

        # Verify result structure
        assert result.winner is not None
        assert len(result.all_results) == 3
        assert result.confidence_in_winner > 0.0
        assert result.performance_gap >= 0.0

        # Winner should be one of the three
        assert result.winner.variant_id in [
            "conservative_rule",
            "balanced_rule",
            "permissive_rule",
        ]

        print(f"\n{result.summary()}")

    def test_strategy_learning_over_multiple_tests(self):
        """Test that framework learns which strategies perform best."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.F1_SCORE
        )

        # Run 10 A/B tests with different rule formulations
        for i in range(10):
            conservative = self.create_realistic_variant(
                f"cons_{i}", "conservative", confidence=0.92, complexity=4
            )
            balanced = self.create_realistic_variant(
                f"bal_{i}", "balanced", confidence=0.88, complexity=3
            )
            permissive = self.create_realistic_variant(
                f"perm_{i}", "permissive", confidence=0.83, complexity=2
            )

            framework.test_variants([conservative, balanced, permissive])

        # Analyze strategy performance
        stats = framework.analyze_strategy_performance()

        assert len(stats) == 3
        assert "conservative" in stats
        assert "balanced" in stats
        assert "permissive" in stats

        # All strategies should have been tested 10 times
        assert stats["conservative"].total_tests == 10
        assert stats["balanced"].total_tests == 10
        assert stats["permissive"].total_tests == 10

        # Wins should sum to 10 (one winner per test)
        total_wins = (
            stats["conservative"].wins
            + stats["balanced"].wins
            + stats["permissive"].wins
        )
        assert total_wins == 10

        # Win rates should sum to ~1.0 (accounting for rounding)
        total_win_rate = (
            stats["conservative"].win_rate
            + stats["balanced"].win_rate
            + stats["permissive"].win_rate
        )
        assert 0.95 <= total_win_rate <= 1.05

        print("\nStrategy Performance Analysis:")
        for strategy, stat in stats.items():
            print(
                f"  {strategy}: {stat.win_rate:.1%} win rate "
                f"(wins: {stat.wins}/{stat.total_tests}, avg score: {stat.avg_score:.3f})"
            )

    def test_high_confidence_vs_simple_tradeoff(self):
        """Test tradeoff between high confidence complex rules vs simple rules."""
        # Test with F1 criterion (balanced)
        framework_f1 = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.F1_SCORE
        )

        # High confidence but complex
        complex_rule = self.create_realistic_variant(
            "complex", "conservative", confidence=0.95, complexity=6
        )

        # Lower confidence but simple
        simple_rule = self.create_realistic_variant(
            "simple", "permissive", confidence=0.85, complexity=2
        )

        result_f1 = framework_f1.test_variants([complex_rule, simple_rule])

        # Test with simplicity criterion
        framework_simple = ABTestingFramework(
            test_suite=SimpleTestSuite(),
            selection_criterion=SelectionCriteria.SIMPLICITY,
        )

        result_simple = framework_simple.test_variants([complex_rule, simple_rule])

        # With simplicity criterion, simpler rule should win
        assert result_simple.winner.variant_id == "simple"

        print(f"\nF1 winner: {result_f1.winner.variant_id}")
        print(f"Simplicity winner: {result_simple.winner.variant_id}")

    def test_tie_breaking_scenario(self):
        """Test scenario where variants perform very similarly."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.F1_SCORE
        )

        # Create variants with same confidence and complexity
        v1 = self.create_realistic_variant(
            "v1", "balanced", confidence=0.88, complexity=3
        )
        v2 = self.create_realistic_variant(
            "v2", "balanced", confidence=0.88, complexity=3
        )
        v3 = self.create_realistic_variant(
            "v3", "balanced", confidence=0.88, complexity=3
        )

        result = framework.test_variants([v1, v2, v3])

        # Should still select a winner
        assert result.winner is not None

        # Performance gap should be small
        assert result.performance_gap < 0.1

        # Confidence in winner should reflect the tie
        assert result.confidence_in_winner < 0.9  # Not very confident

    def test_dominant_variant_scenario(self):
        """Test scenario with one clearly superior variant."""
        framework = ABTestingFramework(
            test_suite=SimpleTestSuite(), selection_criterion=SelectionCriteria.F1_SCORE
        )

        # Create dominant variant (high confidence, reasonable complexity)
        dominant = self.create_realistic_variant(
            "dominant", "balanced", confidence=0.98, complexity=3
        )

        # Create inferior variants
        inferior1 = self.create_realistic_variant(
            "inferior1", "permissive", confidence=0.70, complexity=2
        )
        inferior2 = self.create_realistic_variant(
            "inferior2", "conservative", confidence=0.75, complexity=5
        )

        result = framework.test_variants([dominant, inferior1, inferior2])

        # Dominant variant should win
        assert result.winner.variant_id == "dominant"

        # Should have high confidence in winner
        assert result.confidence_in_winner > 0.7

        # Should have significant performance gap
        assert result.performance_gap > 0.05

    def test_all_selection_criteria(self):
        """Test that all selection criteria produce valid results."""
        variants = [
            self.create_realistic_variant("v1", "conservative", 0.90, 4),
            self.create_realistic_variant("v2", "balanced", 0.88, 3),
            self.create_realistic_variant("v3", "permissive", 0.85, 2),
        ]

        criteria = [
            SelectionCriteria.ACCURACY,
            SelectionCriteria.PRECISION,
            SelectionCriteria.RECALL,
            SelectionCriteria.F1_SCORE,
            SelectionCriteria.CONFIDENCE,
            SelectionCriteria.SIMPLICITY,
        ]

        print("\nTesting all selection criteria:")
        for criterion in criteria:
            framework = ABTestingFramework(
                test_suite=SimpleTestSuite(), selection_criterion=criterion
            )

            result = framework.test_variants(variants)

            assert result.winner is not None
            assert result.selection_criterion == criterion

            print(f"  {criterion.value}: winner = {result.winner.variant_id}")
