"""
A/B Testing Framework for competing rule formulations.

Enables testing multiple rule variants in parallel to determine which performs
best before incorporating into the symbolic core.
"""

import copy
import statistics
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from loft.core.ab_testing_schemas import (
    ABTestResult,
    RuleVariant,
    SelectionCriteria,
    StrategyStats,
    VariantPerformance,
)
from loft.symbolic.stratification import StratificationLevel


class SimpleTestSuite:
    """
    Simplified test suite for A/B testing.

    In production, would integrate with actual test framework.
    """

    def __init__(self, test_cases: Optional[List[Dict]] = None):
        """
        Initialize test suite.

        Args:
            test_cases: List of test case dictionaries
        """
        self.test_cases = test_cases or []
        self.base_accuracy = 0.85

    def split(self, test_fraction: float = 0.2):
        """
        Split test suite into two parts.

        Args:
            test_fraction: Fraction for first split

        Returns:
            Tuple of (test_suite_1, test_suite_2)
        """
        split_point = int(len(self.test_cases) * test_fraction)
        if split_point == 0:
            split_point = len(self.test_cases) // 2

        suite1 = SimpleTestSuite(self.test_cases[:split_point] if self.test_cases else [])
        suite2 = SimpleTestSuite(self.test_cases[split_point:] if self.test_cases else [])

        return suite1, suite2

    def run_on_variant(self, variant: RuleVariant, core=None):
        """
        Run tests on a rule variant.

        Args:
            variant: Rule variant to test
            core: ASP core (optional, not used in simple version)

        Returns:
            Test results dictionary
        """
        # Simulate test results based on variant strategy
        # In production, would run actual ASP queries

        # Base metrics
        accuracy = self.base_accuracy
        precision = 0.80
        recall = 0.85

        # Adjust based on strategy
        if variant.strategy == "conservative":
            precision += 0.10  # Fewer false positives
            recall -= 0.05  # More false negatives
        elif variant.strategy == "permissive":
            precision -= 0.05  # More false positives
            recall += 0.10  # Fewer false negatives
        # "balanced" stays at baseline

        # Add some variance based on rule confidence
        confidence_factor = variant.rule.confidence
        accuracy *= confidence_factor
        precision *= confidence_factor
        recall *= confidence_factor

        # Clamp to [0, 1]
        accuracy = min(1.0, max(0.0, accuracy))
        precision = min(1.0, max(0.0, precision))
        recall = min(1.0, max(0.0, recall))

        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_confidence": variant.rule.confidence,
            "failures": [],
        }


class ABTestingFramework:
    """
    Framework for A/B testing competing rule formulations.

    Workflow:
    1. Generate multiple rule variants
    2. Create isolated test environments
    3. Run test suite on each variant
    4. Compare performance
    5. Select best variant
    6. Track strategy performance over time
    """

    def __init__(
        self,
        base_core=None,
        test_suite: Optional[SimpleTestSuite] = None,
        selection_criterion: SelectionCriteria = SelectionCriteria.F1_SCORE,
    ):
        """
        Initialize A/B testing framework.

        Args:
            base_core: Base ASP core (optional, for testing)
            test_suite: Test suite for evaluation
            selection_criterion: Criterion for selecting winner
        """
        self.base_core = base_core
        self.test_suite = test_suite or SimpleTestSuite()
        self.selection_criterion = selection_criterion
        self.test_history: List[ABTestResult] = []

        logger.info(f"Initialized ABTestingFramework (criterion: {selection_criterion.value})")

    def test_variants(
        self,
        variants: List[RuleVariant],
        target_layer: Optional[StratificationLevel] = None,
        test_split: float = 0.2,
    ) -> ABTestResult:
        """
        Test multiple rule variants and select the best.

        Args:
            variants: List of rule variants to test
            target_layer: Stratification layer for incorporation (optional)
            test_split: Fraction of test suite to use for A/B (rest for validation)

        Returns:
            ABTestResult with winner and performance comparison
        """
        if len(variants) == 0:
            raise ValueError("Must provide at least one variant to test")

        logger.info(f"Starting A/B test with {len(variants)} variants")

        # Split test suite
        ab_tests, holdout_tests = self.test_suite.split(test_split)

        # Test each variant
        results = []
        for variant in variants:
            logger.debug(f"Testing variant: {variant.variant_id} ({variant.strategy})")
            performance = self._test_single_variant(variant, target_layer, ab_tests)
            results.append(performance)

        # Select winner based on criterion
        winner_performance = self._select_winner(results)
        winner_variant = winner_performance.variant

        # Validate winner on holdout set (if available)
        if holdout_tests.test_cases:
            final_performance = self._test_single_variant(
                winner_variant, target_layer, holdout_tests
            )
            logger.debug(f"Winner validation on holdout: {final_performance.f1_score:.3f}")

        # Calculate performance gap
        sorted_results = sorted(results, key=self._get_score, reverse=True)
        if len(sorted_results) > 1:
            gap = self._get_score(sorted_results[0]) - self._get_score(sorted_results[1])
        else:
            gap = 0.0

        # Calculate confidence in winner
        confidence = self._calculate_winner_confidence(results, winner_performance)

        # Create test ID
        test_id = f"abtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = ABTestResult(
            winner=winner_variant,
            all_results=results,
            selection_criterion=self.selection_criterion,
            confidence_in_winner=confidence,
            performance_gap=gap,
            test_id=test_id,
        )

        self.test_history.append(result)

        logger.info(f"Winner: {winner_variant.variant_id} (strategy: {winner_variant.strategy})")
        logger.info(f"Performance gap: {gap:.3f}, Confidence: {confidence:.2%}")

        return result

    def _test_single_variant(
        self,
        variant: RuleVariant,
        target_layer: Optional[StratificationLevel],
        tests: SimpleTestSuite,
    ) -> VariantPerformance:
        """
        Test a single rule variant.

        Args:
            variant: Rule variant to test
            target_layer: Target stratification layer
            tests: Test suite to run

        Returns:
            Performance metrics for the variant
        """
        # Create isolated copy of core (if available)
        test_core = copy.deepcopy(self.base_core) if self.base_core else None

        # Run tests
        test_results = tests.run_on_variant(variant, test_core)

        # Calculate metrics
        return VariantPerformance(
            variant=variant,
            accuracy=test_results["accuracy"],
            precision=test_results["precision"],
            recall=test_results["recall"],
            f1_score=test_results["f1_score"],
            avg_confidence=test_results["avg_confidence"],
            num_predicates=len(variant.rule.predicates_used),
            test_failures=test_results["failures"],
        )

    def _select_winner(self, results: List[VariantPerformance]) -> VariantPerformance:
        """
        Select winner based on selection criterion.

        Args:
            results: List of variant performances

        Returns:
            Best performing variant
        """
        return max(results, key=self._get_score)

    def _get_score(self, performance: VariantPerformance) -> float:
        """
        Get score for a variant based on selection criterion.

        Args:
            performance: Variant performance metrics

        Returns:
            Score for ranking
        """
        if self.selection_criterion == SelectionCriteria.ACCURACY:
            return performance.accuracy
        elif self.selection_criterion == SelectionCriteria.PRECISION:
            return performance.precision
        elif self.selection_criterion == SelectionCriteria.RECALL:
            return performance.recall
        elif self.selection_criterion == SelectionCriteria.F1_SCORE:
            return performance.f1_score
        elif self.selection_criterion == SelectionCriteria.CONFIDENCE:
            return performance.avg_confidence
        elif self.selection_criterion == SelectionCriteria.SIMPLICITY:
            return 1.0 / (1.0 + performance.num_predicates)  # Prefer fewer predicates
        else:
            raise ValueError(f"Unknown criterion: {self.selection_criterion}")

    def _calculate_winner_confidence(
        self, all_results: List[VariantPerformance], winner: VariantPerformance
    ) -> float:
        """
        Calculate confidence that winner is truly best.

        Uses statistical approach to determine if performance gap is significant.

        Args:
            all_results: All variant performances
            winner: Winner variant performance

        Returns:
            Confidence score (0-1)
        """
        if len(all_results) < 2:
            return 1.0

        # Calculate z-score: (winner - mean_others) / std_others
        winner_score = self._get_score(winner)
        other_scores = [self._get_score(r) for r in all_results if r != winner]

        if not other_scores:
            return 1.0

        mean_others = statistics.mean(other_scores)
        std_others = statistics.stdev(other_scores) if len(other_scores) > 1 else 0.1

        # Convert to confidence (0-1 range)
        z_score = (winner_score - mean_others) / std_others if std_others > 0 else 0
        confidence = min(1.0, max(0.5, 0.5 + z_score / 6))  # Map to [0.5, 1.0]

        return confidence

    def analyze_strategy_performance(self) -> Dict[str, StrategyStats]:
        """
        Analyze which rule generation strategies perform best.

        Learns meta-patterns: "conservative rules work better for X"

        Returns:
            Dictionary mapping strategy name to performance statistics
        """
        strategy_results: Dict[str, Dict] = {}

        for test in self.test_history:
            for result in test.all_results:
                strategy = result.variant.strategy

                if strategy not in strategy_results:
                    strategy_results[strategy] = {"wins": 0, "total": 0, "scores": []}

                strategy_results[strategy]["total"] += 1
                strategy_results[strategy]["scores"].append(self._get_score(result))

                # Check if this variant won
                if result.variant.variant_id == test.winner.variant_id:
                    strategy_results[strategy]["wins"] += 1

        # Calculate statistics
        stats = {}
        for strategy, data in strategy_results.items():
            stats[strategy] = StrategyStats(
                strategy=strategy,
                total_tests=data["total"],
                wins=data["wins"],
                win_rate=data["wins"] / data["total"] if data["total"] > 0 else 0.0,
                avg_score=statistics.mean(data["scores"]) if data["scores"] else 0.0,
            )

        return stats

    def get_test_history(self) -> List[ABTestResult]:
        """
        Get history of all A/B tests.

        Returns:
            List of test results
        """
        return self.test_history.copy()
