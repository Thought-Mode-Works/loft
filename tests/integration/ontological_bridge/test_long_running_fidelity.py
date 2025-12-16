"""
Long-Running Translation Fidelity Tests

Extended validation of translation fidelity over large batches,
measuring fidelity trends, detecting regressions, and documenting patterns.
"""

import pytest
import sys
from pathlib import Path
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loft.translation.fidelity_tracker import (
    FidelityTracker,
    TranslationResult,
)
from loft.translation.pattern_documenter import TranslationPatternDocumenter
from loft.translation.asp_to_nl import asp_to_nl_statement
from loft.translation.nl_to_asp import NLToASPTranslator
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider

from tests.integration.ontological_bridge.utils.semantic_similarity import (
    SemanticSimilarityCalculator,
)


# Comprehensive set of test ASP rules covering different types
TEST_ASP_RULES = [
    # Basic rules
    "contract_valid(X) :- offer(X), acceptance(X), consideration(X).",
    "breach_occurred(X) :- contract_valid(X), obligation_unfulfilled(X).",
    "damages_owed(X, Y) :- breach_occurred(X), harm_caused(X, Y).",
    # Negation
    "not_liable(X) :- not breach_occurred(X).",
    "exempt(X) :- contract_valid(X), not fraudulent(X).",
    # Facts
    "statute_applies(statute_of_frauds).",
    "legal_doctrine(promissory_estoppel).",
    "jurisdiction(california).",
    # Constraints
    ":- contract_valid(X), fraudulent(X).",
    ":- breach_occurred(X), not damages_calculated(X).",
    # Disjunctions
    "remedy_available(X) :- specific_performance(X) ; damages_owed(X, _).",
    "writing_required(X) :- land_sale(X) ; goods_over_500(X).",
    # Aggregates (simplified)
    "total_damages(X, N) :- claim(X), N = #count{Y : damages_owed(X, Y)}.",
    # Choice rules (simplified representation)
    "accepted(X) | rejected(X) :- offer(X).",
    # Complex rules
    "enforceable(X) :- contract_valid(X), not void_for_fraud(X), not unconscionable(X).",
    "statute_satisfied(X) :- writing_exists(X), signed(X), essential_terms(X).",
    # Property law
    "adverse_possession(X) :- occupation_continuous(X), occupation_hostile(X), years_elapsed(X, Y), Y >= 20.",
    "easement_created(X) :- landlocked(X), common_ownership_history(X).",
    # Additional rule types for comprehensive coverage
    "mutual_assent(X) :- offer(X), acceptance(X), meeting_of_minds(X).",
    "valid_acceptance(X) :- offer(X), mirror_image(X), timely(X).",
]


# Additional ASP rules for extended testing (100+ rules)
def generate_test_rules() -> List[str]:
    """Generate comprehensive test rules covering various patterns."""
    rules = TEST_ASP_RULES.copy()

    # Add variations
    for i in range(20):
        rules.append(f"predicate_{i}(X) :- condition_{i}(X), requirement_{i}(X).")
        rules.append(f"derived_{i}(X, Y) :- base_{i}(X), related_{i}(X, Y).")

    # Add more complex patterns
    for i in range(10):
        rules.append(f"complex_{i}(X) :- cond1_{i}(X), cond2_{i}(X), not exception_{i}(X).")

    # Add facts
    for i in range(20):
        rules.append(f"fact_{i}(entity_{i}).")

    return rules


@pytest.fixture(scope="module")
def translation_setup():
    """Initialize translation components for long-running tests."""
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    # Use Haiku for cost-effectiveness
    provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
    llm = LLMInterface(provider=provider)
    nl_to_asp = NLToASPTranslator(llm_interface=llm)

    return nl_to_asp, asp_to_nl_statement


@pytest.fixture(scope="module")
def similarity_calculator():
    """Initialize semantic similarity calculator."""
    return SemanticSimilarityCalculator()


def measure_asp_equivalence(original: str, back_translated: str) -> float:
    """
    Measure equivalence between original and back-translated ASP.

    Args:
        original: Original ASP text
        back_translated: Back-translated ASP text

    Returns:
        Fidelity score (0.0 to 1.0)
    """
    # Normalize for comparison
    orig_normalized = original.strip().lower().replace(" ", "")
    back_normalized = back_translated.strip().lower().replace(" ", "")

    # Exact match
    if orig_normalized == back_normalized:
        return 1.0

    # Calculate similarity based on character overlap
    # This is a simple heuristic; real implementation could use AST comparison
    orig_chars = set(orig_normalized)
    back_chars = set(back_normalized)

    if not orig_chars and not back_chars:
        return 1.0

    if not orig_chars or not back_chars:
        return 0.0

    overlap = len(orig_chars & back_chars)
    union = len(orig_chars | back_chars)

    char_similarity = overlap / union if union > 0 else 0.0

    # Boost score if key elements match
    key_elements = [":-", "(", ")", ",", "not", "#count"]
    matches = sum(1 for elem in key_elements if elem in original and elem in back_translated)
    element_bonus = matches / len(key_elements) * 0.2

    return min(char_similarity + element_bonus, 1.0)


class TestLongRunningTranslationFidelity:
    """Extended validation of translation fidelity."""

    @pytest.mark.slow
    def test_100_roundtrip_translations(self, translation_setup, similarity_calculator, tmp_path):
        """
        Test fidelity across 100 roundtrip translations.

        Validates:
        - Average fidelity >= 0.70
        - Perfect roundtrip rate >= 0.30
        - No catastrophic failures
        """
        nl_to_asp, asp_to_nl = translation_setup

        test_rules = generate_test_rules()[:100]

        fidelity_tracker = FidelityTracker()
        pattern_documenter = TranslationPatternDocumenter()

        translation_results = []

        for idx, asp_rule in enumerate(test_rules):
            # ASP -> NL
            nl_translation = asp_to_nl(asp_rule)

            # NL -> ASP
            asp_back_result = nl_to_asp.translate(nl_translation)
            asp_back = asp_back_result.asp_code if asp_back_result.asp_code else ""

            # Measure fidelity
            fidelity = measure_asp_equivalence(asp_rule, asp_back)

            result = TranslationResult(
                original=asp_rule,
                translated=nl_translation,
                back_translated=asp_back,
                fidelity=fidelity,
            )
            translation_results.append(result)

            # Document pattern
            pattern_documenter.record_translation(
                original=asp_rule,
                translated=nl_translation,
                back_translated=asp_back,
                fidelity=fidelity,
            )

            # Print progress every 20 translations
            if (idx + 1) % 20 == 0:
                print(f"\nCompleted {idx + 1}/100 translations...")

        # Record snapshot
        snapshot = fidelity_tracker.record_snapshot(
            translation_results,
            metadata={"test": "100_roundtrip", "rule_count": len(test_rules)},
        )

        # Assertions
        print("\n=== Fidelity Results ===")
        print(f"Average Fidelity: {snapshot.avg_fidelity:.2%}")
        print(f"Perfect Roundtrips: {snapshot.perfect_rate:.2%}")
        print(f"Min Fidelity: {snapshot.min_fidelity:.2%}")
        print(f"Max Fidelity: {snapshot.max_fidelity:.2%}")

        assert snapshot.avg_fidelity >= 0.70, (
            f"Average fidelity {snapshot.avg_fidelity:.2%} below threshold 70%"
        )
        assert snapshot.perfect_rate >= 0.30, (
            f"Perfect roundtrip rate {snapshot.perfect_rate:.2%} below threshold 30%"
        )

        # Save reports
        fidelity_tracker.save_history(str(tmp_path / "fidelity_history.json"))
        pattern_guide_path = pattern_documenter.save_guide(
            str(tmp_path / "translation_patterns.md")
        )

        print(f"\nReports saved to {tmp_path}")
        print(f"Pattern guide: {pattern_guide_path}")

    @pytest.mark.slow
    def test_translation_across_rule_types(
        self, translation_setup, similarity_calculator, tmp_path
    ):
        """
        Test translation across different ASP rule types.

        Validates that all rule types achieve reasonable fidelity.
        """
        nl_to_asp, asp_to_nl = translation_setup

        rule_types = {
            "basic": [
                "valid(X) :- condition(X).",
                "derived(X, Y) :- base(X, Y).",
            ],
            "negation": [
                "not_valid(X) :- not condition(X).",
                "exempt(X) :- valid(X), not exception(X).",
            ],
            "fact": [
                "constant_value(42).",
                "legal_doctrine(estoppel).",
            ],
            "constraint": [
                ":- invalid(X), critical(X).",
                ":- conflict(X, Y), X != Y.",
            ],
            "disjunction": [
                "option(X) :- path_a(X) ; path_b(X).",
            ],
        }

        pattern_documenter = TranslationPatternDocumenter()
        results_by_type = {}

        for rtype, rules in rule_types.items():
            type_results = []

            for asp_rule in rules:
                nl_translation = asp_to_nl(asp_rule)
                asp_back_result = nl_to_asp.translate(nl_translation)
                asp_back = asp_back_result.asp_code if asp_back_result.asp_code else ""

                fidelity = measure_asp_equivalence(asp_rule, asp_back)

                type_results.append(
                    TranslationResult(
                        original=asp_rule,
                        translated=nl_translation,
                        back_translated=asp_back,
                        fidelity=fidelity,
                        rule_type=rtype,
                    )
                )

                pattern_documenter.record_translation(
                    original=asp_rule,
                    translated=nl_translation,
                    back_translated=asp_back,
                    fidelity=fidelity,
                )

            results_by_type[rtype] = type_results

        # Analyze patterns
        analysis = pattern_documenter.analyze_patterns()

        print("\n=== Fidelity by Rule Type ===")
        for rtype, fidelity in analysis.fidelity_by_type.items():
            print(f"{rtype}: {fidelity:.2%}")

            # Each rule type should have some success
            assert fidelity >= 0.50, f"Rule type '{rtype}' fidelity {fidelity:.2%} too low"

        # Save pattern guide
        guide_path = pattern_documenter.save_guide(str(tmp_path / "rule_type_patterns.md"))
        print(f"\nPattern guide saved: {guide_path}")

    def test_fidelity_regression_detection(self, translation_setup, tmp_path):
        """
        Test fidelity regression detection.

        Simulates a regression and validates it's detected.
        """
        nl_to_asp, asp_to_nl = translation_setup

        tracker = FidelityTracker()

        # Baseline: good fidelity
        baseline_rules = [
            "valid(X) :- condition(X).",
            "invalid(X) :- not valid(X).",
            "derived(X, Y) :- valid(X), related(X, Y).",
        ]

        baseline_results = []
        for rule in baseline_rules:
            nl = asp_to_nl(rule)
            asp_back_result = nl_to_asp.translate(nl)
            asp_back = asp_back_result.asp_code if asp_back_result.asp_code else ""
            fidelity = measure_asp_equivalence(rule, asp_back)

            baseline_results.append(
                TranslationResult(
                    original=rule,
                    translated=nl,
                    back_translated=asp_back,
                    fidelity=fidelity,
                )
            )

        tracker.record_snapshot(baseline_results, metadata={"phase": "baseline"})

        # Simulate regression: inject low-fidelity results
        regression_results = []
        for i in range(3):
            regression_results.append(
                TranslationResult(
                    original=f"complex_rule_{i}(X).",
                    translated=f"Rule {i} is complex",
                    back_translated=f"different_rule_{i}(Y).",  # Poor match
                    fidelity=0.3,  # Low fidelity
                )
            )

        tracker.record_snapshot(regression_results, metadata={"phase": "regression"})

        # Detect regression
        regression = tracker.detect_regression(threshold=0.05)

        assert regression is not None, "Regression should be detected"
        assert regression.degradation > 0.05, "Degradation should be significant"

        print("\n=== Regression Detected ===")
        print(f"Baseline: {regression.baseline_fidelity:.2%}")
        print(f"Current: {regression.current_fidelity:.2%}")
        print(f"Degradation: {regression.degradation:.2%}")

    def test_fidelity_trend_analysis(self, translation_setup):
        """
        Test fidelity trend analysis.

        Validates trend detection (improving/stable/degrading).
        """
        nl_to_asp, asp_to_nl = translation_setup

        tracker = FidelityTracker()

        # Simulate improving trend
        test_rule = "valid(X) :- condition(X)."

        for i in range(5):
            # Gradually improve fidelity (simulated)
            simulated_fidelity = 0.6 + (i * 0.08)  # 0.6 -> 0.92

            result = TranslationResult(
                original=test_rule,
                translated="X is valid if X has condition",
                back_translated=test_rule,
                fidelity=min(simulated_fidelity, 1.0),
            )

            tracker.record_snapshot([result], metadata={"iteration": i})

        trend = tracker.get_trend(window=5)

        print("\n=== Trend Analysis ===")
        print(f"Trend: {trend}")

        assert trend == "improving", f"Expected improving trend, got {trend}"
