"""
Tests for ASP-based reasoning.

Issue #100: Replace Heuristic-Based Predictions with Actual ASP Reasoning

This module tests the ASPReasoner class that performs actual ASP solving
instead of keyword heuristics for predictions.
"""

import pytest
from loft.symbolic.asp_reasoner import ASPReasoner


class TestASPReasonerBasics:
    """Test basic ASP reasoning functionality."""

    @pytest.fixture
    def reasoner(self):
        """Create an ASP reasoner instance."""
        return ASPReasoner()

    def test_empty_knowledge_base_returns_unknown(self, reasoner):
        """Test that empty knowledge base returns unknown prediction."""
        facts = "contract(c1). has_writing(c1, yes)."
        result = reasoner.reason([], facts)

        assert result.prediction == "unknown"
        assert result.satisfiable is True
        assert result.error is None

    def test_simple_positive_rule(self, reasoner):
        """Test simple rule that derives enforceable."""
        rules = ["enforceable(C) :- contract(C), has_writing(C, yes)."]
        facts = "contract(c1). has_writing(c1, yes)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "enforceable"
        assert result.confidence > 0.5
        assert result.satisfiable is True
        assert "enforceable(c1)" in result.derived_atoms

    def test_simple_negative_rule(self, reasoner):
        """Test simple rule that derives unenforceable."""
        rules = ["unenforceable(C) :- contract(C), has_writing(C, no)."]
        facts = "contract(c1). has_writing(c1, no)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "unenforceable"
        assert result.confidence > 0.5
        assert "unenforceable(c1)" in result.derived_atoms

    def test_no_matching_rules(self, reasoner):
        """Test when rules don't match facts."""
        rules = ["enforceable(C) :- contract(C), special_condition(C)."]
        facts = "contract(c1). other_condition(c1)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "unknown"
        assert "enforceable(c1)" not in result.derived_atoms

    def test_rule_with_negation(self, reasoner):
        """Test rule with negation as failure."""
        rules = ["enforceable(C) :- contract(C), not void(C)."]
        facts = "contract(c1)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "enforceable"
        assert "enforceable(c1)" in result.derived_atoms

    def test_rule_with_negation_blocked(self, reasoner):
        """Test rule blocked by negated condition."""
        rules = ["enforceable(C) :- contract(C), not void(C)."]
        facts = "contract(c1). void(c1)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "unknown"
        assert "enforceable(c1)" not in result.derived_atoms


class TestASPReasonerComplexScenarios:
    """Test more complex ASP reasoning scenarios."""

    @pytest.fixture
    def reasoner(self):
        """Create an ASP reasoner instance."""
        return ASPReasoner()

    def test_multiple_rules_chain(self, reasoner):
        """Test chain of rules leading to prediction."""
        rules = [
            "land_contract(C) :- contract(C), subject_matter(C, land).",
            "requires_writing(C) :- land_contract(C).",
            "unenforceable(C) :- requires_writing(C), not has_writing(C, yes).",
        ]
        facts = "contract(c1). subject_matter(c1, land). has_writing(c1, no)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "unenforceable"
        assert "land_contract(c1)" in result.derived_atoms
        assert "requires_writing(c1)" in result.derived_atoms

    def test_statute_of_frauds_scenario(self, reasoner):
        """Test realistic Statute of Frauds scenario."""
        rules = [
            # Land contracts require writing
            "unenforceable(C) :- contract(C), subject_matter(C, land), has_writing(C, no).",
            # Written contracts are enforceable
            "enforceable(C) :- contract(C), has_writing(C, yes).",
        ]
        facts = "contract(c1). subject_matter(c1, land). has_writing(c1, no)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "unenforceable"

    def test_property_law_adverse_possession(self, reasoner):
        """Test property law adverse possession scenario."""
        rules = [
            # Adverse possession requirements
            "adverse_possession_met(C) :- claim(C), occupation_continuous(C, yes), "
            "occupation_open(C, yes), occupation_hostile(C, yes), sufficient_time(C).",
            "sufficient_time(C) :- claim(C), occupation_years(C, Y), statutory_period(C, P), Y >= P.",
            "enforceable(C) :- adverse_possession_met(C).",
        ]
        facts = (
            "claim(c1). occupation_continuous(c1, yes). occupation_open(c1, yes). "
            "occupation_hostile(c1, yes). occupation_years(c1, 25). statutory_period(c1, 20)."
        )

        result = reasoner.reason(rules, facts)

        assert result.prediction == "enforceable"
        assert "adverse_possession_met(c1)" in result.derived_atoms

    def test_conflicting_rules_returns_unknown(self, reasoner):
        """Test that conflicting derivations return unknown."""
        # Note: In ASP, both atoms can be derived if rules allow
        # This tests the reasoner's handling of such cases
        rules = [
            "enforceable(C) :- contract(C), condition_a(C).",
            "unenforceable(C) :- contract(C), condition_b(C).",
        ]
        facts = "contract(c1). condition_a(c1). condition_b(c1)."

        result = reasoner.reason(rules, facts)

        # Both enforceable and unenforceable are derived - should be unknown
        assert result.prediction == "unknown"


class TestASPReasonerStatistics:
    """Test ASP reasoning statistics tracking."""

    @pytest.fixture
    def reasoner(self):
        """Create an ASP reasoner instance."""
        return ASPReasoner()

    def test_stats_initialization(self, reasoner):
        """Test initial stats are zero."""
        stats = reasoner.get_stats()
        assert stats.total_scenarios == 0
        assert stats.correct_predictions == 0
        assert stats.coverage == 0.0

    def test_stats_update_after_prediction(self, reasoner):
        """Test stats update after making prediction."""
        rules = ["enforceable(C) :- contract(C)."]
        facts = "contract(c1)."

        reasoner.make_prediction(rules, facts, expected_outcome="enforceable")

        stats = reasoner.get_stats()
        assert stats.total_scenarios == 1
        assert stats.correct_predictions == 1
        assert stats.get_accuracy() == 1.0

    def test_stats_track_unknown(self, reasoner):
        """Test stats track unknown predictions."""
        reasoner.make_prediction([], "some_fact(x).", expected_outcome="enforceable")

        stats = reasoner.get_stats()
        assert stats.total_scenarios == 1
        assert stats.unknown_predictions == 1
        assert stats.coverage == 0.0

    def test_stats_reset(self, reasoner):
        """Test stats reset."""
        rules = ["enforceable(C) :- contract(C)."]
        reasoner.make_prediction(rules, "contract(c1).", expected_outcome="enforceable")

        reasoner.reset_stats()
        stats = reasoner.get_stats()

        assert stats.total_scenarios == 0

    def test_coverage_calculation(self, reasoner):
        """Test coverage calculation."""
        rules = ["enforceable(C) :- contract(C)."]

        # One definitive, one unknown
        reasoner.make_prediction(rules, "contract(c1).", expected_outcome="enforceable")
        reasoner.make_prediction([], "other(x).", expected_outcome="enforceable")

        stats = reasoner.get_stats()
        assert stats.coverage == 0.5  # 1 definitive out of 2 total


class TestASPReasonerErrorHandling:
    """Test ASP reasoning error handling."""

    @pytest.fixture
    def reasoner(self):
        """Create an ASP reasoner instance."""
        return ASPReasoner()

    def test_invalid_asp_syntax(self, reasoner):
        """Test handling of invalid ASP syntax."""
        rules = ["this is not valid asp"]
        facts = "contract(c1)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "unknown"
        assert result.error is not None

    def test_empty_facts(self, reasoner):
        """Test handling of empty facts."""
        rules = ["enforceable(C) :- contract(C)."]
        facts = ""

        result = reasoner.reason(rules, facts)

        # Should work but return unknown (no contract facts)
        assert result.prediction == "unknown"
        assert result.error is None

    def test_grounding_error(self, reasoner):
        """Test handling of grounding errors."""
        # Unsafe rule that should cause grounding error
        rules = ["result(X) :- input(Y)."]  # X is unsafe
        facts = "input(a)."

        result = reasoner.reason(rules, facts)

        # Should handle error gracefully
        assert result.prediction == "unknown"


class TestASPReasonerRulesFired:
    """Test identification of fired rules."""

    @pytest.fixture
    def reasoner(self):
        """Create an ASP reasoner instance."""
        return ASPReasoner()

    def test_identifies_fired_rules(self, reasoner):
        """Test that fired rules are identified."""
        rules = [
            "enforceable(C) :- contract(C), has_writing(C, yes).",
            "unenforceable(C) :- contract(C), has_writing(C, no).",
        ]
        facts = "contract(c1). has_writing(c1, yes)."

        result = reasoner.reason(rules, facts)

        assert len(result.rules_fired) >= 1
        assert any("enforceable" in rule for rule in result.rules_fired)

    def test_no_rules_fired_when_no_match(self, reasoner):
        """Test no rules fired when facts don't match."""
        rules = ["enforceable(C) :- special_condition(C)."]
        facts = "contract(c1)."

        result = reasoner.reason(rules, facts)

        assert len(result.rules_fired) == 0


class TestASPReasonerIntegration:
    """Integration tests for ASP reasoning with realistic scenarios."""

    @pytest.fixture
    def reasoner(self):
        """Create an ASP reasoner instance."""
        return ASPReasoner()

    def test_sof_001_oral_land_sale(self, reasoner):
        """Test SOF scenario: Oral agreement for land sale."""
        rules = [
            "unenforceable(C) :- contract(C), subject_matter(C, land), has_writing(C, no).",
        ]
        facts = (
            "contract(c1). subject_matter(c1, land). parties(c1, alice, bob). "
            "has_writing(c1, no). sale_amount(c1, 50000)."
        )

        result = reasoner.reason(rules, facts)

        assert result.prediction == "unenforceable"

    def test_sof_002_written_land_sale(self, reasoner):
        """Test SOF scenario: Written agreement for land sale."""
        rules = [
            "enforceable(C) :- contract(C), subject_matter(C, land), has_writing(C, yes).",
            "unenforceable(C) :- contract(C), subject_matter(C, land), has_writing(C, no).",
        ]
        facts = "contract(c1). subject_matter(c1, land). parties(c1, alice, bob). has_writing(c1, yes)."

        result = reasoner.reason(rules, facts)

        assert result.prediction == "enforceable"

    def test_prop_001_adverse_possession(self, reasoner):
        """Test property law: Adverse possession claim."""
        rules = [
            "sufficient_time(C) :- claim(C), occupation_years(C, Y), "
            "statutory_period(C, P), Y >= P.",
            "all_requirements_met(C) :- claim(C), occupation_continuous(C, yes), "
            "occupation_open(C, yes), occupation_notorious(C, yes), "
            "occupation_hostile(C, yes), sufficient_time(C).",
            "enforceable(C) :- all_requirements_met(C).",
        ]
        facts = (
            "claim(claim1). claimant(claim1, alice). property_type(claim1, land). "
            "occupation_years(claim1, 22). occupation_continuous(claim1, yes). "
            "occupation_open(claim1, yes). occupation_notorious(claim1, yes). "
            "occupation_hostile(claim1, yes). taxes_paid(claim1, yes). "
            "statutory_period(claim1, 20)."
        )

        result = reasoner.reason(rules, facts)

        assert result.prediction == "enforceable"
        assert "sufficient_time(claim1)" in result.derived_atoms
        assert "all_requirements_met(claim1)" in result.derived_atoms
