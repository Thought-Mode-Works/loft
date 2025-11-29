"""
Tests for RuleGeneralizer class.

Tests for:
- Constant replacement with variables
- Preservation of intentional constants
- Entity extraction from facts
- Validation of generalized rules
"""

from loft.neural.rule_generalizer import (
    RuleGeneralizer,
    GeneralizationResult,
    GeneralizationStats,
    extract_entities_from_facts,
    generalize_rule_with_validation,
)


class TestGeneralizationResult:
    """Test GeneralizationResult dataclass."""

    def test_was_modified_true(self):
        """Result with changes should report modified."""
        result = GeneralizationResult(
            original_rule="enforceable(X) :- claimant(X, alice).",
            generalized_rule="enforceable(X) :- claimant(X, _).",
            changes=["Replaced 'alice' with '_' in claimant()"],
        )
        assert result.was_modified

    def test_was_modified_false(self):
        """Result without changes should report not modified."""
        result = GeneralizationResult(
            original_rule="enforceable(X) :- claimant(X, Y).",
            generalized_rule="enforceable(X) :- claimant(X, Y).",
            changes=[],
        )
        assert not result.was_modified


class TestGeneralizationStats:
    """Test GeneralizationStats dataclass."""

    def test_initial_values(self):
        """Stats should initialize to zero."""
        stats = GeneralizationStats()
        assert stats.rules_processed == 0
        assert stats.rules_modified == 0
        assert stats.constants_replaced == 0

    def test_modification_rate(self):
        """Modification rate should be calculated correctly."""
        stats = GeneralizationStats(rules_processed=10, rules_modified=3)
        assert stats.to_dict()["modification_rate"] == 0.3

    def test_modification_rate_zero_division(self):
        """Modification rate should handle zero processed."""
        stats = GeneralizationStats()
        assert stats.to_dict()["modification_rate"] == 0.0


class TestRuleGeneralizerBasics:
    """Test basic RuleGeneralizer functionality."""

    def test_generalize_known_entity(self):
        """Known entities should be replaced with anonymous variable."""
        generalizer = RuleGeneralizer(known_entities={"alice", "bob"})

        result = generalizer.generalize(
            "enforceable(X) :- claimant(X, alice), occupation_years(X, N), N >= 20."
        )

        assert "claimant(X, _)" in result.generalized_rule
        assert "alice" not in result.generalized_rule
        assert result.was_modified
        assert len(result.changes) == 1

    def test_generalize_multiple_entities(self):
        """Multiple known entities should be replaced."""
        generalizer = RuleGeneralizer(known_entities={"alice", "bob", "c1"})

        result = generalizer.generalize(
            "related(X) :- owner(X, alice), buyer(X, bob), contract(c1)."
        )

        assert "owner(X, _)" in result.generalized_rule
        assert "buyer(X, _)" in result.generalized_rule
        assert "contract(_)" in result.generalized_rule
        assert len(result.changes) == 3

    def test_preserve_variables(self):
        """Variables should not be generalized."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        result = generalizer.generalize("enforceable(X) :- claimant(X, Y), years(X, N).")

        assert result.generalized_rule == "enforceable(X) :- claimant(X, Y), years(X, N)."
        assert not result.was_modified

    def test_preserve_anonymous_variable(self):
        """Anonymous variables should not be changed."""
        generalizer = RuleGeneralizer()

        result = generalizer.generalize("enforceable(X) :- claimant(X, _), years(X, N).")

        assert "claimant(X, _)" in result.generalized_rule
        assert not result.was_modified


class TestPreserveConstants:
    """Test preservation of intentional constants."""

    def test_preserve_yes_no(self):
        """Yes/no values should be preserved."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        result = generalizer.generalize(
            "enforceable(X) :- occupation_continuous(X, yes), occupation_hostile(X, no)."
        )

        assert "occupation_continuous(X, yes)" in result.generalized_rule
        assert "occupation_hostile(X, no)" in result.generalized_rule
        assert not result.was_modified

    def test_preserve_land_goods(self):
        """Property type constants should be preserved."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        result = generalizer.generalize("applies(X) :- property_type(X, land), category(X, goods).")

        assert "property_type(X, land)" in result.generalized_rule
        assert "category(X, goods)" in result.generalized_rule
        assert not result.was_modified

    def test_preserve_numbers(self):
        """Numeric thresholds should be preserved."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        result = generalizer.generalize("enforceable(X) :- years(X, N), N >= 20, amount(X, 500).")

        # Numbers in comparison and as arguments should be preserved
        assert "20" in result.generalized_rule
        assert "500" in result.generalized_rule

    def test_preserve_written_oral(self):
        """Document type constants should be preserved."""
        generalizer = RuleGeneralizer()

        result = generalizer.generalize(
            "valid(X) :- document_type(X, written), contract_form(X, oral)."
        )

        assert "written" in result.generalized_rule
        assert "oral" in result.generalized_rule

    def test_preserve_with_additional_constants(self):
        """Additional preserve constants should work."""
        generalizer = RuleGeneralizer(
            known_entities={"alice"}, additional_preserve={"special_value", "custom_const"}
        )

        result = generalizer.generalize("rule(X) :- type(X, special_value), mode(X, custom_const).")

        assert "special_value" in result.generalized_rule
        assert "custom_const" in result.generalized_rule


class TestEntityDetection:
    """Test heuristic entity detection."""

    def test_detect_short_names(self):
        """Short lowercase names should be detected as entities."""
        generalizer = RuleGeneralizer()

        result = generalizer.generalize("related(X) :- owner(X, bob), buyer(X, sue).")

        # Short names like 'bob', 'sue' should be generalized
        assert "owner(X, _)" in result.generalized_rule
        assert "buyer(X, _)" in result.generalized_rule

    def test_detect_numbered_identifiers(self):
        """Numbered identifiers like c1, claim2 should be detected."""
        generalizer = RuleGeneralizer()

        result = generalizer.generalize("related(X) :- contract(c1), claim(claim2).")

        assert "contract(_)" in result.generalized_rule
        assert "claim(_)" in result.generalized_rule

    def test_skip_legal_terms(self):
        """Common legal terms should not be generalized."""
        generalizer = RuleGeneralizer()

        result = generalizer.generalize("applies(X) :- claim(X), case(Y).")

        # 'claim' and 'case' are predicate names, not arguments to generalize
        # This tests that the regex correctly identifies them as predicates
        assert result.generalized_rule == "applies(X) :- claim(X), case(Y)."


class TestComplexRules:
    """Test generalization of complex rules."""

    def test_multiple_predicates(self):
        """Rules with many predicates should be handled."""
        generalizer = RuleGeneralizer(known_entities={"alice", "c1"})

        result = generalizer.generalize(
            "enforceable(X) :- claim(X), claimant(X, alice), contract(X, c1), "
            "occupation_continuous(X, yes), years(X, N), N >= 20."
        )

        assert "claimant(X, _)" in result.generalized_rule
        assert "contract(X, _)" in result.generalized_rule
        assert "occupation_continuous(X, yes)" in result.generalized_rule
        assert "N >= 20" in result.generalized_rule

    def test_preserve_comparisons(self):
        """Comparison expressions should be preserved."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        result = generalizer.generalize(
            "valid(X) :- years(X, N), threshold(X, T), N >= T, N < 100."
        )

        assert "N >= T" in result.generalized_rule
        assert "N < 100" in result.generalized_rule

    def test_comments_unchanged(self):
        """Comment lines should not be modified."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        result = generalizer.generalize("% This rule handles alice's case")

        assert result.generalized_rule == "% This rule handles alice's case"
        assert not result.was_modified

    def test_empty_rule(self):
        """Empty rules should not be modified."""
        generalizer = RuleGeneralizer()

        result = generalizer.generalize("")

        assert result.generalized_rule == ""
        assert not result.was_modified


class TestExtractEntitiesFromFacts:
    """Test entity extraction from ASP facts."""

    def test_extract_simple_entities(self):
        """Simple entity names should be extracted."""
        facts = """
        claim(c1).
        claimant(c1, alice).
        property(c1, lot42).
        """

        entities = extract_entities_from_facts(facts)

        assert "c1" in entities
        assert "alice" in entities
        assert "lot42" in entities

    def test_exclude_value_constants(self):
        """Value constants should not be extracted as entities."""
        facts = """
        claim(c1).
        occupation_continuous(c1, yes).
        occupation_hostile(c1, no).
        property_type(c1, land).
        """

        entities = extract_entities_from_facts(facts)

        assert "c1" in entities
        assert "yes" not in entities
        assert "no" not in entities
        assert "land" not in entities

    def test_extract_multiple_arguments(self):
        """Entities from multi-argument predicates should be extracted."""
        facts = """
        related(alice, bob).
        contract(c1, buyer, seller).
        """

        entities = extract_entities_from_facts(facts)

        assert "alice" in entities
        assert "bob" in entities
        assert "c1" in entities
        assert "buyer" in entities
        assert "seller" in entities


class TestGeneralizeWithValidation:
    """Test generalization with validation."""

    def test_valid_generalization(self):
        """Valid generalizations should pass."""
        rule = "enforceable(X) :- claimant(X, alice), years(X, N)."
        facts = "claimant(c1, alice). years(c1, 25)."

        result = generalize_rule_with_validation(rule, facts)

        assert result.is_valid
        assert "claimant(X, _)" in result.generalized_rule

    def test_auto_extract_entities(self):
        """Entities should be auto-extracted from facts."""
        rule = "enforceable(X) :- claimant(X, alice), contract(X, c1)."
        facts = "claimant(c1, alice). contract(c1, c1)."

        result = generalize_rule_with_validation(rule, facts, known_entities=None)

        assert "claimant(X, _)" in result.generalized_rule
        assert "contract(X, _)" in result.generalized_rule


class TestStatisticsTracking:
    """Test statistics tracking."""

    def test_stats_track_processing(self):
        """Processing should be tracked."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        generalizer.generalize("rule1(X) :- pred(X, alice).")
        generalizer.generalize("rule2(X) :- pred(X, Y).")

        stats = generalizer.get_stats()
        assert stats.rules_processed == 2
        assert stats.rules_modified == 1

    def test_stats_track_constants(self):
        """Constant replacements should be tracked."""
        generalizer = RuleGeneralizer(known_entities={"alice", "bob"})

        generalizer.generalize("rule(X) :- p1(X, alice), p2(X, bob).")

        stats = generalizer.get_stats()
        assert stats.constants_replaced == 2

    def test_stats_reset(self):
        """Stats should be resettable."""
        generalizer = RuleGeneralizer(known_entities={"alice"})

        generalizer.generalize("rule(X) :- pred(X, alice).")
        generalizer.reset_stats()

        stats = generalizer.get_stats()
        assert stats.rules_processed == 0


class TestGeneralizeRules:
    """Test batch rule generalization."""

    def test_generalize_multiple_rules(self):
        """Multiple rules should be generalized."""
        generalizer = RuleGeneralizer(known_entities={"alice", "bob"})

        rules = [
            "rule1(X) :- owner(X, alice).",
            "rule2(X) :- buyer(X, bob).",
            "rule3(X) :- type(X, land).",
        ]

        results = generalizer.generalize_rules(rules)

        assert len(results) == 3
        assert results[0].was_modified
        assert results[1].was_modified
        assert not results[2].was_modified  # 'land' is preserved
