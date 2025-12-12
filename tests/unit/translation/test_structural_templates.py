"""
Unit tests for structural template functions in asp_to_nl.py.

Tests the new detect_rule_type() and apply_structural_template() functions
added for issue #219 to improve ASP→NL translation fidelity.
"""

import pytest

from loft.translation.asp_to_nl import (
    STRUCTURAL_TEMPLATES,
    RULE_TYPE_INDICATORS,
    detect_rule_type,
    apply_structural_template,
    asp_to_nl_statement,
    _humanize_predicate_for_structure,
    _get_subject_from_args,
    _get_contract_subject,
    _build_conditions_string,
    _extract_requirement,
    _extract_validity_state,
    _extract_state,
    _apply_structural_translation,
    _parse_body_predicates,
)


class TestStructuralTemplatesDict:
    """Tests for STRUCTURAL_TEMPLATES dictionary."""

    def test_structural_templates_has_required_types(self):
        """Verify all required template types exist."""
        required_types = [
            "requirement",
            "condition",
            "obligation",
            "prohibition",
            "exception",
            "satisfaction",
            "disjunction",
            "conjunction",
            "validity",
            "default",
        ]
        for rule_type in required_types:
            assert any(
                key.startswith(rule_type) for key in STRUCTURAL_TEMPLATES.keys()
            ), f"Missing template type: {rule_type}"

    def test_structural_templates_count(self):
        """Issue #219 requires 10+ structural templates."""
        assert (
            len(STRUCTURAL_TEMPLATES) >= 10
        ), f"Expected at least 10 templates, got {len(STRUCTURAL_TEMPLATES)}"

    def test_structural_templates_are_format_strings(self):
        """All templates should be valid format strings."""
        for name, template in STRUCTURAL_TEMPLATES.items():
            assert isinstance(template, str), f"{name} is not a string"
            # Templates should contain placeholders
            assert "{" in template, f"{name} has no placeholders"


class TestRuleTypeIndicators:
    """Tests for RULE_TYPE_INDICATORS dictionary."""

    def test_has_all_rule_types(self):
        """Verify all expected rule types have indicators."""
        expected_types = [
            "requirement",
            "obligation",
            "prohibition",
            "exception",
            "condition",
            "satisfaction",
            "validity",
            "default",
        ]
        for rule_type in expected_types:
            assert (
                rule_type in RULE_TYPE_INDICATORS
            ), f"Missing indicators for: {rule_type}"

    def test_indicators_are_lists(self):
        """Each indicator set should be a list of strings."""
        for rule_type, indicators in RULE_TYPE_INDICATORS.items():
            assert isinstance(indicators, list), f"{rule_type} indicators not a list"
            for indicator in indicators:
                assert isinstance(
                    indicator, str
                ), f"Non-string indicator in {rule_type}"


class TestDetectRuleType:
    """Tests for detect_rule_type() function."""

    @pytest.mark.parametrize(
        "rule,expected_type",
        [
            # Requirement rules
            ("requires_writing(X) :- land_sale(X).", "requirement"),
            ("requires_signature(X) :- contract(X).", "requirement"),
            # Validity rules
            ("enforceable(X) :- contract(X), signed(X).", "condition"),
            ("valid(X) :- has_offer(X).", "validity"),
            (
                "valid(X) :- has_offer(X), has_acceptance(X), has_consideration(X).",
                "conjunction",
            ),
            # Satisfaction rules
            ("satisfies_sof(X) :- part_performance(X).", "satisfaction"),
            ("satisfies(X) :- writing(X).", "satisfaction"),
            # Exception rules
            ("exception(X) :- specially_manufactured(X).", "exception"),
            ("exempt(X) :- part_performance(X).", "exception"),
            # Prohibition rules (negation in body)
            ("unenforceable(X) :- not has_writing(X).", "prohibition"),
            ("invalid(X) :- not signed(X).", "prohibition"),
            # Default rules
            ("assumed_valid(X) :- not proven_invalid(X).", "default"),
            # Facts (no body)
            ("contract(c1).", "fact"),
            ("merchant(seller1).", "fact"),
            # Disjunction
            ("valid(X) :- option1(X); option2(X).", "disjunction"),
        ],
    )
    def test_detect_rule_type(self, rule, expected_type):
        """Test rule type detection for various patterns."""
        result = detect_rule_type(rule)
        assert (
            result == expected_type
        ), f"Expected {expected_type} for '{rule}', got {result}"

    def test_detect_rule_type_handles_whitespace(self):
        """Rule detection should handle various whitespace patterns."""
        rules = [
            "requires_writing(X):-land_sale(X).",
            "requires_writing(X) :- land_sale(X).",
            "requires_writing(X)  :-  land_sale(X).",
        ]
        for rule in rules:
            assert detect_rule_type(rule) == "requirement"

    def test_detect_rule_type_case_insensitive(self):
        """Rule detection should be case-insensitive."""
        assert detect_rule_type("REQUIRES_WRITING(X) :- LAND_SALE(X).") == "requirement"
        assert detect_rule_type("Enforceable(X) :- Contract(X).") == "validity"


class TestApplyStructuralTemplate:
    """Tests for apply_structural_template() function."""

    def test_requirement_template_with_writing(self):
        """Requirement rules about writing produce proper sentences."""
        result = apply_structural_template(
            rule_type="requirement",
            head_predicate="requires_writing",
            head_args=["X"],
            body_predicates=[("land_sale", ["X"])],
            original_asp="requires_writing(X) :- land_sale(X).",
        )
        assert "must be in writing" in result.lower()
        assert "land sale" in result.lower()

    def test_satisfaction_template(self):
        """Satisfaction rules produce proper sentences."""
        result = apply_structural_template(
            rule_type="satisfaction",
            head_predicate="satisfies_sof",
            head_args=["X"],
            body_predicates=[("part_performance", ["X"])],
            original_asp="satisfies_sof(X) :- part_performance(X).",
        )
        assert "satisfy" in result.lower()
        assert "statute of frauds" in result.lower()

    def test_validity_template(self):
        """Validity rules produce proper sentences."""
        result = apply_structural_template(
            rule_type="validity",
            head_predicate="enforceable",
            head_args=["X"],
            body_predicates=[("signed", ["X"])],
            original_asp="enforceable(X) :- signed(X).",
        )
        assert "enforceable" in result.lower()

    def test_prohibition_template(self):
        """Prohibition rules produce proper sentences."""
        result = apply_structural_template(
            rule_type="prohibition",
            head_predicate="unenforceable",
            head_args=["X"],
            body_predicates=[("not_has_writing", ["X"])],
            original_asp="unenforceable(X) :- not has_writing(X).",
        )
        assert "not" in result.lower()

    def test_conjunction_template(self):
        """Conjunction rules produce proper sentences with 'and'."""
        result = apply_structural_template(
            rule_type="conjunction",
            head_predicate="valid",
            head_args=["X"],
            body_predicates=[
                ("has_offer", ["X"]),
                ("has_acceptance", ["X"]),
                ("has_consideration", ["X"]),
            ],
            original_asp="valid(X) :- has_offer(X), has_acceptance(X), has_consideration(X).",
        )
        assert "and" in result.lower()
        assert "offer" in result.lower()
        assert "acceptance" in result.lower()


class TestHelperFunctions:
    """Tests for helper functions used in structural templates."""

    def test_humanize_predicate_for_structure(self):
        """Test predicate humanization."""
        assert _humanize_predicate_for_structure("has_offer") == "offer"
        # "writing" maps to "a written document" via replacements
        assert "written" in _humanize_predicate_for_structure("requires_writing")
        assert _humanize_predicate_for_structure("is_valid") == "valid"
        assert "statute of frauds" in _humanize_predicate_for_structure("sof")

    def test_get_subject_from_args(self):
        """Test subject extraction from arguments."""
        assert _get_subject_from_args(["X"]) == "a contract"
        assert _get_subject_from_args(["contract1"]) == "the contract"
        assert _get_subject_from_args([]) == "the contract"

    def test_get_contract_subject(self):
        """Test contract subject extraction from body predicates."""
        assert "Land sale" in _get_contract_subject([("land_sale", ["X"])])
        assert "Suretyship" in _get_contract_subject([("suretyship", ["X"])])
        assert "goods over $500" in _get_contract_subject([("goods_over_500", ["X"])])

    def test_build_conditions_string_single(self):
        """Test building conditions string with single condition."""
        result = _build_conditions_string([("has_offer", ["X"])])
        assert "offer" in result.lower()

    def test_build_conditions_string_multiple(self):
        """Test building conditions string with multiple conditions."""
        result = _build_conditions_string(
            [
                ("has_offer", ["X"]),
                ("has_acceptance", ["X"]),
            ]
        )
        assert "and" in result.lower()

    def test_extract_requirement(self):
        """Test requirement extraction."""
        assert "statute of frauds" in _extract_requirement("satisfies_sof")
        assert "writing" in _extract_requirement("writing_requirement")

    def test_extract_validity_state(self):
        """Test validity state extraction."""
        assert _extract_validity_state("enforceable") == "enforceable"
        assert _extract_validity_state("unenforceable") == "unenforceable"
        assert _extract_validity_state("valid_contract") == "valid"

    def test_extract_state(self):
        """Test general state extraction."""
        assert _extract_state("is_valid") == "valid"
        assert _extract_state("has_offer") == "offer"


class TestApplyStructuralTranslation:
    """Tests for _apply_structural_translation() function."""

    def test_returns_string_for_valid_rule(self):
        """Should return string for valid ASP rule."""
        result = _apply_structural_translation("requires_writing(X) :- land_sale(X).")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_none_for_fact(self):
        """Should return None for facts (no rule body)."""
        result = _apply_structural_translation("contract(c1).")
        assert result is None

    def test_handles_complex_rules(self):
        """Should handle rules with multiple conditions."""
        result = _apply_structural_translation(
            "valid(X) :- has_offer(X), has_acceptance(X), has_consideration(X)."
        )
        assert result is not None
        assert "and" in result.lower()


class TestParseBodyPredicates:
    """Tests for _parse_body_predicates() function."""

    def test_single_predicate(self):
        """Parse single body predicate."""
        result = _parse_body_predicates("land_sale(X)")
        assert len(result) == 1
        assert result[0][0] == "land_sale"
        assert result[0][1] == ["X"]

    def test_multiple_predicates(self):
        """Parse multiple body predicates."""
        result = _parse_body_predicates("has_offer(X), has_acceptance(X)")
        assert len(result) == 2

    def test_negation_handling(self):
        """Parse predicates with negation."""
        result = _parse_body_predicates("not has_writing(X)")
        assert len(result) == 1
        assert result[0][0] == "not_has_writing"

    def test_skips_comparisons(self):
        """Skip comparison operators."""
        result = _parse_body_predicates("price(X, P), P >= 500")
        assert len(result) == 1
        assert result[0][0] == "price"


class TestAspToNlStatementWithStructuralTemplates:
    """Integration tests for asp_to_nl_statement with structural templates."""

    @pytest.mark.parametrize(
        "asp_code,expected_fragments",
        [
            (
                "requires_writing(X) :- land_sale(X).",
                ["land sale", "writing"],
            ),
            (
                "enforceable(X) :- contract(X), satisfies_sof(X).",
                ["enforceable"],
            ),
            (
                "valid(X) :- has_offer(X), has_acceptance(X), has_consideration(X).",
                ["offer", "acceptance", "consideration"],
            ),
            (
                "satisfies_sof(X) :- part_performance(X).",
                ["satisfy", "statute of frauds"],
            ),
        ],
    )
    def test_produces_expected_fragments(self, asp_code, expected_fragments):
        """Test that output contains expected fragments."""
        result = asp_to_nl_statement(asp_code, use_structural_templates=True)
        result_lower = result.lower()
        for fragment in expected_fragments:
            assert (
                fragment.lower() in result_lower
            ), f"Missing '{fragment}' in: {result}"

    def test_can_disable_structural_templates(self):
        """Test that structural templates can be disabled."""
        asp = "custom_predicate(X) :- other_predicate(X)."
        with_templates = asp_to_nl_statement(asp, use_structural_templates=True)
        without_templates = asp_to_nl_statement(asp, use_structural_templates=False)
        # Both should produce valid output
        assert len(with_templates) > 0
        assert len(without_templates) > 0

    def test_fallback_on_exact_match(self):
        """Exact template matches should take precedence."""
        # This rule has an exact match in RULE_STATEMENT_TEMPLATES
        asp = "requires_writing(X) :- land_sale(X)."
        result = asp_to_nl_statement(asp)
        assert "Land sale contracts must be in writing" in result
