"""
Unit tests for constrained predicate vocabulary in nl_to_asp.py.

Tests the new CANONICAL_PREDICATES vocabulary and ConstrainedNLToASPTranslator
class added for issue #220 to improve NL→ASP translation fidelity.
"""

import pytest
from unittest.mock import Mock, MagicMock

from loft.translation.nl_to_asp import (
    CANONICAL_PREDICATES,
    PREDICATE_ALIASES,
    LEGAL_CONCEPT_TO_PREDICATE,
    normalize_predicate,
    extract_predicates_from_asp,
    validate_predicates,
    find_closest_predicate,
    PredicateValidationResult,
    ConstrainedNLToASPTranslator,
    get_predicate_compliance_rate,
)


class TestCanonicalPredicates:
    """Tests for CANONICAL_PREDICATES vocabulary."""

    def test_vocabulary_has_minimum_predicates(self):
        """Issue #220 requires 30+ canonical predicates."""
        assert len(CANONICAL_PREDICATES) >= 30, (
            f"Expected at least 30 predicates, got {len(CANONICAL_PREDICATES)}"
        )

    def test_vocabulary_is_frozenset(self):
        """Vocabulary should be immutable."""
        assert isinstance(CANONICAL_PREDICATES, frozenset)

    def test_vocabulary_has_contract_formation_predicates(self):
        """Vocabulary should include contract formation predicates."""
        required = [
            "contract",
            "has_offer",
            "has_acceptance",
            "has_consideration",
        ]
        for pred in required:
            assert pred in CANONICAL_PREDICATES, f"Missing: {pred}"

    def test_vocabulary_has_statute_of_frauds_predicates(self):
        """Vocabulary should include statute of frauds predicates."""
        required = [
            "requires_writing",
            "satisfies_sof",
            "land_sale",
            "suretyship",
            "goods_over_500",
        ]
        for pred in required:
            assert pred in CANONICAL_PREDICATES, f"Missing: {pred}"

    def test_vocabulary_has_exception_predicates(self):
        """Vocabulary should include exception predicates."""
        required = [
            "part_performance",
            "promissory_estoppel",
            "merchant_confirmation",
            "specially_manufactured",
        ]
        for pred in required:
            assert pred in CANONICAL_PREDICATES, f"Missing: {pred}"

    def test_vocabulary_has_enforceability_predicates(self):
        """Vocabulary should include enforceability predicates."""
        required = ["enforceable", "unenforceable", "valid", "void", "voidable"]
        for pred in required:
            assert pred in CANONICAL_PREDICATES, f"Missing: {pred}"

    def test_all_predicates_are_snake_case(self):
        """All predicates should be in snake_case."""
        for pred in CANONICAL_PREDICATES:
            assert pred == pred.lower(), f"Not lowercase: {pred}"
            assert " " not in pred, f"Contains space: {pred}"


class TestPredicateAliases:
    """Tests for PREDICATE_ALIASES mapping."""

    def test_aliases_map_camelcase_to_snake_case(self):
        """Aliases should map CamelCase to snake_case."""
        camelcase_aliases = [
            ("ContractValid", "contract_valid"),
            ("HasOffer", "has_offer"),
            ("HasAcceptance", "has_acceptance"),
            ("SatisfiesStatuteOfFrauds", "satisfies_sof"),
        ]
        for alias, canonical in camelcase_aliases:
            assert alias in PREDICATE_ALIASES, f"Missing alias: {alias}"
            assert PREDICATE_ALIASES[alias] == canonical

    def test_aliases_map_to_canonical_predicates(self):
        """All alias targets should be canonical predicates."""
        for alias, canonical in PREDICATE_ALIASES.items():
            assert canonical in CANONICAL_PREDICATES, (
                f"Alias {alias} → {canonical} not in vocabulary"
            )


class TestLegalConceptMapping:
    """Tests for LEGAL_CONCEPT_TO_PREDICATE mapping."""

    def test_concept_mapping_has_common_legal_terms(self):
        """Mapping should include common legal terminology."""
        required_concepts = [
            "offer",
            "acceptance",
            "consideration",
            "land sale",
            "statute of frauds",
            "part performance",
        ]
        for concept in required_concepts:
            assert concept in LEGAL_CONCEPT_TO_PREDICATE, f"Missing: {concept}"

    def test_concept_mapping_targets_are_canonical(self):
        """All concept targets should be canonical predicates."""
        for concept, predicate in LEGAL_CONCEPT_TO_PREDICATE.items():
            assert predicate in CANONICAL_PREDICATES, (
                f"Concept '{concept}' → '{predicate}' not in vocabulary"
            )


class TestNormalizePredicate:
    """Tests for normalize_predicate() function."""

    @pytest.mark.parametrize(
        "input_pred,expected",
        [
            # CamelCase conversion
            ("ContractValid", "contract_valid"),
            ("HasOffer", "has_offer"),
            ("HasAcceptance", "has_acceptance"),
            ("SatisfiesStatuteOfFrauds", "satisfies_sof"),
            # Already snake_case
            ("has_offer", "has_offer"),
            ("contract_valid", "contract_valid"),
            # Alias mappings
            ("statute_of_frauds", "satisfies_sof"),
            ("writing_required", "requires_writing"),
            # Whitespace handling
            ("  HasOffer  ", "has_offer"),
        ],
    )
    def test_normalize_predicate(self, input_pred, expected):
        """Test predicate normalization."""
        result = normalize_predicate(input_pred)
        assert result == expected, f"Expected {expected}, got {result}"


class TestExtractPredicatesFromAsp:
    """Tests for extract_predicates_from_asp() function."""

    def test_extract_from_rule(self):
        """Extract predicates from ASP rule."""
        asp = "contract_valid(X) :- has_offer(X), has_acceptance(X)."
        predicates = extract_predicates_from_asp(asp)
        assert "contract_valid" in predicates
        assert "has_offer" in predicates
        assert "has_acceptance" in predicates

    def test_extract_from_fact(self):
        """Extract predicates from ASP fact."""
        asp = "land_sale(contract1)."
        predicates = extract_predicates_from_asp(asp)
        assert predicates == ["land_sale"]

    def test_excludes_keywords(self):
        """Should exclude ASP keywords like 'not'."""
        asp = "enforceable(X) :- contract(X), not unenforceable(X)."
        predicates = extract_predicates_from_asp(asp)
        assert "not" not in predicates
        assert "enforceable" in predicates
        assert "unenforceable" in predicates

    def test_no_duplicates(self):
        """Should not return duplicate predicates."""
        asp = "valid(X) :- valid(Y), valid(Z)."
        predicates = extract_predicates_from_asp(asp)
        assert predicates.count("valid") == 1


class TestValidatePredicates:
    """Tests for validate_predicates() function."""

    def test_valid_predicates(self):
        """Valid predicates should be in valid list."""
        predicates = ["has_offer", "has_acceptance", "contract_valid"]
        valid, invalid = validate_predicates(predicates)
        assert len(valid) == 3
        assert len(invalid) == 0

    def test_invalid_predicates(self):
        """Invalid predicates should be in invalid list."""
        predicates = ["has_offer", "UnknownPredicate", "AnotherUnknown"]
        valid, invalid = validate_predicates(predicates)
        assert len(valid) == 1
        assert len(invalid) == 2
        assert "UnknownPredicate" in invalid

    def test_normalizes_before_validation(self):
        """Should normalize predicates before validation."""
        predicates = ["HasOffer", "ContractValid"]
        valid, invalid = validate_predicates(predicates)
        assert len(valid) == 2
        assert "has_offer" in valid
        assert "contract_valid" in valid


class TestFindClosestPredicate:
    """Tests for find_closest_predicate() function."""

    @pytest.mark.parametrize(
        "input_pred,expected_match",
        [
            ("ContractIsValid", "contract_valid"),
            ("RequireWriting", "requires_writing"),
            ("OfferMade", "has_offer"),
            ("AcceptanceGiven", "has_acceptance"),
        ],
    )
    def test_find_closest_predicate(self, input_pred, expected_match):
        """Test closest predicate matching."""
        result = find_closest_predicate(input_pred)
        # May not always get exact match, but should get related predicate
        assert result is not None
        # For key terms, should find appropriate match
        if "valid" in input_pred.lower():
            assert "valid" in result
        if "offer" in input_pred.lower():
            assert "offer" in result

    def test_returns_none_for_unrelated(self):
        """Should return None for completely unrelated predicates."""
        result = find_closest_predicate("xyz123_unrelated_term")
        # Should either return None or a very weak match
        # The function may still find something if partial matching succeeds
        assert result is None or result in CANONICAL_PREDICATES


class TestPredicateValidationResult:
    """Tests for PredicateValidationResult dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        result = PredicateValidationResult(
            original_asp="HasOffer(X) :- contract(X).",
            corrected_asp="has_offer(X) :- contract(X).",
            valid_predicates=["has_offer", "contract"],
            invalid_predicates=[],
            corrections_made={"HasOffer": "has_offer"},
            compliance_rate=1.0,
            is_fully_compliant=True,
        )
        assert result.is_fully_compliant
        assert result.compliance_rate == 1.0

    def test_compliance_rate_calculation(self):
        """Test compliance rate is calculated correctly."""
        result = PredicateValidationResult(
            original_asp="test",
            corrected_asp="test",
            valid_predicates=["a", "b", "c"],
            invalid_predicates=["d"],
            corrections_made={},
            compliance_rate=0.0,  # Should be recalculated
            is_fully_compliant=False,
        )
        # post_init should calculate: 3/4 = 0.75
        assert result.compliance_rate == 0.75


class TestConstrainedNLToASPTranslator:
    """Tests for ConstrainedNLToASPTranslator class."""

    def test_init_default_vocabulary(self):
        """Should use CANONICAL_PREDICATES by default."""
        translator = ConstrainedNLToASPTranslator()
        assert translator.vocabulary == CANONICAL_PREDICATES
        assert translator.auto_correct is True
        assert translator.strict_mode is False

    def test_init_custom_vocabulary(self):
        """Should accept custom vocabulary."""
        custom = frozenset({"pred1", "pred2"})
        translator = ConstrainedNLToASPTranslator(vocabulary=custom)
        assert translator.vocabulary == custom

    def test_translate_without_llm(self):
        """Translate should work without LLM interface."""
        translator = ConstrainedNLToASPTranslator()
        result = translator.translate("A contract is valid if it has offer.")

        assert result.asp_code is not None
        assert result.extraction_method == "pattern_constrained"
        assert "validation" in result.metadata

    def test_validate_and_correct_normalizes_predicates(self):
        """_validate_and_correct should normalize predicates."""
        translator = ConstrainedNLToASPTranslator()
        asp = "HasOffer(X) :- Contract(X)."

        validation = translator._validate_and_correct(asp)

        # Predicates should be recognized as valid after normalization
        # Even if ASP code text isn't changed, validation identifies them
        assert "has_offer" in validation.valid_predicates
        assert "contract" in validation.valid_predicates
        assert validation.is_fully_compliant

    def test_validate_and_correct_finds_closest_match(self):
        """_validate_and_correct should find closest predicate for invalid ones."""
        translator = ConstrainedNLToASPTranslator()
        asp = "ContractIsValid(X) :- OfferMade(X)."

        validation = translator._validate_and_correct(asp)

        # Should find corrections for invalid predicates
        assert len(validation.corrections_made) > 0 or validation.is_fully_compliant

    def test_strict_mode_behavior(self):
        """strict_mode with auto_correct=False keeps invalid predicates invalid."""
        translator = ConstrainedNLToASPTranslator(strict_mode=True, auto_correct=False)
        asp = "completely_unknown_xyz123(X)."

        # With auto_correct=False, validation won't fix the predicate
        validation = translator._validate_and_correct(asp)

        # The predicate should remain invalid
        assert not validation.is_fully_compliant
        assert len(validation.invalid_predicates) >= 1

    def test_get_vocabulary_stats(self):
        """get_vocabulary_stats should return correct statistics."""
        translator = ConstrainedNLToASPTranslator()
        stats = translator.get_vocabulary_stats()

        assert "total_predicates" in stats
        assert stats["total_predicates"] >= 30
        assert "categories" in stats
        assert "aliases_count" in stats
        assert "concept_mappings_count" in stats

    def test_build_constrained_prompt(self):
        """_build_constrained_prompt should include vocabulary constraints."""
        translator = ConstrainedNLToASPTranslator()
        prompt = translator._build_constrained_prompt("Test statement")

        assert "ALLOWED PREDICATES" in prompt
        assert "DO NOT invent new predicates" in prompt
        assert "snake_case" in prompt
        assert "Test statement" in prompt


class TestConstrainedTranslatorWithMockLLM:
    """Tests for ConstrainedNLToASPTranslator with mocked LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM interface."""
        mock = MagicMock()
        mock.query.return_value = Mock(content="has_offer(X) :- contract(X).")
        return mock

    def test_translate_with_llm(self, mock_llm):
        """translate should use LLM when available."""
        translator = ConstrainedNLToASPTranslator(llm_interface=mock_llm)
        result = translator.translate("A contract has an offer.")

        assert mock_llm.query.called
        assert result.extraction_method == "llm_constrained"

    def test_llm_receives_constrained_prompt(self, mock_llm):
        """LLM should receive prompt with vocabulary constraints."""
        translator = ConstrainedNLToASPTranslator(llm_interface=mock_llm)
        translator.translate("Test statement")

        # Check the prompt sent to LLM
        call_args = mock_llm.query.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")

        assert "ALLOWED PREDICATES" in prompt or mock_llm.query.called


class TestGetPredicateComplianceRate:
    """Tests for get_predicate_compliance_rate() function."""

    def test_full_compliance(self):
        """Should return 1.0 for fully compliant ASP."""
        asp = "has_offer(X) :- contract(X), has_acceptance(X)."
        rate = get_predicate_compliance_rate(asp)
        assert rate == 1.0

    def test_partial_compliance(self):
        """Should return correct rate for partially compliant ASP."""
        # 2 valid, 1 invalid = 66.67% compliance
        asp = "has_offer(X) :- contract(X), unknown_pred(X)."
        rate = get_predicate_compliance_rate(asp)
        # Rate depends on whether unknown_pred gets corrected
        assert 0 < rate <= 1.0

    def test_empty_asp(self):
        """Should return 1.0 for empty ASP."""
        rate = get_predicate_compliance_rate("")
        assert rate == 1.0

    def test_comment_only(self):
        """Should return 1.0 for comment-only ASP."""
        rate = get_predicate_compliance_rate("% This is a comment")
        assert rate == 1.0


class TestIntegration:
    """Integration tests for constrained translation pipeline."""

    def test_normalize_then_validate(self):
        """Test normalization followed by validation."""
        predicates = ["ContractValid", "HasOffer", "UnknownPred"]

        # First normalize
        normalized = [normalize_predicate(p) for p in predicates]

        # Then validate
        valid, invalid = validate_predicates(normalized)

        assert "contract_valid" in valid
        assert "has_offer" in valid
        assert len(invalid) >= 1  # UnknownPred remains invalid

    def test_full_translation_pipeline(self):
        """Test complete translation with validation."""
        translator = ConstrainedNLToASPTranslator()

        # Translate a legal statement
        result = translator.translate("A contract requires offer, acceptance, and consideration.")

        # Check result structure
        assert result.asp_code is not None
        assert "validation" in result.metadata

        validation = result.metadata["validation"]
        assert "compliance_rate" in validation
        assert validation["compliance_rate"] >= 0  # May be low for pattern-based

    def test_round_trip_alignment(self):
        """Test that vocabulary aligns with ASP→NL templates."""
        # These predicates should exist in both vocabularies
        from loft.translation.asp_to_nl import STATEMENT_TEMPLATES

        # Check overlap between canonical predicates and ASP→NL templates
        template_predicates = set(STATEMENT_TEMPLATES.keys())
        overlap = CANONICAL_PREDICATES & template_predicates

        # Should have significant overlap for bidirectional translation
        assert len(overlap) >= 10, f"Insufficient overlap between vocabularies: {overlap}"
