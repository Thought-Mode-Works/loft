"""
Unit tests for ASP to Natural Language translation.
"""

from loft.translation import (
    asp_to_nl,
    asp_rule_to_nl,
    asp_facts_to_nl,
    ASPToNLTranslator,
    extract_predicates,
    validate_translation_quality,
    compute_fidelity,
    compute_quality_metrics,
)


class TestExtractPredicates:
    """Test predicate extraction."""

    def test_single_predicate(self) -> None:
        """Test extracting single predicate."""
        predicates = extract_predicates("contract(c1).")
        assert predicates == ["contract"]

    def test_multiple_predicates(self) -> None:
        """Test extracting multiple predicates."""
        predicates = extract_predicates("contract(c1), enforceable(c1).")
        assert predicates == ["contract", "enforceable"]

    def test_rule_predicates(self) -> None:
        """Test extracting predicates from rule."""
        predicates = extract_predicates("enforceable(C) :- contract(C), not void(C).")
        assert predicates == ["enforceable", "contract", "void"]

    def test_no_duplicates(self) -> None:
        """Test that duplicates are removed."""
        predicates = extract_predicates("contract(c1), contract(c2).")
        assert predicates == ["contract"]

    def test_filters_keywords(self) -> None:
        """Test that ASP keywords are filtered."""
        predicates = extract_predicates("contract(C) :- not void(C).")
        # 'not' should be filtered out
        assert "not" not in predicates


class TestAspToNl:
    """Test ASP query to natural language translation."""

    def test_simple_constant_query(self) -> None:
        """Test translating query with constant."""
        nl = asp_to_nl("contract(c1)?")
        assert "c1 is a contract" in nl.lower()
        assert nl.endswith("?")

    def test_variable_query(self) -> None:
        """Test translating query with variable."""
        nl = asp_to_nl("contract(C)?")
        assert "which" in nl.lower()
        assert nl.endswith("?")

    def test_statute_of_frauds_query(self) -> None:
        """Test translating statute of frauds query."""
        nl = asp_to_nl("satisfies_statute_of_frauds(contract_123)?")
        assert "contract_123" in nl
        assert "statute of frauds" in nl.lower()
        assert nl.endswith("?")

    def test_enforceable_variable_query(self) -> None:
        """Test enforceable with variable."""
        nl = asp_to_nl("enforceable(C)?")
        assert "which" in nl.lower() or "enforceable" in nl.lower()

    def test_binary_predicate_query(self) -> None:
        """Test query with binary predicate."""
        nl = asp_to_nl("signed_by(w1, john)?")
        assert "w1" in nl
        assert "john" in nl
        assert "signed" in nl.lower()

    def test_removes_question_mark(self) -> None:
        """Test that input question marks are handled."""
        nl1 = asp_to_nl("contract(c1)?")
        nl2 = asp_to_nl("contract(c1)")
        # Should produce similar results
        assert "contract" in nl1.lower() and "contract" in nl2.lower()


class TestAspRuleToNl:
    """Test ASP rule to natural language translation."""

    def test_simple_fact(self) -> None:
        """Test translating a fact."""
        nl = asp_rule_to_nl("contract(c1).")
        assert "c1" in nl
        assert "contract" in nl.lower()
        assert nl.endswith(".")

    def test_simple_rule(self) -> None:
        """Test translating simple rule."""
        nl = asp_rule_to_nl("enforceable(C) :- contract(C).")
        assert "enforceable" in nl.lower()
        assert "contract" in nl.lower()
        assert "if" in nl.lower()

    def test_rule_with_negation(self) -> None:
        """Test translating rule with negation."""
        nl = asp_rule_to_nl("enforceable(C) :- contract(C), not unenforceable(C).")
        assert "enforceable" in nl.lower()
        assert "not" in nl.lower()
        assert "unenforceable" in nl.lower()

    def test_rule_with_conjunction(self) -> None:
        """Test translating rule with multiple conditions."""
        nl = asp_rule_to_nl("satisfies_statute_of_frauds(C) :- has_writing(C, W), signed_by(W, _).")
        assert "statute" in nl.lower() and "frauds" in nl.lower()
        assert "writing" in nl.lower()
        assert "signed" in nl.lower()
        assert "and" in nl.lower() or "," in nl

    def test_constraint(self) -> None:
        """Test translating constraint."""
        nl = asp_rule_to_nl(":- void(C), enforceable(C).")
        assert "constraint" in nl.lower() or "not allowed" in nl.lower()


class TestAspFactsToNl:
    """Test ASP facts to natural language translation."""

    def test_single_fact(self) -> None:
        """Test translating single fact."""
        nl = asp_facts_to_nl(["contract(c1)."])
        assert "c1" in nl
        assert "contract" in nl.lower()

    def test_multiple_facts(self) -> None:
        """Test translating multiple facts."""
        facts = [
            "contract(c1).",
            "land_sale_contract(c1).",
            "signed_by(w1, john).",
        ]
        nl = asp_facts_to_nl(facts)
        assert "c1" in nl
        assert "contract" in nl.lower()
        assert "land" in nl.lower() or "sale" in nl.lower()
        assert "john" in nl

    def test_empty_facts(self) -> None:
        """Test with empty fact list."""
        nl = asp_facts_to_nl([])
        assert nl == ""

    def test_fact_narrative_flow(self) -> None:
        """Test that facts form coherent narrative."""
        facts = ["contract(c1).", "enforceable(c1)."]
        nl = asp_facts_to_nl(facts)
        # Should have multiple sentences
        assert nl.count(".") >= 2


class TestASPToNLTranslator:
    """Test the ASPToNLTranslator class."""

    def test_initialization(self) -> None:
        """Test translator initialization."""
        translator = ASPToNLTranslator(domain="legal")
        assert translator.domain == "legal"
        assert len(translator.templates) > 0

    def test_translate_query(self) -> None:
        """Test translate_query method."""
        translator = ASPToNLTranslator()
        result = translator.translate_query("contract(c1)?")

        assert result.natural_language
        assert result.asp_source == "contract(c1)?"
        assert "contract" in result.predicates_used
        assert 0.0 <= result.confidence <= 1.0

    def test_translate_rule(self) -> None:
        """Test translate_rule method."""
        translator = ASPToNLTranslator()
        result = translator.translate_rule("enforceable(C) :- contract(C).")

        assert result.natural_language
        assert "enforceable" in result.predicates_used
        assert "contract" in result.predicates_used
        assert 0.0 <= result.confidence <= 1.0

    def test_translate_facts(self) -> None:
        """Test translate_facts method."""
        translator = ASPToNLTranslator()
        facts = ["contract(c1).", "void(c1)."]
        result = translator.translate_facts(facts)

        assert result.natural_language
        assert "contract" in result.predicates_used
        assert "void" in result.predicates_used
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_score_known_predicates(self) -> None:
        """Test confidence is high for known predicates."""
        translator = ASPToNLTranslator(domain="legal")
        result = translator.translate_query("contract(c1)?")
        # Contract is a known legal predicate
        assert result.confidence >= 0.7

    def test_confidence_score_unknown_predicates(self) -> None:
        """Test confidence is lower for unknown predicates."""
        translator = ASPToNLTranslator(domain="legal")
        result = translator.translate_query("unknown_predicate(x)?")
        # Unknown predicate should have lower confidence
        assert result.confidence <= 1.0


class TestValidateTranslationQuality:
    """Test translation quality validation."""

    def test_perfect_translation(self) -> None:
        """Test quality score for perfect translation."""
        asp = "contract(c1)."
        nl = "c1 is a contract."
        score = validate_translation_quality(asp, nl)
        assert score > 0.8  # Should be high quality

    def test_missing_predicates(self) -> None:
        """Test quality score when predicates are missing."""
        asp = "contract(c1), enforceable(c1)."
        nl = "c1 is a contract."  # Missing enforceable
        score = validate_translation_quality(asp, nl)
        assert score < 1.0  # Should not be perfect

    def test_incomplete_sentence(self) -> None:
        """Test quality score for incomplete sentences."""
        asp = "contract(c1)."
        nl = "c1 is a contract"  # Missing period
        score = validate_translation_quality(asp, nl)
        # Should still be decent but penalized
        assert 0.3 < score < 1.0

    def test_empty_translation(self) -> None:
        """Test quality score for empty translation."""
        asp = "contract(c1)."
        nl = ""
        score = validate_translation_quality(asp, nl)
        assert score < 0.5  # Should be low


class TestComputeFidelity:
    """Test semantic fidelity computation."""

    def test_identical_asp(self) -> None:
        """Test fidelity of identical ASP."""
        asp = "contract(c1)."
        fidelity = compute_fidelity(asp, asp)
        assert fidelity == 1.0

    def test_similar_predicates(self) -> None:
        """Test fidelity with same predicates."""
        asp1 = "contract(c1), enforceable(c1)."
        asp2 = "enforceable(c1), contract(c1)."  # Different order
        fidelity = compute_fidelity(asp1, asp2)
        # Should be high since predicates are the same
        assert fidelity > 0.8

    def test_different_predicates(self) -> None:
        """Test fidelity with different predicates."""
        asp1 = "contract(c1)."
        asp2 = "void(c1)."
        fidelity = compute_fidelity(asp1, asp2)
        # Should be low since predicates differ
        assert fidelity < 0.5

    def test_whitespace_normalization(self) -> None:
        """Test that whitespace doesn't affect fidelity."""
        asp1 = "contract(c1)."
        asp2 = "contract( c1 ) ."  # Extra whitespace
        fidelity = compute_fidelity(asp1, asp2)
        assert fidelity == 1.0


class TestComputeQualityMetrics:
    """Test comprehensive quality metrics."""

    def test_quality_metrics_structure(self) -> None:
        """Test that quality metrics have all fields."""
        asp = "contract(c1)."
        nl = "c1 is a contract."
        metrics = compute_quality_metrics(asp, nl)

        assert hasattr(metrics, "completeness")
        assert hasattr(metrics, "readability")
        assert hasattr(metrics, "fidelity")
        assert hasattr(metrics, "overall")

        # All scores should be between 0 and 1
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.readability <= 1.0
        assert 0.0 <= metrics.fidelity <= 1.0
        assert 0.0 <= metrics.overall <= 1.0

    def test_high_quality_translation_metrics(self) -> None:
        """Test metrics for high quality translation."""
        asp = "contract(c1)."
        nl = "c1 is a contract."
        metrics = compute_quality_metrics(asp, nl)

        # Should all be relatively high
        assert metrics.completeness > 0.8
        assert metrics.readability > 0.5
        assert metrics.overall > 0.6

    def test_low_quality_translation_metrics(self) -> None:
        """Test metrics for low quality translation."""
        asp = "contract(c1), enforceable(c1), valid(c1)."
        nl = "Something."  # Missing all predicates
        metrics = compute_quality_metrics(asp, nl)

        # Should be low
        assert metrics.completeness < 0.5
        assert metrics.overall < 0.7
