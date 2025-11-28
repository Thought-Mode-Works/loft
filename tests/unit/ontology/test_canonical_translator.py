"""
Tests for CanonicalTranslator class.

Tests for:
- Ontology loading and parsing
- Domain to canonical translation
- Canonical to domain translation
- Cross-domain rule translation
- Translation coverage metrics
"""

import pytest
from pathlib import Path

from loft.ontology.canonical_translator import CanonicalTranslator


class TestOntologyLoading:
    """Test ontology loading and parsing."""

    def test_load_default_ontology(self):
        """Default ontology should load successfully."""
        translator = CanonicalTranslator()
        assert translator.graph is not None
        assert len(translator.get_domains()) > 0

    def test_load_ontology_with_path(self):
        """Loading with explicit path should work."""
        path = Path(__file__).parent.parent.parent.parent / "loft" / "ontology" / "canonical_predicates.ttl"
        translator = CanonicalTranslator(ontology_path=path)
        assert translator.graph is not None

    def test_ontology_not_found_raises_error(self):
        """Non-existent ontology path should raise error."""
        with pytest.raises(FileNotFoundError):
            CanonicalTranslator(ontology_path=Path("/nonexistent/path.ttl"))

    def test_domains_loaded(self):
        """All expected domains should be loaded."""
        translator = CanonicalTranslator()
        domains = translator.get_domains()

        assert "adverse_possession" in domains
        assert "property_law" in domains
        assert "statute_of_frauds" in domains

    def test_canonical_predicates_loaded(self):
        """Canonical predicates should be loaded with metadata."""
        translator = CanonicalTranslator()
        canonical = translator.get_canonical_predicates()

        assert "continuous_possession" in canonical
        assert "duration_period" in canonical
        assert "hostile_action" in canonical
        assert "claim_enforceable" in canonical


class TestDomainToCanonical:
    """Test domain predicate to canonical translation."""

    def test_translate_adverse_possession_predicates(self):
        """Adverse possession predicates should map to canonical."""
        translator = CanonicalTranslator()

        assert translator.to_canonical("occupation_continuous", "adverse_possession") == "continuous_possession"
        assert translator.to_canonical("occupation_years", "adverse_possession") == "duration_period"
        assert translator.to_canonical("occupation_hostile", "adverse_possession") == "hostile_action"
        assert translator.to_canonical("enforceable", "adverse_possession") == "claim_enforceable"

    def test_translate_property_law_predicates(self):
        """Property law predicates should map to canonical."""
        translator = CanonicalTranslator()

        assert translator.to_canonical("use_continuous", "property_law") == "continuous_possession"
        assert translator.to_canonical("use_duration_years", "property_law") == "duration_period"
        assert translator.to_canonical("use_adverse", "property_law") == "hostile_action"

    def test_unknown_predicate_returns_none(self):
        """Unknown predicate should return None."""
        translator = CanonicalTranslator()

        assert translator.to_canonical("unknown_predicate", "adverse_possession") is None

    def test_unknown_domain_returns_none(self):
        """Unknown domain should return None."""
        translator = CanonicalTranslator()

        assert translator.to_canonical("claim", "unknown_domain") is None


class TestCanonicalToDomain:
    """Test canonical to domain predicate translation."""

    def test_translate_to_adverse_possession(self):
        """Canonical predicates should map to adverse possession."""
        translator = CanonicalTranslator()

        assert translator.from_canonical("continuous_possession", "adverse_possession") == "occupation_continuous"
        assert translator.from_canonical("duration_period", "adverse_possession") == "occupation_years"
        assert translator.from_canonical("hostile_action", "adverse_possession") == "occupation_hostile"

    def test_translate_to_property_law(self):
        """Canonical predicates should map to property law."""
        translator = CanonicalTranslator()

        assert translator.from_canonical("continuous_possession", "property_law") == "use_continuous"
        assert translator.from_canonical("duration_period", "property_law") == "use_duration_years"
        assert translator.from_canonical("hostile_action", "property_law") == "use_adverse"

    def test_unmapped_canonical_returns_none(self):
        """Canonical predicate without domain mapping should return None."""
        translator = CanonicalTranslator()

        # taxes_payment is only in adverse_possession, not property_law
        assert translator.from_canonical("taxes_payment", "property_law") is None


class TestCrossDomainTranslation:
    """Test cross-domain predicate translation."""

    def test_translate_predicate_between_domains(self):
        """Predicates should translate between domains."""
        translator = CanonicalTranslator()

        # Adverse possession -> Property law
        result = translator.translate_predicate(
            "occupation_continuous", "adverse_possession", "property_law"
        )
        assert result == "use_continuous"

        # Property law -> Adverse possession
        result = translator.translate_predicate(
            "use_adverse", "property_law", "adverse_possession"
        )
        assert result == "occupation_hostile"

    def test_translate_predicate_no_target_mapping(self):
        """Predicate without target domain mapping should return None."""
        translator = CanonicalTranslator()

        # taxes_paid exists in adverse_possession but not property_law
        result = translator.translate_predicate(
            "taxes_paid", "adverse_possession", "property_law"
        )
        assert result is None


class TestRuleTranslation:
    """Test full ASP rule translation."""

    def test_translate_simple_rule(self):
        """Simple rule should translate correctly."""
        translator = CanonicalTranslator()

        rule = "enforceable(X) :- occupation_continuous(X, yes)."
        translated, success, failed = translator.translate_rule(
            rule, "adverse_possession", "property_law"
        )

        assert "use_continuous(X, yes)" in translated
        assert "enforceable(X)" in translated  # Same in both domains
        assert len(success) >= 1

    def test_translate_complex_rule(self):
        """Complex rule with multiple predicates should translate."""
        translator = CanonicalTranslator()

        rule = """enforceable(X) :-
            claim(X),
            occupation_continuous(X, yes),
            occupation_hostile(X, yes),
            occupation_years(X, N),
            statutory_period(X, P),
            N >= P."""

        translated, success, failed = translator.translate_rule(
            rule, "adverse_possession", "property_law"
        )

        assert "use_continuous(X, yes)" in translated
        assert "use_adverse(X, yes)" in translated
        assert "use_duration_years(X, N)" in translated

    def test_translate_rule_with_untranslatable(self):
        """Rule with untranslatable predicates should keep them as-is."""
        translator = CanonicalTranslator()

        rule = "enforceable(X) :- occupation_continuous(X, yes), taxes_paid(X, yes)."
        translated, success, failed = translator.translate_rule(
            rule, "adverse_possession", "property_law"
        )

        # taxes_paid has no property_law equivalent
        assert "taxes_paid(X, yes)" in translated
        assert "taxes_paid" in failed

    def test_translate_rule_strict_mode(self):
        """Strict mode should raise error on untranslatable predicates."""
        translator = CanonicalTranslator()

        rule = "enforceable(X) :- taxes_paid(X, yes)."

        with pytest.raises(ValueError) as exc_info:
            translator.translate_rule(
                rule, "adverse_possession", "property_law", strict=True
            )

        assert "taxes_paid" in str(exc_info.value)

    def test_translate_preserves_comparison_operators(self):
        """Comparison operators should be preserved."""
        translator = CanonicalTranslator()

        rule = "enforceable(X) :- occupation_years(X, N), statutory_period(X, P), N >= P."
        translated, _, _ = translator.translate_rule(
            rule, "adverse_possession", "property_law"
        )

        assert "N >= P" in translated


class TestCanonicalMetadata:
    """Test canonical predicate metadata retrieval."""

    def test_get_metadata(self):
        """Metadata should be retrievable for canonical predicates."""
        translator = CanonicalTranslator()

        meta = translator.get_canonical_metadata("continuous_possession")
        assert meta is not None
        assert meta["arity"] == 2
        assert "temporal" in meta["legal_category"].lower()

    def test_get_metadata_unknown_predicate(self):
        """Unknown predicate should return None."""
        translator = CanonicalTranslator()

        assert translator.get_canonical_metadata("unknown_pred") is None


class TestTranslationCoverage:
    """Test translation coverage metrics."""

    def test_coverage_between_similar_domains(self):
        """Similar domains should have high coverage."""
        translator = CanonicalTranslator()

        coverage = translator.get_translation_coverage(
            "adverse_possession", "property_law"
        )

        assert coverage["source_predicates"] > 0
        assert coverage["common_canonical"] > 0
        assert coverage["coverage_ratio"] > 0.5  # Should share many predicates

    def test_find_common_canonical(self):
        """Common canonical predicates should be found."""
        translator = CanonicalTranslator()

        common = translator.find_common_canonical(
            "adverse_possession", "property_law"
        )

        assert "continuous_possession" in common
        assert "hostile_action" in common
        assert "claim_enforceable" in common


class TestDomainPredicates:
    """Test domain predicate retrieval."""

    def test_get_domain_predicates(self):
        """All domain predicates should be retrievable."""
        translator = CanonicalTranslator()

        ap_preds = translator.get_domain_predicates("adverse_possession")
        assert "claim" in ap_preds
        assert "occupation_continuous" in ap_preds
        assert "taxes_paid" in ap_preds

    def test_get_domain_predicates_unknown_domain(self):
        """Unknown domain should return empty list."""
        translator = CanonicalTranslator()

        preds = translator.get_domain_predicates("unknown_domain")
        assert preds == []


class TestSerialization:
    """Test serialization methods."""

    def test_to_dict(self):
        """Mappings should be exportable as dict."""
        translator = CanonicalTranslator()

        data = translator.to_dict()

        assert "domain_to_canonical" in data
        assert "canonical_to_domain" in data
        assert "canonical_metadata" in data
        assert "adverse_possession" in data["domain_to_canonical"]

    def test_repr(self):
        """Repr should show useful info."""
        translator = CanonicalTranslator()

        repr_str = repr(translator)

        assert "CanonicalTranslator" in repr_str
        assert "domains=" in repr_str
        assert "canonical_predicates=" in repr_str
