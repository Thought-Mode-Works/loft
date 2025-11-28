"""
Tests for HybridTranslator class.

Tests for:
- Canonical translation (deterministic path)
- LLM fallback translation (mocked)
- Rule translation with mixed methods
- Translation validation and promotion
- Statistics tracking
"""

import pytest
from unittest.mock import MagicMock, patch

from loft.ontology.hybrid_translator import (
    HybridTranslator,
    TranslationResult,
    TranslationStats,
)
from loft.ontology.canonical_translator import CanonicalTranslator


class TestTranslationStats:
    """Test TranslationStats dataclass."""

    def test_initial_values(self):
        """Stats should initialize to zero."""
        stats = TranslationStats()
        assert stats.canonical_translations == 0
        assert stats.llm_translations == 0
        assert stats.failed_translations == 0
        assert stats.total_predicates == 0

    def test_canonical_rate(self):
        """Canonical rate should be calculated correctly."""
        stats = TranslationStats(canonical_translations=8, llm_translations=2, total_predicates=10)
        assert stats.canonical_rate == 0.8

    def test_llm_rate(self):
        """LLM rate should be calculated correctly."""
        stats = TranslationStats(canonical_translations=8, llm_translations=2, total_predicates=10)
        assert stats.llm_rate == 0.2

    def test_success_rate(self):
        """Success rate should be calculated correctly."""
        stats = TranslationStats(
            canonical_translations=8,
            llm_translations=2,
            failed_translations=0,
            total_predicates=10,
        )
        assert stats.success_rate == 1.0

    def test_rates_with_zero_total(self):
        """Rates should be 0.0 when total is zero."""
        stats = TranslationStats()
        assert stats.canonical_rate == 0.0
        assert stats.llm_rate == 0.0
        assert stats.success_rate == 0.0

    def test_to_dict(self):
        """Stats should serialize to dictionary."""
        stats = TranslationStats(
            canonical_translations=5,
            llm_translations=3,
            failed_translations=2,
            total_predicates=10,
        )
        d = stats.to_dict()
        assert d["canonical_translations"] == 5
        assert d["llm_translations"] == 3
        assert d["success_rate"] == 0.8


class TestTranslationResult:
    """Test TranslationResult dataclass."""

    def test_canonical_result(self):
        """Canonical result should have confidence 1.0."""
        result = TranslationResult(
            translated="use_continuous(X, yes)",
            confidence=1.0,
            method="canonical",
            translations={"occupation_continuous": "use_continuous"},
        )
        assert result.confidence == 1.0
        assert result.method == "canonical"

    def test_llm_result(self):
        """LLM result should have lower confidence."""
        result = TranslationResult(
            translated="some_pred(X)",
            confidence=0.85,
            method="llm",
            reasoning="Semantic match based on meaning",
        )
        assert result.confidence < 1.0
        assert result.method == "llm"

    def test_failed_result(self):
        """Failed result should have confidence 0.0."""
        result = TranslationResult(
            translated=None, confidence=0.0, method="none", reasoning="No match found"
        )
        assert result.translated is None
        assert result.confidence == 0.0


class TestHybridTranslatorInit:
    """Test HybridTranslator initialization."""

    def test_default_initialization(self):
        """Translator should initialize with defaults."""
        translator = HybridTranslator(enable_llm=False)
        assert translator.canonical is not None
        assert translator.min_llm_confidence == 0.6
        assert not translator.enable_llm

    def test_custom_canonical_translator(self):
        """Should accept custom canonical translator."""
        canonical = CanonicalTranslator()
        translator = HybridTranslator(canonical_translator=canonical, enable_llm=False)
        assert translator.canonical is canonical

    def test_custom_confidence_threshold(self):
        """Should accept custom confidence threshold."""
        translator = HybridTranslator(min_llm_confidence=0.8, enable_llm=False)
        assert translator.min_llm_confidence == 0.8


class TestCanonicalTranslation:
    """Test canonical (deterministic) translation path."""

    def test_translate_predicate_canonical(self):
        """Known predicate should translate via canonical path."""
        translator = HybridTranslator(enable_llm=False)

        result, confidence, method = translator.translate_predicate(
            "occupation_continuous", "adverse_possession", "property_law"
        )

        assert result == "use_continuous"
        assert confidence == 1.0
        assert method == "canonical"

    def test_translate_predicate_no_canonical_no_llm(self):
        """Unknown predicate without LLM should fail."""
        translator = HybridTranslator(enable_llm=False)

        result, confidence, method = translator.translate_predicate(
            "unknown_predicate", "adverse_possession", "property_law"
        )

        assert result is None
        assert confidence == 0.0
        assert method == "none"

    def test_translate_rule_all_canonical(self):
        """Rule with all canonical predicates should translate fully."""
        translator = HybridTranslator(enable_llm=False)

        result = translator.translate_rule(
            "enforceable(X) :- occupation_continuous(X, yes), occupation_hostile(X, yes).",
            "adverse_possession",
            "property_law",
        )

        assert result.translated is not None
        assert "use_continuous(X, yes)" in result.translated
        assert "use_adverse(X, yes)" in result.translated
        assert result.confidence == 1.0
        assert result.method == "canonical"


class TestLLMFallbackTranslation:
    """Test LLM fallback translation with mocked LLM."""

    @patch("loft.ontology.hybrid_translator.anthropic")
    def test_llm_translation_success(self, mock_anthropic):
        """LLM should provide translation when canonical fails."""
        # Mock the anthropic client
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"target_predicate": "similar_pred", "confidence": 0.85, "reasoning": "Match"}'
            )
        ]
        mock_client.messages.create.return_value = mock_response

        # Patch ANTHROPIC_AVAILABLE
        with patch("loft.ontology.hybrid_translator.ANTHROPIC_AVAILABLE", True):
            translator = HybridTranslator(enable_llm=True)
            translator._llm_client = mock_client

            result, confidence, method = translator.translate_predicate(
                "unknown_pred",
                "adverse_possession",
                "property_law",
                target_predicates=["similar_pred", "other_pred"],
            )

        assert result == "similar_pred"
        assert confidence == 0.85
        assert method == "llm"

    @patch("loft.ontology.hybrid_translator.anthropic")
    def test_llm_translation_no_match(self, mock_anthropic):
        """LLM returning NO_MATCH should result in None."""
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"target_predicate": "NO_MATCH", "confidence": 0.0, "reasoning": "No match"}'
            )
        ]
        mock_client.messages.create.return_value = mock_response

        with patch("loft.ontology.hybrid_translator.ANTHROPIC_AVAILABLE", True):
            translator = HybridTranslator(enable_llm=True)
            translator._llm_client = mock_client

            result, confidence, method = translator.translate_predicate(
                "completely_unknown",
                "adverse_possession",
                "property_law",
                target_predicates=["pred1", "pred2"],
            )

        assert result is None
        assert confidence == 0.0

    @patch("loft.ontology.hybrid_translator.anthropic")
    def test_llm_translation_below_threshold(self, mock_anthropic):
        """LLM with low confidence should be rejected."""
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"target_predicate": "maybe_pred", "confidence": 0.4, "reasoning": "Weak"}'
            )
        ]
        mock_client.messages.create.return_value = mock_response

        with patch("loft.ontology.hybrid_translator.ANTHROPIC_AVAILABLE", True):
            translator = HybridTranslator(enable_llm=True, min_llm_confidence=0.6)
            translator._llm_client = mock_client

            result, confidence, method = translator.translate_predicate(
                "uncertain_pred",
                "adverse_possession",
                "property_law",
                target_predicates=["maybe_pred"],
            )

        assert result is None
        assert confidence == 0.0


class TestRuleTranslation:
    """Test full rule translation."""

    def test_translate_rule_no_predicates(self):
        """Rule without predicates should pass through."""
        translator = HybridTranslator(enable_llm=False)

        result = translator.translate_rule("% comment only", "adverse_possession", "property_law")

        assert result.translated == "% comment only"
        assert result.confidence == 1.0

    def test_translate_rule_partial_failure(self):
        """Rule with untranslatable predicate should fail."""
        translator = HybridTranslator(enable_llm=False)

        result = translator.translate_rule(
            "enforceable(X) :- occupation_continuous(X, yes), unknown_pred(X).",
            "adverse_possession",
            "property_law",
        )

        assert result.translated is None
        assert result.confidence == 0.0
        assert "unknown_pred" in result.reasoning

    def test_translate_rule_tracks_translations(self):
        """Rule translation should track individual predicate mappings."""
        translator = HybridTranslator(enable_llm=False)

        result = translator.translate_rule(
            "enforceable(X) :- occupation_continuous(X, yes).",
            "adverse_possession",
            "property_law",
        )

        assert "occupation_continuous" in result.translations
        assert result.translations["occupation_continuous"] == "use_continuous"


class TestStatisticsTracking:
    """Test translation statistics tracking."""

    def test_stats_track_canonical(self):
        """Canonical translations should be tracked."""
        translator = HybridTranslator(enable_llm=False)

        translator.translate_predicate(
            "occupation_continuous", "adverse_possession", "property_law"
        )

        assert translator.stats.canonical_translations == 1
        assert translator.stats.total_predicates == 1

    def test_stats_track_failures(self):
        """Failed translations should be tracked."""
        translator = HybridTranslator(enable_llm=False)

        translator.translate_predicate("unknown_pred", "adverse_possession", "property_law")

        assert translator.stats.failed_translations == 1
        assert translator.stats.total_predicates == 1

    def test_stats_reset(self):
        """Stats should be resettable."""
        translator = HybridTranslator(enable_llm=False)

        translator.translate_predicate(
            "occupation_continuous", "adverse_possession", "property_law"
        )
        translator.reset_stats()

        assert translator.stats.total_predicates == 0


class TestValidationAndPromotion:
    """Test translation validation and promotion to canonical."""

    def test_validate_translation(self):
        """Translations should be recorded for validation."""
        translator = HybridTranslator(enable_llm=False)

        translator.validate_translation(
            source_pred="custom_pred",
            target_pred="target_pred",
            source_domain="domain_a",
            target_domain="domain_b",
            success=True,
        )

        assert len(translator._validated_translations) == 1

    def test_get_validated_for_promotion(self):
        """Sufficiently validated translations should be promotable."""
        translator = HybridTranslator(enable_llm=False)

        # Add 3 successful validations
        for _ in range(3):
            translator.validate_translation(
                source_pred="custom_pred",
                target_pred="target_pred",
                source_domain="domain_a",
                target_domain="domain_b",
                success=True,
            )

        promotable = translator.get_validated_for_promotion(min_success_count=3)
        assert len(promotable) == 1
        assert promotable[0]["source_pred"] == "custom_pred"

    def test_promotion_requires_min_validations(self):
        """Translations need minimum validations for promotion."""
        translator = HybridTranslator(enable_llm=False)

        # Add only 2 successful validations
        for _ in range(2):
            translator.validate_translation(
                source_pred="custom_pred",
                target_pred="target_pred",
                source_domain="domain_a",
                target_domain="domain_b",
                success=True,
            )

        promotable = translator.get_validated_for_promotion(min_success_count=3)
        assert len(promotable) == 0

    def test_promote_to_canonical(self):
        """Validated translation should be promotable to canonical."""
        translator = HybridTranslator(enable_llm=False)

        # Promote a new mapping
        success = translator.promote_to_canonical(
            source_pred="new_source",
            target_pred="new_target",
            source_domain="adverse_possession",
            target_domain="property_law",
            canonical_name="auto_new_concept",
        )

        assert success

        # Verify it now translates canonically
        result, confidence, method = translator.translate_predicate(
            "new_source", "adverse_possession", "property_law"
        )

        assert result == "new_target"
        assert confidence == 1.0
        assert method == "canonical"


class TestCaching:
    """Test LLM translation caching."""

    @patch("loft.ontology.hybrid_translator.anthropic")
    def test_cache_prevents_duplicate_llm_calls(self, mock_anthropic):
        """Same predicate should only call LLM once."""
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"target_predicate": "cached_pred", "confidence": 0.9, "reasoning": "Cache test"}'
            )
        ]
        mock_client.messages.create.return_value = mock_response

        with patch("loft.ontology.hybrid_translator.ANTHROPIC_AVAILABLE", True):
            translator = HybridTranslator(enable_llm=True)
            translator._llm_client = mock_client

            # First call
            translator.translate_predicate(
                "cache_test",
                "adverse_possession",
                "property_law",
                target_predicates=["cached_pred"],
            )

            # Second call (should use cache)
            translator.translate_predicate(
                "cache_test",
                "adverse_possession",
                "property_law",
                target_predicates=["cached_pred"],
            )

        # LLM should only be called once
        assert mock_client.messages.create.call_count == 1

    def test_clear_cache(self):
        """Cache should be clearable."""
        translator = HybridTranslator(enable_llm=False)
        translator._llm_cache[("pred", "dom1", "dom2")] = ("result", 0.9, "llm")

        translator.clear_cache()

        assert len(translator._llm_cache) == 0


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Repr should show useful configuration info."""
        translator = HybridTranslator(model="test-model", min_llm_confidence=0.7, enable_llm=False)

        repr_str = repr(translator)

        assert "HybridTranslator" in repr_str
        assert "test-model" in repr_str
        assert "0.7" in repr_str
