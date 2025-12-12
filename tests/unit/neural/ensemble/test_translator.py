"""
Tests for TranslatorLLM - Symbolic to natural language bidirectional conversion.

Part of Phase 6: Heterogeneous Neural Ensemble (Issue #190).
"""

import pytest
from unittest.mock import MagicMock, patch

from loft.neural.ensemble.translator import (
    # Main class
    TranslatorLLM,
    TranslatorConfig,
    # Abstract base class
    TranslationResult,
    RoundtripResult,
    TranslatorBenchmarkResult,
    # Strategy Pattern components
    TranslationStrategyType,
    LiteralStrategy,
    ContextualStrategy,
    LegalDomainStrategy,
    PedagogicalStrategy,
    create_translation_strategy,
    # Pydantic schemas
    ASPToNLSchema,
    NLToASPSchema,
    FidelityAssessmentSchema,
    # Exceptions
    TranslationError,
    LLMResponseParsingError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.model = "claude-3-5-haiku-20241022"
    return provider


@pytest.fixture
def default_config():
    """Create default translator config."""
    return TranslatorConfig(
        model="claude-3-5-haiku-20241022",
        temperature=0.2,
        strategy=TranslationStrategyType.LEGAL_DOMAIN,
        enable_cache=True,
    )


@pytest.fixture
def translator(mock_provider, default_config):
    """Create TranslatorLLM instance with mocked provider."""
    with patch("loft.neural.ensemble.translator.LLMInterface"):
        return TranslatorLLM(mock_provider, default_config)


@pytest.fixture
def sample_asp_rule():
    """Sample ASP rule for testing."""
    return "enforceable(C) :- contract(C), signed(C), has_consideration(C)."


@pytest.fixture
def sample_nl_description():
    """Sample natural language description for testing."""
    return "A contract is enforceable if it is signed and has consideration."


@pytest.fixture
def sample_predicates():
    """Sample predicates for testing."""
    return ["contract(X)", "signed(X)", "has_consideration(X)", "enforceable(X)"]


# =============================================================================
# Test Data Classes
# =============================================================================


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_translation_result_creation(self):
        """Test creating a TranslationResult."""
        result = TranslationResult(
            source="enforceable(C) :- contract(C).",
            target="A contract is enforceable if it is a contract.",
            direction="asp_to_nl",
            confidence=0.85,
        )
        assert result.source == "enforceable(C) :- contract(C)."
        assert result.target == "A contract is enforceable if it is a contract."
        assert result.direction == "asp_to_nl"
        assert result.confidence == 0.85
        assert result.predicates_used == []
        assert result.ambiguities == []
        assert result.from_cache is False

    def test_translation_result_with_all_fields(self):
        """Test TranslationResult with all fields populated."""
        result = TranslationResult(
            source="test source",
            target="test target",
            direction="nl_to_asp",
            confidence=0.9,
            predicates_used=["pred1", "pred2"],
            ambiguities=["ambiguity1"],
            reasoning="Reasoning explanation",
            translation_time_ms=150.5,
            from_cache=True,
        )
        assert result.predicates_used == ["pred1", "pred2"]
        assert result.ambiguities == ["ambiguity1"]
        assert result.reasoning == "Reasoning explanation"
        assert result.translation_time_ms == 150.5
        assert result.from_cache is True


class TestRoundtripResult:
    """Tests for RoundtripResult dataclass."""

    def test_roundtrip_result_creation(self):
        """Test creating a RoundtripResult."""
        result = RoundtripResult(
            original="enforceable(C) :- contract(C).",
            intermediate="A contract is enforceable.",
            final="enforceable(X) :- contract(X).",
            fidelity_score=0.92,
            is_asp_original=True,
        )
        assert result.original == "enforceable(C) :- contract(C)."
        assert result.intermediate == "A contract is enforceable."
        assert result.final == "enforceable(X) :- contract(X)."
        assert result.fidelity_score == 0.92
        assert result.is_asp_original is True

    def test_roundtrip_result_with_details(self):
        """Test RoundtripResult with all detail fields."""
        result = RoundtripResult(
            original="test",
            intermediate="intermediate",
            final="final",
            fidelity_score=0.85,
            is_asp_original=False,
            preserved_predicates=["pred1", "pred2"],
            lost_information=["lost1"],
            added_information=["added1"],
            confidence=0.8,
            total_time_ms=500.0,
        )
        assert result.preserved_predicates == ["pred1", "pred2"]
        assert result.lost_information == ["lost1"]
        assert result.added_information == ["added1"]
        assert result.confidence == 0.8
        assert result.total_time_ms == 500.0


class TestTranslatorBenchmarkResult:
    """Tests for TranslatorBenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a TranslatorBenchmarkResult."""
        result = TranslatorBenchmarkResult(
            translator_fidelity=0.95,
            general_llm_fidelity=0.75,
            translator_avg_time_ms=200.0,
            general_llm_avg_time_ms=150.0,
            test_cases_count=50,
            improvement_percentage=26.7,
            translator_roundtrip_success_rate=0.90,
            general_llm_roundtrip_success_rate=0.60,
        )
        assert result.translator_fidelity == 0.95
        assert result.general_llm_fidelity == 0.75
        assert result.improvement_percentage == 26.7


# =============================================================================
# Test Strategies
# =============================================================================


class TestTranslationStrategyType:
    """Tests for TranslationStrategyType enum."""

    def test_strategy_types_exist(self):
        """Test all strategy types are defined."""
        assert TranslationStrategyType.LITERAL.value == "literal"
        assert TranslationStrategyType.CONTEXTUAL.value == "contextual"
        assert TranslationStrategyType.LEGAL_DOMAIN.value == "legal_domain"
        assert TranslationStrategyType.PEDAGOGICAL.value == "pedagogical"


class TestLiteralStrategy:
    """Tests for LiteralStrategy."""

    def test_literal_strategy_type(self):
        """Test LiteralStrategy returns correct type."""
        strategy = LiteralStrategy()
        assert strategy.strategy_type == TranslationStrategyType.LITERAL

    def test_literal_asp_to_nl_prompt(self):
        """Test LiteralStrategy adds literal instructions to prompt."""
        strategy = LiteralStrategy()
        base_prompt = "Translate this ASP rule."
        result = strategy.prepare_asp_to_nl_prompt(base_prompt, "rule", {})
        assert "Literal Translation Instructions" in result
        assert "Preserve the logical relationship" in result

    def test_literal_nl_to_asp_prompt(self):
        """Test LiteralStrategy prepares NL to ASP prompt."""
        strategy = LiteralStrategy()
        base_prompt = "Translate this description."
        result = strategy.prepare_nl_to_asp_prompt(base_prompt, "desc", [], {})
        assert "Literal Translation Instructions" in result


class TestContextualStrategy:
    """Tests for ContextualStrategy."""

    def test_contextual_strategy_type(self):
        """Test ContextualStrategy returns correct type."""
        strategy = ContextualStrategy()
        assert strategy.strategy_type == TranslationStrategyType.CONTEXTUAL

    def test_contextual_asp_to_nl_prompt(self):
        """Test ContextualStrategy adds contextual instructions."""
        strategy = ContextualStrategy()
        base_prompt = "Translate this ASP rule."
        result = strategy.prepare_asp_to_nl_prompt(base_prompt, "rule", {})
        assert "Contextual Translation Instructions" in result
        assert "domain-appropriate terminology" in result


class TestLegalDomainStrategy:
    """Tests for LegalDomainStrategy."""

    def test_legal_domain_strategy_type(self):
        """Test LegalDomainStrategy returns correct type."""
        strategy = LegalDomainStrategy()
        assert strategy.strategy_type == TranslationStrategyType.LEGAL_DOMAIN

    def test_legal_domain_asp_to_nl_prompt(self):
        """Test LegalDomainStrategy adds legal instructions."""
        strategy = LegalDomainStrategy()
        base_prompt = "Translate this ASP rule."
        result = strategy.prepare_asp_to_nl_prompt(base_prompt, "rule", {})
        assert "Legal Domain Translation Instructions" in result
        assert "legal terminology" in result

    def test_legal_domain_nl_to_asp_prompt(self):
        """Test LegalDomainStrategy prepares NL to ASP prompt."""
        strategy = LegalDomainStrategy()
        base_prompt = "Translate this description."
        result = strategy.prepare_nl_to_asp_prompt(base_prompt, "desc", [], {})
        assert "Legal Domain Translation Instructions" in result
        assert "legal elements" in result


class TestPedagogicalStrategy:
    """Tests for PedagogicalStrategy."""

    def test_pedagogical_strategy_type(self):
        """Test PedagogicalStrategy returns correct type."""
        strategy = PedagogicalStrategy()
        assert strategy.strategy_type == TranslationStrategyType.PEDAGOGICAL

    def test_pedagogical_asp_to_nl_prompt(self):
        """Test PedagogicalStrategy adds educational instructions."""
        strategy = PedagogicalStrategy()
        base_prompt = "Translate this ASP rule."
        result = strategy.prepare_asp_to_nl_prompt(base_prompt, "rule", {})
        assert "Pedagogical Translation Instructions" in result
        assert "educational clarity" in result


class TestCreateTranslationStrategy:
    """Tests for strategy factory function."""

    def test_create_literal_strategy(self):
        """Test creating LiteralStrategy via factory."""
        strategy = create_translation_strategy(TranslationStrategyType.LITERAL)
        assert isinstance(strategy, LiteralStrategy)

    def test_create_contextual_strategy(self):
        """Test creating ContextualStrategy via factory."""
        strategy = create_translation_strategy(TranslationStrategyType.CONTEXTUAL)
        assert isinstance(strategy, ContextualStrategy)

    def test_create_legal_domain_strategy(self):
        """Test creating LegalDomainStrategy via factory."""
        strategy = create_translation_strategy(TranslationStrategyType.LEGAL_DOMAIN)
        assert isinstance(strategy, LegalDomainStrategy)

    def test_create_pedagogical_strategy(self):
        """Test creating PedagogicalStrategy via factory."""
        strategy = create_translation_strategy(TranslationStrategyType.PEDAGOGICAL)
        assert isinstance(strategy, PedagogicalStrategy)


# =============================================================================
# Test Configuration
# =============================================================================


class TestTranslatorConfig:
    """Tests for TranslatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TranslatorConfig()
        assert config.model == "claude-3-5-haiku-20241022"
        assert config.temperature == 0.2
        assert config.strategy == TranslationStrategyType.LEGAL_DOMAIN
        assert config.enable_cache is True
        assert config.fidelity_threshold == 0.95

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TranslatorConfig(
            model="custom-model",
            temperature=0.5,
            strategy=TranslationStrategyType.LITERAL,
            fidelity_threshold=0.90,
            enable_cache=False,
        )
        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.strategy == TranslationStrategyType.LITERAL
        assert config.fidelity_threshold == 0.90
        assert config.enable_cache is False


# =============================================================================
# Test TranslatorLLM Initialization
# =============================================================================


class TestTranslatorLLMInit:
    """Tests for TranslatorLLM initialization."""

    def test_init_with_provider(self, mock_provider):
        """Test initialization with provider."""
        with patch("loft.neural.ensemble.translator.LLMInterface"):
            translator = TranslatorLLM(mock_provider)
            assert translator.provider == mock_provider
            assert translator.config.model == "claude-3-5-haiku-20241022"

    def test_init_with_config(self, mock_provider, default_config):
        """Test initialization with custom config."""
        with patch("loft.neural.ensemble.translator.LLMInterface"):
            translator = TranslatorLLM(mock_provider, default_config)
            assert translator.config == default_config

    def test_init_with_custom_strategy(self, mock_provider):
        """Test initialization with custom strategy."""
        custom_strategy = LiteralStrategy()
        with patch("loft.neural.ensemble.translator.LLMInterface"):
            translator = TranslatorLLM(mock_provider, strategy=custom_strategy)
            assert translator._strategy == custom_strategy

    def test_init_with_none_provider_raises(self):
        """Test that None provider raises ValueError."""
        with pytest.raises(ValueError, match="provider cannot be None"):
            TranslatorLLM(None)

    def test_init_default_strategy_from_config(self, mock_provider):
        """Test default strategy is created from config."""
        config = TranslatorConfig(strategy=TranslationStrategyType.PEDAGOGICAL)
        with patch("loft.neural.ensemble.translator.LLMInterface"):
            translator = TranslatorLLM(mock_provider, config)
            assert (
                translator._strategy.strategy_type
                == TranslationStrategyType.PEDAGOGICAL
            )


class TestTranslatorLLMSetStrategy:
    """Tests for TranslatorLLM.set_strategy method."""

    def test_set_strategy(self, translator):
        """Test setting a new strategy."""
        new_strategy = LiteralStrategy()
        translator.set_strategy(new_strategy)
        assert translator._strategy == new_strategy

    def test_set_strategy_none_raises(self, translator):
        """Test that setting None strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy cannot be None"):
            translator.set_strategy(None)


# =============================================================================
# Test ASP to NL Translation
# =============================================================================


class TestASPToNL:
    """Tests for TranslatorLLM.asp_to_nl method."""

    def test_asp_to_nl_empty_raises(self, translator):
        """Test that empty ASP rule raises ValueError."""
        with pytest.raises(ValueError, match="asp_rule cannot be empty"):
            translator.asp_to_nl("")

    def test_asp_to_nl_whitespace_raises(self, translator):
        """Test that whitespace-only ASP rule raises ValueError."""
        with pytest.raises(ValueError, match="asp_rule cannot be empty"):
            translator.asp_to_nl("   ")

    def test_asp_to_nl_successful(self, translator, sample_asp_rule):
        """Test successful ASP to NL translation."""
        mock_response = MagicMock()
        mock_response.content = ASPToNLSchema(
            natural_language="A contract is enforceable if it is signed.",
            predicates_identified=["enforceable", "contract", "signed"],
            reasoning="Direct translation of predicates.",
            confidence=0.85,
            ambiguities=[],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        result = translator.asp_to_nl(sample_asp_rule)

        assert isinstance(result, TranslationResult)
        assert result.direction == "asp_to_nl"
        assert result.source == sample_asp_rule
        assert result.target == "A contract is enforceable if it is signed."
        assert result.confidence == 0.85

    def test_asp_to_nl_with_context(self, translator, sample_asp_rule):
        """Test ASP to NL translation with context."""
        mock_response = MagicMock()
        mock_response.content = ASPToNLSchema(
            natural_language="In contract law, a contract is enforceable...",
            predicates_identified=["enforceable", "contract"],
            reasoning="Legal domain translation.",
            confidence=0.9,
            ambiguities=[],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        result = translator.asp_to_nl(
            sample_asp_rule,
            context={"domain": "contracts", "predicates": ["contract(X)"]},
        )

        assert result.confidence == 0.9

    def test_asp_to_nl_caching(self, translator, sample_asp_rule):
        """Test that results are cached."""
        mock_response = MagicMock()
        mock_response.content = ASPToNLSchema(
            natural_language="Translation result",
            predicates_identified=[],
            reasoning="Test",
            confidence=0.8,
            ambiguities=[],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        # First call
        result1 = translator.asp_to_nl(sample_asp_rule)
        assert result1.from_cache is False

        # Second call should hit cache
        result2 = translator.asp_to_nl(sample_asp_rule)
        assert result2.from_cache is True
        assert translator._cache_hits == 1


# =============================================================================
# Test NL to ASP Translation
# =============================================================================


class TestNLToASP:
    """Tests for TranslatorLLM.nl_to_asp method."""

    def test_nl_to_asp_empty_raises(self, translator, sample_predicates):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            translator.nl_to_asp("", sample_predicates)

    def test_nl_to_asp_whitespace_raises(self, translator, sample_predicates):
        """Test that whitespace-only description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            translator.nl_to_asp("   ", sample_predicates)

    def test_nl_to_asp_successful(
        self, translator, sample_nl_description, sample_predicates
    ):
        """Test successful NL to ASP translation."""
        mock_response = MagicMock()
        mock_response.content = NLToASPSchema(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            predicates_used=["enforceable", "contract", "signed"],
            reasoning="Mapped conditions to predicates.",
            confidence=0.88,
            assumptions=["Assuming C represents any contract"],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        result = translator.nl_to_asp(sample_nl_description, sample_predicates)

        assert isinstance(result, TranslationResult)
        assert result.direction == "nl_to_asp"
        assert result.source == sample_nl_description
        assert "enforceable" in result.target
        assert result.confidence == 0.88

    def test_nl_to_asp_with_empty_predicates(self, translator, sample_nl_description):
        """Test NL to ASP translation with empty predicates list."""
        mock_response = MagicMock()
        mock_response.content = NLToASPSchema(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["enforceable", "contract"],
            reasoning="Inferred predicates.",
            confidence=0.7,
            assumptions=[],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        result = translator.nl_to_asp(sample_nl_description, [])
        assert result.target == "enforceable(C) :- contract(C)."


# =============================================================================
# Test Roundtrip Validation
# =============================================================================


class TestRoundtripValidate:
    """Tests for TranslatorLLM.roundtrip_validate method."""

    def test_roundtrip_empty_raises(self, translator):
        """Test that empty original raises ValueError."""
        with pytest.raises(ValueError, match="original cannot be empty"):
            translator.roundtrip_validate("", is_asp=True)

    def test_roundtrip_asp_original(
        self, translator, sample_asp_rule, sample_predicates
    ):
        """Test roundtrip validation starting with ASP."""
        # Mock ASP to NL
        mock_asp_to_nl = MagicMock()
        mock_asp_to_nl.content = ASPToNLSchema(
            natural_language="A contract is enforceable if signed.",
            predicates_identified=["enforceable", "contract", "signed"],
            reasoning="Translation",
            confidence=0.9,
            ambiguities=[],
        )

        # Mock NL to ASP
        mock_nl_to_asp = MagicMock()
        mock_nl_to_asp.content = NLToASPSchema(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            predicates_used=["enforceable", "contract", "signed"],
            reasoning="Back translation",
            confidence=0.85,
            assumptions=[],
        )

        # Mock fidelity assessment
        mock_fidelity = MagicMock()
        mock_fidelity.content = FidelityAssessmentSchema(
            fidelity_score=0.92,
            preserved_elements=["enforceable", "contract", "signed"],
            lost_elements=[],
            added_elements=[],
            reasoning="High fidelity preservation",
            confidence=0.9,
        )

        translator._llm.query = MagicMock(
            side_effect=[mock_asp_to_nl, mock_nl_to_asp, mock_fidelity]
        )

        result = translator.roundtrip_validate(
            sample_asp_rule,
            is_asp=True,
            predicates=sample_predicates,
        )

        assert isinstance(result, RoundtripResult)
        assert result.is_asp_original is True
        assert result.original == sample_asp_rule
        assert result.fidelity_score == 0.92

    def test_roundtrip_nl_original(
        self, translator, sample_nl_description, sample_predicates
    ):
        """Test roundtrip validation starting with NL."""
        # Mock NL to ASP
        mock_nl_to_asp = MagicMock()
        mock_nl_to_asp.content = NLToASPSchema(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            predicates_used=["enforceable", "contract", "signed"],
            reasoning="Translation",
            confidence=0.9,
            assumptions=[],
        )

        # Mock ASP to NL
        mock_asp_to_nl = MagicMock()
        mock_asp_to_nl.content = ASPToNLSchema(
            natural_language="A contract is enforceable if it is signed.",
            predicates_identified=["enforceable", "contract", "signed"],
            reasoning="Back translation",
            confidence=0.85,
            ambiguities=[],
        )

        # Mock fidelity assessment
        mock_fidelity = MagicMock()
        mock_fidelity.content = FidelityAssessmentSchema(
            fidelity_score=0.88,
            preserved_elements=["enforceable", "signed"],
            lost_elements=["consideration"],
            added_elements=[],
            reasoning="Minor loss",
            confidence=0.85,
        )

        translator._llm.query = MagicMock(
            side_effect=[mock_nl_to_asp, mock_asp_to_nl, mock_fidelity]
        )

        result = translator.roundtrip_validate(
            sample_nl_description,
            is_asp=False,
            predicates=sample_predicates,
        )

        assert result.is_asp_original is False
        assert result.fidelity_score == 0.88


# =============================================================================
# Test Statistics
# =============================================================================


class TestStatistics:
    """Tests for TranslatorLLM statistics methods."""

    def test_get_statistics_initial(self, translator):
        """Test initial statistics are zero."""
        stats = translator.get_statistics()
        assert stats["total_translations"] == 0
        assert stats["asp_to_nl_count"] == 0
        assert stats["nl_to_asp_count"] == 0
        assert stats["roundtrip_count"] == 0
        assert stats["cache_hits"] == 0

    def test_statistics_after_translations(self, translator, sample_asp_rule):
        """Test statistics are updated after translations."""
        mock_response = MagicMock()
        mock_response.content = ASPToNLSchema(
            natural_language="Translation",
            predicates_identified=[],
            reasoning="Test",
            confidence=0.8,
            ambiguities=[],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        translator.asp_to_nl(sample_asp_rule)
        translator.asp_to_nl(sample_asp_rule)  # Cache hit

        stats = translator.get_statistics()
        assert stats["total_translations"] == 2
        assert stats["asp_to_nl_count"] == 2
        assert stats["cache_hits"] == 1

    def test_reset_statistics(self, translator):
        """Test resetting statistics."""
        translator._total_translations = 10
        translator._cache_hits = 5

        translator.reset_statistics()

        stats = translator.get_statistics()
        assert stats["total_translations"] == 0
        assert stats["cache_hits"] == 0


# =============================================================================
# Test Cache Operations
# =============================================================================


class TestCacheOperations:
    """Tests for TranslatorLLM cache operations."""

    def test_clear_cache(self, translator, sample_asp_rule):
        """Test clearing the cache."""
        mock_response = MagicMock()
        mock_response.content = ASPToNLSchema(
            natural_language="Translation",
            predicates_identified=[],
            reasoning="Test",
            confidence=0.8,
            ambiguities=[],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        # Populate cache
        translator.asp_to_nl(sample_asp_rule)
        assert len(translator._cache) > 0

        # Clear cache
        translator.clear_cache()
        assert len(translator._cache) == 0

    def test_cache_disabled(self, mock_provider):
        """Test with cache disabled."""
        config = TranslatorConfig(enable_cache=False)
        with patch("loft.neural.ensemble.translator.LLMInterface"):
            translator = TranslatorLLM(mock_provider, config)

        mock_response = MagicMock()
        mock_response.content = ASPToNLSchema(
            natural_language="Translation",
            predicates_identified=[],
            reasoning="Test",
            confidence=0.8,
            ambiguities=[],
        )
        translator._llm.query = MagicMock(return_value=mock_response)

        result1 = translator.asp_to_nl("enforceable(C) :- contract(C).")
        result2 = translator.asp_to_nl("enforceable(C) :- contract(C).")

        # Both should not be from cache when caching is disabled
        assert result1.from_cache is False
        assert result2.from_cache is False


# =============================================================================
# Test Predicate Extraction
# =============================================================================


class TestPredicateExtraction:
    """Tests for predicate extraction helper."""

    def test_extract_predicates_simple(self, translator):
        """Test extracting predicates from simple rule."""
        predicates = translator._extract_predicates(
            "enforceable(C) :- contract(C), signed(C)."
        )
        assert "enforceable" in predicates
        assert "contract" in predicates
        assert "signed" in predicates

    def test_extract_predicates_with_negation(self, translator):
        """Test extracting predicates excludes 'not'."""
        predicates = translator._extract_predicates(
            "enforceable(C) :- contract(C), not void(C)."
        )
        assert "not" not in predicates
        assert "void" in predicates

    def test_extract_predicates_empty(self, translator):
        """Test extracting predicates from empty string."""
        predicates = translator._extract_predicates("")
        assert predicates == []


# =============================================================================
# Test Heuristic Fidelity Assessment
# =============================================================================


class TestHeuristicFidelity:
    """Tests for heuristic fidelity assessment."""

    def test_heuristic_fidelity_asp(self, translator):
        """Test heuristic fidelity for ASP comparison."""
        result = translator._heuristic_fidelity_assessment(
            "enforceable(C) :- contract(C), signed(C).",
            "enforceable(X) :- contract(X), signed(X).",
            is_asp_original=True,
        )
        assert result["fidelity_score"] > 0.5
        assert "confidence" in result

    def test_heuristic_fidelity_nl(self, translator):
        """Test heuristic fidelity for NL comparison."""
        result = translator._heuristic_fidelity_assessment(
            "A contract is enforceable if signed.",
            "A contract becomes enforceable when signed.",
            is_asp_original=False,
        )
        assert result["fidelity_score"] > 0.3
        assert "preserved" in result


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in TranslatorLLM."""

    def test_translation_error_creation(self):
        """Test TranslationError creation."""
        error = TranslationError(
            "Translation failed",
            direction="asp_to_nl",
            attempts=3,
            last_error="Network timeout",
        )
        assert str(error) == "Translation failed"
        assert error.direction == "asp_to_nl"
        assert error.attempts == 3
        assert error.last_error == "Network timeout"

    def test_llm_response_parsing_error(self):
        """Test LLMResponseParsingError creation."""
        error = LLMResponseParsingError("Failed to parse response")
        assert str(error) == "Failed to parse response"


# =============================================================================
# Test Pydantic Schemas
# =============================================================================


class TestPydanticSchemas:
    """Tests for Pydantic schemas."""

    def test_asp_to_nl_schema_validation(self):
        """Test ASPToNLSchema validation."""
        schema = ASPToNLSchema(
            natural_language="Test translation",
            predicates_identified=["pred1", "pred2"],
            reasoning="Test reasoning",
            confidence=0.85,
            ambiguities=["amb1"],
        )
        assert schema.natural_language == "Test translation"
        assert schema.confidence == 0.85

    def test_asp_to_nl_schema_confidence_bounds(self):
        """Test ASPToNLSchema confidence bounds."""
        with pytest.raises(Exception):  # Pydantic validation error
            ASPToNLSchema(
                natural_language="Test",
                predicates_identified=[],
                reasoning="Test",
                confidence=1.5,  # Invalid: > 1.0
                ambiguities=[],
            )

    def test_nl_to_asp_schema_validation(self):
        """Test NLToASPSchema validation."""
        schema = NLToASPSchema(
            asp_rule="test(X) :- pred(X).",
            predicates_used=["test", "pred"],
            reasoning="Test reasoning",
            confidence=0.9,
            assumptions=[],
        )
        assert schema.asp_rule == "test(X) :- pred(X)."

    def test_fidelity_assessment_schema_validation(self):
        """Test FidelityAssessmentSchema validation."""
        schema = FidelityAssessmentSchema(
            fidelity_score=0.95,
            preserved_elements=["elem1", "elem2"],
            lost_elements=[],
            added_elements=[],
            reasoning="High fidelity",
            confidence=0.9,
        )
        assert schema.fidelity_score == 0.95


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_exports_from_init(self):
        """Test that classes are exported from __init__.py."""
        from loft.neural.ensemble import (
            TranslatorLLM,
            TranslatorConfig,
            create_translation_strategy,
        )

        assert TranslatorLLM is not None
        assert TranslatorConfig is not None
        assert create_translation_strategy is not None
