"""Unit tests for the LogicGeneratorLLM class.

Tests the specialized ASP rule generator for Phase 6 heterogeneous neural ensemble.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from datetime import datetime

from loft.neural.ensemble.logic_generator import (
    LogicGeneratorLLM,
    LogicGeneratorConfig,
    ASPGenerationResult,
    BenchmarkResult,
    OptimizationStrategyType,
    ASP_LOGIC_SYSTEM_PROMPT,
)
from loft.neural.rule_schemas import GeneratedRule
from loft.neural.llm_interface import ResponseMetadata


@dataclass
class MockLLMResponse:
    """Mock LLM response for testing."""

    content: GeneratedRule
    raw_text: str = ""
    confidence: float = 0.9
    metadata: ResponseMetadata = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = ResponseMetadata(
                model="test-model",
                tokens_input=100,
                tokens_output=50,
                tokens_total=150,
                latency_ms=100.0,
                cost_usd=0.001,
                timestamp=datetime.utcnow().isoformat(),
                provider="test",
            )


class TestLogicGeneratorConfig:
    """Tests for LogicGeneratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogicGeneratorConfig()

        assert config.model == "claude-3-5-haiku-20241022"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.optimization_strategy == OptimizationStrategyType.PROMPT_OPTIMIZATION
        assert config.enable_syntax_validation is True
        assert config.enable_variable_safety_check is True
        assert config.max_generation_retries == 3
        assert config.few_shot_examples == []

    def test_custom_config(self):
        """Test custom configuration."""
        config = LogicGeneratorConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=0.5,
            optimization_strategy=OptimizationStrategyType.CHAIN_OF_THOUGHT,
            few_shot_examples=["example1", "example2"],
        )

        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.5
        assert config.optimization_strategy == OptimizationStrategyType.CHAIN_OF_THOUGHT
        assert len(config.few_shot_examples) == 2


class TestASPGenerationResult:
    """Tests for ASPGenerationResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid generation result."""
        result = ASPGenerationResult(
            rule="enforceable(C) :- contract(C), signed(C).",
            is_valid=True,
            confidence=0.95,
            generation_time_ms=150.5,
            retries_needed=0,
        )

        assert result.rule == "enforceable(C) :- contract(C), signed(C)."
        assert result.is_valid is True
        assert result.confidence == 0.95
        assert result.validation_errors == []

    def test_invalid_result_with_errors(self):
        """Test creating an invalid generation result."""
        result = ASPGenerationResult(
            rule="invalid_rule",
            is_valid=False,
            confidence=0.3,
            validation_errors=["Rule must end with period", "Syntax error"],
        )

        assert result.is_valid is False
        assert len(result.validation_errors) == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating benchmark results."""
        result = BenchmarkResult(
            logic_generator_accuracy=0.95,
            general_llm_accuracy=0.75,
            logic_generator_syntax_valid_rate=0.98,
            general_llm_syntax_valid_rate=0.80,
            logic_generator_avg_time_ms=150.0,
            general_llm_avg_time_ms=200.0,
            test_cases_count=100,
            improvement_percentage=26.67,
        )

        assert result.logic_generator_accuracy == 0.95
        assert result.improvement_percentage == 26.67
        assert result.test_cases_count == 100


class TestLogicGeneratorLLM:
    """Tests for the LogicGeneratorLLM class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.get_provider_name.return_value = "test"
        return provider

    @pytest.fixture
    def logic_generator(self, mock_provider):
        """Create a LogicGeneratorLLM instance with mock provider."""
        return LogicGeneratorLLM(mock_provider)

    def test_initialization_default_config(self, mock_provider):
        """Test initialization with default configuration."""
        gen = LogicGeneratorLLM(mock_provider)

        assert gen.provider == mock_provider
        assert gen.config.model == "claude-3-5-haiku-20241022"
        assert len(gen.config.few_shot_examples) > 0  # Default examples loaded

    def test_initialization_custom_config(self, mock_provider):
        """Test initialization with custom configuration."""
        config = LogicGeneratorConfig(
            temperature=0.5,
            optimization_strategy=OptimizationStrategyType.FEW_SHOT_LEARNING,
        )
        gen = LogicGeneratorLLM(mock_provider, config)

        assert gen.config.temperature == 0.5
        assert gen.config.optimization_strategy == OptimizationStrategyType.FEW_SHOT_LEARNING

    def test_validate_rule_valid(self, logic_generator):
        """Test validation of a valid ASP rule."""
        valid_rule = "enforceable(C) :- contract(C), signed(C, P), party(P)."
        is_valid, errors = logic_generator.validate_rule(valid_rule)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_rule_missing_period(self, logic_generator):
        """Test validation catches missing period."""
        invalid_rule = "enforceable(C) :- contract(C)"
        is_valid, errors = logic_generator.validate_rule(invalid_rule)

        assert is_valid is False
        assert len(errors) > 0
        assert any("period" in e.lower() for e in errors)

    def test_validate_rule_empty(self, logic_generator):
        """Test validation of empty rule."""
        is_valid, errors = logic_generator.validate_rule("")

        assert is_valid is False
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_validate_rule_unbalanced_parentheses(self, logic_generator):
        """Test validation catches unbalanced parentheses."""
        invalid_rule = "enforceable(C :- contract(C)."
        is_valid, errors = logic_generator.validate_rule(invalid_rule)

        assert is_valid is False
        assert len(errors) > 0
        assert any("parenthes" in e.lower() for e in errors)

    def test_build_generation_prompt_basic(self, logic_generator):
        """Test building a basic generation prompt."""
        prompt = logic_generator._build_generation_prompt(
            principle="A contract requires offer and acceptance",
            predicates=["contract(X)", "offer(X)", "acceptance(X)"],
            domain="contracts",
            context=None,
        )

        assert "A contract requires offer and acceptance" in prompt
        assert "contract(X)" in prompt
        assert "contracts" in prompt

    def test_build_generation_prompt_with_context(self, logic_generator):
        """Test building prompt with additional context."""
        prompt = logic_generator._build_generation_prompt(
            principle="Test principle",
            predicates=None,
            domain="legal",
            context={"jurisdiction": "CA", "case_type": "civil"},
        )

        assert "jurisdiction" in prompt
        assert "CA" in prompt

    def test_build_generation_prompt_few_shot(self, mock_provider):
        """Test that few-shot strategy adds reference examples to prompt."""
        config = LogicGeneratorConfig(
            optimization_strategy=OptimizationStrategyType.FEW_SHOT_LEARNING,
            few_shot_examples=[
                "valid_contract(C) :- offer(C), acceptance(C).",
            ],
        )
        gen = LogicGeneratorLLM(mock_provider, config)

        base_prompt = gen._build_generation_prompt(
            principle="Test",
            predicates=None,
            domain="legal",
            context=None,
        )

        # Apply the strategy to the base prompt
        final_prompt = gen._strategy.prepare_prompt(
            base_prompt=base_prompt,
            principle="Test",
            predicates=None,
            domain="legal",
            context=None,
            few_shot_examples=config.few_shot_examples,
        )

        assert "Reference Examples" in final_prompt
        assert "valid_contract" in final_prompt

    def test_build_generation_prompt_chain_of_thought(self, mock_provider):
        """Test that chain-of-thought strategy adds reasoning instructions."""
        config = LogicGeneratorConfig(
            optimization_strategy=OptimizationStrategyType.CHAIN_OF_THOUGHT,
        )
        gen = LogicGeneratorLLM(mock_provider, config)

        base_prompt = gen._build_generation_prompt(
            principle="Test",
            predicates=None,
            domain="legal",
            context=None,
        )

        # Apply the strategy to the base prompt
        final_prompt = gen._strategy.prepare_prompt(
            base_prompt=base_prompt,
            principle="Test",
            predicates=None,
            domain="legal",
            context=None,
        )

        assert "Reasoning Process" in final_prompt
        assert "think through" in final_prompt.lower()

    def test_build_gap_filling_prompt(self, logic_generator):
        """Test building gap filling prompt."""
        prompt = logic_generator._build_gap_filling_prompt(
            gap_description="Missing definition for enforceability",
            missing_predicate="enforceable(C)",
            dataset_predicates=["contract(X)", "signed(X, P)"],
            approach="conservative",
        )

        assert "enforceable(C)" in prompt
        assert "contract(X)" in prompt
        assert "conservative" in prompt.lower() or "ALL conditions" in prompt

    def test_build_retry_prompt(self, logic_generator):
        """Test building retry prompt with error feedback."""
        prompt = logic_generator._build_retry_prompt(
            original_prompt="Generate an ASP rule...",
            failed_rule="invalid(X :- broken.",
            errors=["Unbalanced parentheses", "Missing period"],
        )

        assert "PREVIOUS ATTEMPT FAILED" in prompt
        assert "invalid(X :- broken." in prompt
        assert "Unbalanced parentheses" in prompt

    def test_get_statistics_initial(self, logic_generator):
        """Test initial statistics are zero."""
        stats = logic_generator.get_statistics()

        assert stats["total_generations"] == 0
        assert stats["successful_generations"] == 0
        assert stats["success_rate"] == 0.0

    def test_reset_statistics(self, logic_generator):
        """Test resetting statistics."""
        # Manually set some stats
        logic_generator._total_generations = 10
        logic_generator._successful_generations = 8

        logic_generator.reset_statistics()

        assert logic_generator._total_generations == 0
        assert logic_generator._successful_generations == 0

    @patch("loft.neural.ensemble.logic_generator.LLMInterface")
    def test_generate_rule_success(self, mock_llm_class, mock_provider):
        """Test successful rule generation."""
        # Setup mock
        mock_llm_instance = MagicMock()
        mock_generated_rule = GeneratedRule(
            asp_rule="enforceable(C) :- contract(C), signed(C, P), party(P).",
            confidence=0.9,
            reasoning="Contract enforceability requires signature.",
            predicates_used=["contract", "signed", "party"],
            new_predicates=["enforceable"],
            source_type="principle",
            source_text="A contract is enforceable when signed",
        )
        mock_response = MockLLMResponse(content=mock_generated_rule)
        mock_llm_instance.query.return_value = mock_response
        mock_llm_class.return_value = mock_llm_instance

        gen = LogicGeneratorLLM(mock_provider)
        gen._llm = mock_llm_instance

        result = gen.generate_rule(
            principle="A contract is enforceable when signed",
            predicates=["contract(X)", "signed(X, P)", "party(P)"],
            domain="contracts",
        )

        assert result.is_valid is True
        assert result.confidence == 0.9
        assert "enforceable" in result.rule

    @patch("loft.neural.ensemble.logic_generator.LLMInterface")
    def test_generate_rule_with_retries(self, mock_llm_class, mock_provider):
        """Test rule generation with retries on validation failure."""
        mock_llm_instance = MagicMock()

        # Create a mock for an invalid rule response (can't use GeneratedRule
        # directly since Pydantic validates ASP syntax)
        invalid_rule_mock = MagicMock()
        invalid_rule_mock.asp_rule = "invalid(X :- broken"  # Missing period and unbalanced
        invalid_rule_mock.confidence = 0.5
        invalid_rule_mock.reasoning = "Test reasoning"

        # Second call returns valid rule
        valid_rule = GeneratedRule(
            asp_rule="valid(X) :- condition(X).",
            confidence=0.8,
            reasoning="Test reasoning",
            predicates_used=["condition"],
            new_predicates=["valid"],
            source_type="principle",
            source_text="Test principle",
        )

        mock_llm_instance.query.side_effect = [
            MockLLMResponse(content=invalid_rule_mock),
            MockLLMResponse(content=valid_rule),
        ]
        mock_llm_class.return_value = mock_llm_instance

        gen = LogicGeneratorLLM(mock_provider)
        gen._llm = mock_llm_instance

        result = gen.generate_rule(
            principle="Test principle",
            predicates=["condition(X)"],
        )

        assert result.retries_needed == 1
        assert result.is_valid is True

    def test_get_default_few_shot_examples(self, logic_generator):
        """Test default few-shot examples are loaded."""
        examples = logic_generator._get_default_few_shot_examples()

        assert len(examples) > 0
        # Should contain valid ASP rules
        assert any(":-" in ex for ex in examples)
        assert any("." in ex for ex in examples)


class TestOptimizationStrategyType:
    """Tests for OptimizationStrategyType enum."""

    def test_enum_values(self):
        """Test all optimization strategy values."""
        assert OptimizationStrategyType.PROMPT_OPTIMIZATION.value == "prompt_optimization"
        assert OptimizationStrategyType.FEW_SHOT_LEARNING.value == "few_shot_learning"
        assert OptimizationStrategyType.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert OptimizationStrategyType.SELF_CONSISTENCY.value == "self_consistency"


class TestASPLogicSystemPrompt:
    """Tests for the ASP logic system prompt."""

    def test_system_prompt_contains_asp_rules(self):
        """Test that system prompt contains ASP syntax rules."""
        assert "Variables start with UPPERCASE" in ASP_LOGIC_SYSTEM_PROMPT
        assert "Constants start with lowercase" in ASP_LOGIC_SYSTEM_PROMPT
        assert "period" in ASP_LOGIC_SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_examples(self):
        """Test that system prompt contains examples."""
        assert "Example Valid Rules" in ASP_LOGIC_SYSTEM_PROMPT
        assert ":-" in ASP_LOGIC_SYSTEM_PROMPT

    def test_system_prompt_variable_safety(self):
        """Test that system prompt emphasizes variable safety."""
        assert "Variable Safety" in ASP_LOGIC_SYSTEM_PROMPT
        assert "positive" in ASP_LOGIC_SYSTEM_PROMPT.lower()


class TestLogicGeneratorGapFilling:
    """Tests for gap filling functionality."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.get_provider_name.return_value = "test"
        return provider

    @patch("loft.neural.ensemble.logic_generator.LLMInterface")
    def test_generate_gap_filling_candidates_count(self, mock_llm_class, mock_provider):
        """Test generating correct number of candidates."""
        mock_llm_instance = MagicMock()

        valid_rule = GeneratedRule(
            asp_rule="missing_pred(X) :- condition(X).",
            confidence=0.8,
            reasoning="Test reasoning for gap filling",
            predicates_used=["condition"],
            new_predicates=["missing_pred"],
            source_type="gap_fill",
            source_text="Missing predicate definition",
        )
        mock_llm_instance.query.return_value = MockLLMResponse(content=valid_rule)
        mock_llm_class.return_value = mock_llm_instance

        gen = LogicGeneratorLLM(mock_provider)
        gen._llm = mock_llm_instance

        candidates = gen.generate_gap_filling_candidates(
            gap_description="Missing predicate definition",
            missing_predicate="missing_pred(X)",
            dataset_predicates=["condition(X)"],
            num_candidates=3,
        )

        assert len(candidates) == 3
        assert mock_llm_instance.query.call_count == 3


class TestLogicGeneratorBenchmarking:
    """Tests for benchmarking functionality."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.get_provider_name.return_value = "test"
        return provider

    @patch("loft.neural.ensemble.logic_generator.LLMInterface")
    def test_benchmark_against_general_llm(self, mock_llm_class, mock_provider):
        """Test benchmarking against general LLM."""
        mock_llm_instance = MagicMock()

        valid_rule = GeneratedRule(
            asp_rule="test(X) :- condition(X).",
            confidence=0.9,
            reasoning="Test reasoning for benchmark",
            predicates_used=["condition"],
            new_predicates=["test"],
            source_type="principle",
            source_text="Test principle for benchmark",
        )
        mock_llm_instance.query.return_value = MockLLMResponse(content=valid_rule)
        mock_llm_class.return_value = mock_llm_instance

        gen = LogicGeneratorLLM(mock_provider)
        gen._llm = mock_llm_instance

        # Create mock general LLM
        mock_general_llm = MagicMock()
        mock_general_llm.query.return_value = MockLLMResponse(content=valid_rule)

        test_cases = [
            {"principle": "Test 1", "predicates": ["condition(X)"]},
            {"principle": "Test 2", "predicates": ["other(X)"]},
        ]

        result = gen.benchmark_against_general_llm(test_cases, mock_general_llm)

        assert isinstance(result, BenchmarkResult)
        assert result.test_cases_count == 2
        assert result.logic_generator_syntax_valid_rate >= 0.0
        assert result.general_llm_syntax_valid_rate >= 0.0
