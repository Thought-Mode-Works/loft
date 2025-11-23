"""
Unit tests for neural (LLM) interface components.

Tests LLMQuery, LLMResponse, providers, prompts, and error handling.
"""

import pytest
from datetime import datetime
from pydantic import BaseModel
from unittest.mock import Mock

from loft.neural import (
    LLMQuery,
    LLMResponse,
    ResponseMetadata,
    LLMInterface,
    AnthropicProvider,
    OpenAIProvider,
    LocalProvider,
    DefaultResponse,
    PromptTemplate,
    get_template,
    list_templates,
    LLMProviderError,
    LLMParsingError,
    LLMRateLimitError,
)


class TestLLMQuery:
    """Tests for LLMQuery dataclass."""

    def test_query_creation(self) -> None:
        """Test creating an LLM query."""
        query = LLMQuery(
            question="What is contract formation?",
            context={"domain": "contracts"},
            temperature=0.8,
            max_tokens=2048,
        )

        assert query.question == "What is contract formation?"
        assert query.context == {"domain": "contracts"}
        assert query.temperature == 0.8
        assert query.max_tokens == 2048

    def test_query_defaults(self) -> None:
        """Test default values."""
        query = LLMQuery(question="Test question")

        assert query.context == {}
        assert query.output_schema is None
        assert query.temperature == 0.7
        assert query.max_tokens == 4096
        assert query.system_prompt is None
        assert query.few_shot_examples == []
        assert query.cot_enabled is False

    def test_query_to_dict(self) -> None:
        """Test converting query to dict."""
        query = LLMQuery(
            question="Test",
            context={"key": "value"},
            temperature=0.5,
        )

        data = query.to_dict()
        assert data["question"] == "Test"
        assert data["context"] == {"key": "value"}
        assert data["temperature"] == 0.5


class TestResponseMetadata:
    """Tests for ResponseMetadata."""

    def test_metadata_creation(self) -> None:
        """Test creating response metadata."""
        metadata = ResponseMetadata(
            model="claude-3-5-sonnet-20241022",
            tokens_input=100,
            tokens_output=200,
            tokens_total=300,
            latency_ms=1500.0,
            cost_usd=0.0015,
            timestamp=datetime.utcnow().isoformat(),
            provider="anthropic",
        )

        assert metadata.model == "claude-3-5-sonnet-20241022"
        assert metadata.tokens_total == 300
        assert metadata.cost_usd == 0.0015
        assert metadata.provider == "anthropic"

    def test_metadata_to_dict(self) -> None:
        """Test converting metadata to dict."""
        metadata = ResponseMetadata(
            model="gpt-4",
            tokens_input=50,
            tokens_output=100,
            tokens_total=150,
            latency_ms=2000.0,
            cost_usd=0.002,
            timestamp="2025-01-01T00:00:00",
            provider="openai",
            retries=2,
            cache_hit=True,
        )

        data = metadata.to_dict()
        assert data["model"] == "gpt-4"
        assert data["retries"] == 2
        assert data["cache_hit"] is True


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_response_creation(self) -> None:
        """Test creating an LLM response."""
        content = DefaultResponse(response="Test response")
        metadata = ResponseMetadata(
            model="claude-3-5-sonnet-20241022",
            tokens_input=10,
            tokens_output=20,
            tokens_total=30,
            latency_ms=500.0,
            cost_usd=0.0001,
            timestamp=datetime.utcnow().isoformat(),
            provider="anthropic",
        )

        response = LLMResponse(
            content=content,
            raw_text="Test response",
            confidence=0.9,
            metadata=metadata,
        )

        assert response.content.response == "Test response"
        assert response.confidence == 0.9
        assert response.metadata.provider == "anthropic"

    def test_response_to_dict(self) -> None:
        """Test converting response to dict."""
        content = DefaultResponse(response="Test")
        metadata = ResponseMetadata(
            model="test",
            tokens_input=1,
            tokens_output=1,
            tokens_total=2,
            latency_ms=100.0,
            cost_usd=0.0,
            timestamp="2025-01-01T00:00:00",
            provider="test",
        )

        response = LLMResponse(
            content=content,
            raw_text="Test",
            confidence=0.8,
            metadata=metadata,
            reasoning="Because...",
        )

        data = response.to_dict()
        assert "content" in data
        assert data["confidence"] == 0.8
        assert data["reasoning"] == "Because..."


class TestPromptTemplate:
    """Tests for prompt template system."""

    def test_template_creation(self) -> None:
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="test_template",
            version="1.0",
            template="Question: $question\nContext: $context",
            required_variables=["question", "context"],
        )

        assert template.name == "test_template"
        assert template.version == "1.0"
        assert len(template.required_variables) == 2

    def test_template_render(self) -> None:
        """Test rendering a template."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            template="Hello $name, you are $age years old.",
            required_variables=["name", "age"],
        )

        rendered = template.render({"name": "Alice", "age": "30"})
        assert "Alice" in rendered
        assert "30" in rendered

    def test_template_missing_variable(self) -> None:
        """Test error when required variable is missing."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            template="Hello $name",
            required_variables=["name"],
        )

        with pytest.raises(KeyError, match="Missing required variables"):
            template.render({})

    def test_get_template(self) -> None:
        """Test getting templates from registry."""
        template = get_template("gap_identification")
        assert template.name == "gap_identification"
        assert "question" in template.required_variables

    def test_get_nonexistent_template(self) -> None:
        """Test error when template doesn't exist."""
        with pytest.raises(KeyError, match="not found"):
            get_template("nonexistent_template")

    def test_list_templates(self) -> None:
        """Test listing all templates."""
        templates = list_templates()
        assert "gap_identification" in templates
        assert "element_extraction" in templates
        assert "rule_proposal" in templates


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_provider_initialization(self) -> None:
        """Test creating Anthropic provider."""
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
        )

        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.get_provider_name() == "anthropic"

    def test_estimate_cost(self) -> None:
        """Test cost estimation."""
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
        )

        # 1M input tokens, 1M output tokens
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        # Should be 3.00 + 15.00 = 18.00
        assert cost == 18.00

    def test_estimate_cost_haiku(self) -> None:
        """Test cost estimation for Haiku."""
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-haiku-20241022",
        )

        cost = provider.estimate_cost(1_000_000, 1_000_000)
        # Should be 0.80 + 4.00 = 4.80
        assert cost == 4.80


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_provider_initialization(self) -> None:
        """Test creating OpenAI provider."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4",
        )

        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"
        assert provider.get_provider_name() == "openai"

    def test_estimate_cost(self) -> None:
        """Test cost estimation."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4",
        )

        cost = provider.estimate_cost(1_000_000, 1_000_000)
        # Should be 30.00 + 60.00 = 90.00
        assert cost == 90.00


class TestLocalProvider:
    """Tests for Local (Ollama) provider."""

    def test_provider_initialization(self) -> None:
        """Test creating local provider."""
        provider = LocalProvider(
            api_key="",  # Not needed for local
            model="llama2",
            base_url="http://localhost:11434",
        )

        assert provider.model == "llama2"
        assert provider.get_provider_name() == "local"

    def test_estimate_cost(self) -> None:
        """Test cost estimation (should be zero for local)."""
        provider = LocalProvider(api_key="", model="llama2")

        cost = provider.estimate_cost(1_000_000, 1_000_000)
        assert cost == 0.0


class TestLLMInterface:
    """Tests for main LLM interface."""

    def test_interface_initialization(self) -> None:
        """Test creating LLM interface."""
        provider = Mock()
        interface = LLMInterface(provider)

        assert interface.provider == provider
        assert interface.enable_cache is True
        assert interface.max_retries == 3

    def test_cost_tracking(self) -> None:
        """Test total cost tracking."""
        provider = Mock()
        interface = LLMInterface(provider)

        # Create mock responses with costs
        metadata1 = ResponseMetadata(
            model="test",
            tokens_input=100,
            tokens_output=100,
            tokens_total=200,
            latency_ms=100.0,
            cost_usd=0.01,
            timestamp=datetime.utcnow().isoformat(),
            provider="test",
        )

        metadata2 = ResponseMetadata(
            model="test",
            tokens_input=200,
            tokens_output=200,
            tokens_total=400,
            latency_ms=200.0,
            cost_usd=0.02,
            timestamp=datetime.utcnow().isoformat(),
            provider="test",
        )

        response1 = LLMResponse(
            content=DefaultResponse(response="Test 1"),
            raw_text="Test 1",
            confidence=0.8,
            metadata=metadata1,
        )

        response2 = LLMResponse(
            content=DefaultResponse(response="Test 2"),
            raw_text="Test 2",
            confidence=0.8,
            metadata=metadata2,
        )

        provider.query.side_effect = [response1, response2]

        # Make queries
        interface.query("Question 1")
        interface.query("Question 2")

        # Check total cost
        assert interface.get_total_cost() == 0.03
        assert interface.get_total_tokens() == 600

    def test_caching(self) -> None:
        """Test response caching."""
        provider = Mock()
        interface = LLMInterface(provider, enable_cache=True)

        metadata = ResponseMetadata(
            model="test",
            tokens_input=10,
            tokens_output=10,
            tokens_total=20,
            latency_ms=100.0,
            cost_usd=0.001,
            timestamp=datetime.utcnow().isoformat(),
            provider="test",
        )

        response = LLMResponse(
            content=DefaultResponse(response="Cached response"),
            raw_text="Cached response",
            confidence=0.8,
            metadata=metadata,
        )

        provider.query.return_value = response

        # First query - should call provider
        result1 = interface.query("Test question")
        assert provider.query.call_count == 1
        assert not result1.metadata.cache_hit

        # Second identical query - should use cache
        result2 = interface.query("Test question")
        assert provider.query.call_count == 1  # Not called again
        assert result2.metadata.cache_hit

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        provider = Mock()
        interface = LLMInterface(provider, enable_cache=True)

        metadata = ResponseMetadata(
            model="test",
            tokens_input=10,
            tokens_output=10,
            tokens_total=20,
            latency_ms=100.0,
            cost_usd=0.001,
            timestamp=datetime.utcnow().isoformat(),
            provider="test",
        )

        response = LLMResponse(
            content=DefaultResponse(response="Response"),
            raw_text="Response",
            confidence=0.8,
            metadata=metadata,
        )

        provider.query.return_value = response

        # Make a query to populate cache
        interface.query("Test")
        assert provider.query.call_count == 1

        # Clear cache
        interface.clear_cache()

        # Query again - should call provider
        interface.query("Test")
        assert provider.query.call_count == 2


class TestErrors:
    """Tests for error handling."""

    def test_llm_provider_error(self) -> None:
        """Test LLM provider error."""
        error = LLMProviderError(
            "API error",
            provider="anthropic",
            status_code=500,
        )

        assert str(error) == "API error"
        assert error.provider == "anthropic"
        assert error.status_code == 500

    def test_llm_parsing_error(self) -> None:
        """Test parsing error."""
        error = LLMParsingError(
            "Invalid JSON",
            raw_response='{"invalid": }',
        )

        assert "Invalid JSON" in str(error)
        assert error.raw_response == '{"invalid": }'

    def test_llm_rate_limit_error(self) -> None:
        """Test rate limit error."""
        error = LLMRateLimitError(
            "Rate limit exceeded",
            provider="openai",
            retry_after=60,
        )

        assert error.provider == "openai"
        assert error.retry_after == 60
        assert error.status_code == 429


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_structured_output_workflow(self) -> None:
        """Test end-to-end structured output workflow."""

        # Define custom response model
        class ContractAnalysis(BaseModel):
            is_valid: bool
            confidence: float
            missing_elements: list[str]

        # Mock provider
        provider = Mock()

        # Create mock response
        analysis = ContractAnalysis(
            is_valid=True,
            confidence=0.95,
            missing_elements=[],
        )

        metadata = ResponseMetadata(
            model="claude-3-5-sonnet-20241022",
            tokens_input=100,
            tokens_output=50,
            tokens_total=150,
            latency_ms=1200.0,
            cost_usd=0.0015,
            timestamp=datetime.utcnow().isoformat(),
            provider="anthropic",
        )

        mock_response = LLMResponse(
            content=analysis,
            raw_text="...",
            confidence=0.95,
            metadata=metadata,
        )

        provider.query.return_value = mock_response

        # Create interface and query
        interface = LLMInterface(provider)
        response = interface.query(
            question="Is this contract valid?",
            context={"contract": "..."},
            output_schema=ContractAnalysis,
        )

        # Verify structured output
        assert isinstance(response.content, ContractAnalysis)
        assert response.content.is_valid is True
        assert response.content.confidence == 0.95
        assert len(response.content.missing_elements) == 0

    def test_prompt_template_integration(self) -> None:
        """Test using prompt templates with interface."""
        template = get_template("gap_identification")

        prompt = template.render(
            {
                "question": "Is contract enforceable?",
                "context": "Contract for land sale, no writing.",
            }
        )

        assert "Is contract enforceable?" in prompt
        assert "land sale" in prompt
        assert "missing" in prompt.lower()
