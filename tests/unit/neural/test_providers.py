"""
Unit tests for LLM providers with mocked API calls.

Tests AnthropicProvider, OpenAIProvider, and LocalProvider without making
actual API calls using mocked responses.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel

from loft.neural.providers import (
    AnthropicProvider,
    OpenAIProvider,
    LocalProvider,
    DefaultResponse,
)
from loft.neural.llm_interface import LLMQuery, LLMResponse
from loft.neural.errors import (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMProviderError,
    LLMParsingError,
)


# Test schemas
class StructuredOutputSchema(BaseModel):
    """Test Pydantic schema for structured outputs."""

    answer: str
    confidence: float = 0.8


class TestAnthropicProvider:
    """Test AnthropicProvider functionality."""

    @pytest.fixture
    def provider(self):
        """Create AnthropicProvider with mocked client."""
        with patch("loft.neural.providers.anthropic.Anthropic"):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")
            provider.client = Mock()
            return provider

    def test_init(self):
        """Test provider initialization."""
        with patch("loft.neural.providers.anthropic.Anthropic") as mock_anthropic:
            provider = AnthropicProvider(api_key="test-key", model="claude-3-5-haiku-20241022")

            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert provider.model == "claude-3-5-haiku-20241022"
            assert provider.api_key == "test-key"

    def test_get_provider_name(self, provider):
        """Test getting provider name."""
        assert provider.get_provider_name() == "anthropic"

    def test_estimate_cost_sonnet(self, provider):
        """Test cost estimation for Sonnet model."""
        # claude-3-5-sonnet-20241022: input=$3/M, output=$15/M
        cost = provider.estimate_cost(1000, 500)
        expected = (1000 / 1_000_000 * 3.00) + (500 / 1_000_000 * 15.00)
        assert cost == pytest.approx(expected)

    def test_estimate_cost_haiku(self):
        """Test cost estimation for Haiku model."""
        with patch("loft.neural.providers.anthropic.Anthropic"):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-5-haiku-20241022")
            # claude-3-5-haiku-20241022: input=$0.80/M, output=$4.00/M
            cost = provider.estimate_cost(1000, 500)
            expected = (1000 / 1_000_000 * 0.80) + (500 / 1_000_000 * 4.00)
            assert cost == pytest.approx(expected)

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model uses defaults."""
        with patch("loft.neural.providers.anthropic.Anthropic"):
            provider = AnthropicProvider(api_key="test-key", model="unknown-model")
            # Should use default: input=$3/M, output=$15/M
            cost = provider.estimate_cost(1000, 500)
            expected = (1000 / 1_000_000 * 3.00) + (500 / 1_000_000 * 15.00)
            assert cost == pytest.approx(expected)

    def test_build_messages_basic(self, provider):
        """Test building messages from query."""
        query = LLMQuery(question="What is contract law?")
        messages = provider._build_messages(query)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is contract law?"

    def test_build_messages_with_context(self, provider):
        """Test building messages with context."""
        query = LLMQuery(
            question="Is this enforceable?",
            context={"contract_type": "sale", "jurisdiction": "CA"},
        )
        messages = provider._build_messages(query)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert "Context:" in content
        assert "contract_type: sale" in content
        assert "jurisdiction: CA" in content
        assert "Is this enforceable?" in content

    def test_build_messages_with_few_shot(self, provider):
        """Test building messages with few-shot examples."""
        query = LLMQuery(
            question="Is this valid?",
            few_shot_examples=["Example 1: contract", "Example 2: void"],
        )
        messages = provider._build_messages(query)

        # Should have: example1 (user), ... (assistant), example2 (user), ... (assistant), question (user)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Example 1: contract"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "..."

    def test_build_messages_with_cot(self, provider):
        """Test building messages with chain-of-thought."""
        query = LLMQuery(question="Is this valid?", cot_enabled=True)
        messages = provider._build_messages(query)

        assert len(messages) == 1
        assert "Let's approach this step-by-step:" in messages[0]["content"]

    def test_get_default_system_prompt(self, provider):
        """Test default system prompt."""
        prompt = provider._get_default_system_prompt()
        assert "legal reasoning" in prompt.lower()
        assert "assistant" in prompt.lower()

    def test_query_unstructured_success(self, provider):
        """Test successful unstructured query."""
        # Mock response
        mock_content_block = Mock()
        mock_content_block.text = "The contract is enforceable."

        mock_response = Mock()
        mock_response.content = [mock_content_block]
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        provider.client.messages.create = Mock(return_value=mock_response)

        query = LLMQuery(question="Is this enforceable?")
        response = provider.query(query)

        assert isinstance(response, LLMResponse)
        assert isinstance(response.content, DefaultResponse)
        assert response.content.response == "The contract is enforceable."
        assert response.raw_text == "The contract is enforceable."
        assert response.confidence == 0.7
        assert response.metadata.tokens_input == 100
        assert response.metadata.tokens_output == 50
        assert response.metadata.provider == "anthropic"
        assert response.metadata.latency_ms > 0

    def test_query_structured_success(self, provider):
        """Test successful structured query."""
        # Mock instructor response
        mock_structured = StructuredOutputSchema(answer="Valid", confidence=0.9)
        mock_structured._raw_response = Mock()
        mock_structured._raw_response.usage = Mock()
        mock_structured._raw_response.usage.input_tokens = 100
        mock_structured._raw_response.usage.output_tokens = 50

        # Patch instructor module import
        mock_instructor = MagicMock()
        mock_client = Mock()
        mock_client.messages.create = Mock(return_value=mock_structured)
        mock_instructor.from_anthropic = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            query = LLMQuery(question="Is this valid?")
            response = provider.query(query, response_model=StructuredOutputSchema)

            assert isinstance(response, LLMResponse)
            assert isinstance(response.content, StructuredOutputSchema)
            assert response.content.answer == "Valid"
            assert response.confidence == 0.9
            assert response.metadata.tokens_input == 100
            assert response.metadata.tokens_output == 50

    def test_query_structured_with_schema_in_query(self, provider):
        """Test structured query with schema in LLMQuery."""
        mock_structured = StructuredOutputSchema(answer="Valid", confidence=0.9)
        mock_structured._raw_response = Mock()
        mock_structured._raw_response.usage = Mock()
        mock_structured._raw_response.usage.input_tokens = 100
        mock_structured._raw_response.usage.output_tokens = 50

        mock_instructor = MagicMock()
        mock_client = Mock()
        mock_client.messages.create = Mock(return_value=mock_structured)
        mock_instructor.from_anthropic = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            query = LLMQuery(question="Is this valid?", output_schema=StructuredOutputSchema)
            response = provider.query(query)

            assert isinstance(response.content, StructuredOutputSchema)

    def test_query_rate_limit_error(self, provider):
        """Test handling of rate limit errors."""
        from anthropic import RateLimitError

        # Create a mock response required by anthropic errors
        mock_response = Mock()
        mock_response.status_code = 429
        error = RateLimitError("Rate limited", response=mock_response, body=None)
        provider.client.messages.create = Mock(side_effect=error)

        query = LLMQuery(question="Test")
        with pytest.raises(LLMRateLimitError) as exc_info:
            provider.query(query)

        assert exc_info.value.provider == "anthropic"

    def test_query_timeout_error(self, provider):
        """Test handling of timeout errors."""
        import anthropic

        provider.client.messages.create = Mock(side_effect=anthropic.APITimeoutError("Timeout"))

        query = LLMQuery(question="Test")
        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.query(query)

        assert exc_info.value.provider == "anthropic"

    def test_query_api_error(self, provider):
        """Test handling of general API errors."""
        from anthropic import APIError

        # Create a mock request and set status_code as attribute
        mock_request = Mock()
        api_error = APIError("API Error", request=mock_request, body=None)
        api_error.status_code = 500

        provider.client.messages.create = Mock(side_effect=api_error)

        query = LLMQuery(question="Test")
        with pytest.raises(LLMProviderError) as exc_info:
            provider.query(query)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.status_code == 500

    def test_query_structured_parsing_error(self, provider):
        """Test handling of structured parsing errors."""
        mock_instructor = MagicMock()
        mock_client = Mock()
        mock_client.messages.create = Mock(side_effect=ValueError("Parse error"))
        mock_instructor.from_anthropic = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            query = LLMQuery(question="Test")
            with pytest.raises(LLMParsingError):
                provider.query(query, response_model=StructuredOutputSchema)


class TestOpenAIProvider:
    """Test OpenAIProvider functionality."""

    @pytest.fixture
    def provider(self):
        """Create OpenAIProvider with mocked client."""
        with patch("loft.neural.providers.openai.OpenAI"):
            provider = OpenAIProvider(api_key="test-key", model="gpt-4-turbo-preview")
            provider.client = Mock()
            return provider

    def test_init(self):
        """Test provider initialization."""
        with patch("loft.neural.providers.openai.OpenAI") as mock_openai:
            provider = OpenAIProvider(api_key="test-key", model="gpt-4")

            mock_openai.assert_called_once_with(api_key="test-key")
            assert provider.model == "gpt-4"
            assert provider.api_key == "test-key"

    def test_get_provider_name(self, provider):
        """Test getting provider name."""
        assert provider.get_provider_name() == "openai"

    def test_estimate_cost_gpt4_turbo(self, provider):
        """Test cost estimation for GPT-4 Turbo."""
        # gpt-4-turbo-preview: input=$10/M, output=$30/M
        cost = provider.estimate_cost(1000, 500)
        expected = (1000 / 1_000_000 * 10.00) + (500 / 1_000_000 * 30.00)
        assert cost == pytest.approx(expected)

    def test_estimate_cost_gpt35(self):
        """Test cost estimation for GPT-3.5."""
        with patch("loft.neural.providers.openai.OpenAI"):
            provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
            # gpt-3.5-turbo: input=$0.50/M, output=$1.50/M
            cost = provider.estimate_cost(1000, 500)
            expected = (1000 / 1_000_000 * 0.50) + (500 / 1_000_000 * 1.50)
            assert cost == pytest.approx(expected)

    def test_build_messages_basic(self, provider):
        """Test building messages from query."""
        query = LLMQuery(question="What is contract law?")
        messages = provider._build_messages(query)

        # Should have system + user message
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is contract law?"

    def test_build_messages_custom_system_prompt(self, provider):
        """Test building messages with custom system prompt."""
        query = LLMQuery(question="Is this valid?", system_prompt="You are a contract expert.")
        messages = provider._build_messages(query)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a contract expert."

    def test_build_messages_with_context(self, provider):
        """Test building messages with context."""
        query = LLMQuery(question="Is this enforceable?", context={"type": "sale", "state": "CA"})
        messages = provider._build_messages(query)

        user_message = messages[1]["content"]
        assert "Context:" in user_message
        assert "type: sale" in user_message
        assert "state: CA" in user_message

    def test_build_messages_with_cot(self, provider):
        """Test building messages with chain-of-thought."""
        query = LLMQuery(question="Is this valid?", cot_enabled=True)
        messages = provider._build_messages(query)

        assert "Let's think step-by-step:" in messages[-1]["content"]

    def test_get_default_system_prompt(self, provider):
        """Test default system prompt."""
        prompt = provider._get_default_system_prompt()
        assert "legal reasoning" in prompt.lower()
        assert "assistant" in prompt.lower()

    def test_query_unstructured_success(self, provider):
        """Test successful unstructured query."""
        # Mock response
        mock_message = Mock()
        mock_message.content = "The contract is enforceable."

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        provider.client.chat.completions.create = Mock(return_value=mock_response)

        query = LLMQuery(question="Is this enforceable?")
        response = provider.query(query)

        assert isinstance(response, LLMResponse)
        assert isinstance(response.content, DefaultResponse)
        assert response.content.response == "The contract is enforceable."
        assert response.raw_text == "The contract is enforceable."
        assert response.metadata.tokens_input == 100
        assert response.metadata.tokens_output == 50
        assert response.metadata.tokens_total == 150
        assert response.metadata.provider == "openai"

    def test_query_structured_success(self, provider):
        """Test successful structured query."""
        mock_structured = StructuredOutputSchema(answer="Valid", confidence=0.9)
        mock_structured._raw_response = Mock()
        mock_structured._raw_response.usage = Mock()
        mock_structured._raw_response.usage.prompt_tokens = 100
        mock_structured._raw_response.usage.completion_tokens = 50

        mock_instructor = MagicMock()
        mock_client = Mock()
        mock_client.chat.completions.create = Mock(return_value=mock_structured)
        mock_instructor.from_openai = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            query = LLMQuery(question="Is this valid?")
            response = provider.query(query, response_model=StructuredOutputSchema)

            assert isinstance(response.content, StructuredOutputSchema)
            assert response.content.answer == "Valid"
            assert response.confidence == 0.9

    def test_query_rate_limit_error(self, provider):
        """Test handling of rate limit errors."""
        from openai import RateLimitError

        mock_response = Mock()
        mock_response.status_code = 429
        error = RateLimitError("Rate limited", response=mock_response, body=None)
        provider.client.chat.completions.create = Mock(side_effect=error)

        query = LLMQuery(question="Test")
        with pytest.raises(LLMRateLimitError) as exc_info:
            provider.query(query)

        assert exc_info.value.provider == "openai"

    def test_query_timeout_error(self, provider):
        """Test handling of timeout errors."""
        import openai

        provider.client.chat.completions.create = Mock(
            side_effect=openai.APITimeoutError("Timeout")
        )

        query = LLMQuery(question="Test")
        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.query(query)

        assert exc_info.value.provider == "openai"

    def test_query_api_error(self, provider):
        """Test handling of general API errors."""
        from openai import APIError

        mock_request = Mock()
        error = APIError("API Error", request=mock_request, body=None)
        provider.client.chat.completions.create = Mock(side_effect=error)

        query = LLMQuery(question="Test")
        with pytest.raises(LLMProviderError) as exc_info:
            provider.query(query)

        assert exc_info.value.provider == "openai"


class TestLocalProvider:
    """Test LocalProvider functionality."""

    @pytest.fixture
    def provider(self):
        """Create LocalProvider."""
        provider = LocalProvider(
            api_key="not-needed", model="llama2", base_url="http://localhost:11434"
        )
        return provider

    def test_init_default_url(self):
        """Test initialization with default URL."""
        provider = LocalProvider(api_key="", model="llama2")
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama2"

    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        provider = LocalProvider(api_key="", model="llama2", base_url="http://custom:8080")
        assert provider.base_url == "http://custom:8080"

    def test_get_provider_name(self, provider):
        """Test getting provider name."""
        assert provider.get_provider_name() == "local"

    def test_estimate_cost(self, provider):
        """Test that local provider has zero cost."""
        cost = provider.estimate_cost(1000, 500)
        assert cost == 0.0

    def test_build_messages_basic(self, provider):
        """Test building messages from query."""
        query = LLMQuery(question="What is contract law?")
        messages = provider._build_messages(query)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is contract law?"

    def test_build_messages_with_context(self, provider):
        """Test building messages with context."""
        query = LLMQuery(question="Is this enforceable?", context={"type": "sale", "state": "CA"})
        messages = provider._build_messages(query)

        user_content = messages[1]["content"]
        assert "Context:" in user_content
        assert "type: sale" in user_content

    def test_get_default_system_prompt(self, provider):
        """Test default system prompt."""
        prompt = provider._get_default_system_prompt()
        assert "helpful" in prompt.lower()
        assert "legal" in prompt.lower()

    def test_query_unstructured_success(self, provider):
        """Test successful unstructured query."""
        mock_response_data = {
            "message": {"content": "The contract is enforceable."},
            "model": "llama2",
        }

        with patch("requests.post") as mock_post:
            mock_resp = Mock()
            mock_resp.json = Mock(return_value=mock_response_data)
            mock_resp.raise_for_status = Mock()
            mock_post.return_value = mock_resp

            query = LLMQuery(question="Is this enforceable?")
            response = provider.query(query)

            assert isinstance(response, LLMResponse)
            assert isinstance(response.content, DefaultResponse)
            assert response.content.response == "The contract is enforceable."
            assert response.raw_text == "The contract is enforceable."
            assert response.confidence == 0.6  # Lower for local
            assert response.metadata.provider == "local"
            assert response.metadata.cost_usd == 0.0

    def test_query_structured_success(self, provider):
        """Test successful structured query with JSON parsing."""
        mock_response_data = {
            "message": {"content": '{"answer": "Valid", "confidence": 0.85}'},
            "model": "llama2",
        }

        with patch("requests.post") as mock_post:
            mock_resp = Mock()
            mock_resp.json = Mock(return_value=mock_response_data)
            mock_resp.raise_for_status = Mock()
            mock_post.return_value = mock_resp

            query = LLMQuery(question="Is this valid?")
            response = provider.query(query, response_model=StructuredOutputSchema)

            assert isinstance(response.content, StructuredOutputSchema)
            assert response.content.answer == "Valid"
            assert response.content.confidence == 0.85

    def test_query_structured_with_markdown_json(self, provider):
        """Test structured query with JSON in markdown code block."""
        mock_response_data = {
            "message": {"content": '```json\n{"answer": "Valid", "confidence": 0.85}\n```'},
            "model": "llama2",
        }

        with patch("requests.post") as mock_post:
            mock_resp = Mock()
            mock_resp.json = Mock(return_value=mock_response_data)
            mock_resp.raise_for_status = Mock()
            mock_post.return_value = mock_resp

            query = LLMQuery(question="Is this valid?")
            response = provider.query(query, response_model=StructuredOutputSchema)

            assert isinstance(response.content, StructuredOutputSchema)
            assert response.content.answer == "Valid"

    def test_query_request_exception(self, provider):
        """Test handling of request exceptions."""
        import requests

        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Connection failed")

            query = LLMQuery(question="Test")
            with pytest.raises(LLMProviderError) as exc_info:
                provider.query(query)

            assert exc_info.value.provider == "local"

    def test_parse_json_response_invalid_json(self, provider):
        """Test parsing invalid JSON."""
        with pytest.raises(LLMParsingError):
            provider._parse_json_response("not json", StructuredOutputSchema)

    def test_parse_json_response_missing_field(self, provider):
        """Test parsing JSON with missing required field."""
        with pytest.raises(LLMParsingError):
            provider._parse_json_response('{"confidence": 0.8}', StructuredOutputSchema)


class TestDefaultResponse:
    """Test DefaultResponse model."""

    def test_creation(self):
        """Test creating DefaultResponse."""
        response = DefaultResponse(response="Test response")
        assert response.response == "Test response"
        assert response.confidence == 0.7

    def test_creation_with_custom_confidence(self):
        """Test creating DefaultResponse with custom confidence."""
        response = DefaultResponse(response="Test", confidence=0.9)
        assert response.response == "Test"
        assert response.confidence == 0.9
