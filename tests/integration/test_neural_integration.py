"""
Integration tests for neural (LLM) interface.

Tests real workflows with mocked API responses.
"""

from unittest.mock import Mock, patch
from pydantic import BaseModel
from datetime import datetime

from loft.neural import (
    LLMInterface,
    AnthropicProvider,
    OpenAIProvider,
    LocalProvider,
    get_template,
    DefaultResponse,
    LLMResponse,
    ResponseMetadata,
)


class TestAnthropicIntegration:
    """Integration tests with Anthropic provider."""

    @patch("anthropic.Anthropic")
    def test_anthropic_cost_estimation(self, mock_anthropic_client: Mock) -> None:
        """Test cost estimation for Anthropic."""

        # Create provider
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
        )

        # Test cost estimation
        cost = provider.estimate_cost(1_000_000, 500_000)
        # 1M input @ $3/M + 500K output @ $15/M  = $3 + $7.50 = $10.50
        expected = (1_000_000 / 1_000_000 * 3.0) + (500_000 / 1_000_000 * 15.0)
        assert cost == expected

    @patch("anthropic.Anthropic")
    def test_anthropic_unstructured_output(self, mock_anthropic_client: Mock) -> None:
        """Test unstructured output with Anthropic."""

        # Mock client and response
        mock_client = Mock()
        mock_anthropic_client.return_value = mock_client

        # Mock message response
        mock_message = Mock()
        mock_content_block = Mock()
        mock_content_block.text = "This is the response text"
        mock_message.content = [mock_content_block]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 50

        mock_client.messages.create.return_value = mock_message

        # Create provider and query
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
        )

        from loft.neural.llm_interface import LLMQuery

        query = LLMQuery(question="What is a contract?")
        response = provider._query_unstructured(
            messages=[{"role": "user", "content": "What is a contract?"}],
            system_prompt="You are helpful",
            llm_query=query,
        )

        assert isinstance(response.content, DefaultResponse)
        assert "response text" in response.raw_text
        assert response.metadata.tokens_input == 100
        assert response.metadata.tokens_output == 50


class TestOpenAIIntegration:
    """Integration tests with OpenAI provider."""

    @patch("openai.OpenAI")
    def test_openai_cost_estimation(self, mock_openai_client: Mock) -> None:
        """Test cost estimation for OpenAI."""

        # Create provider
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4",
        )

        # Test cost estimation
        cost = provider.estimate_cost(1_000_000, 500_000)
        # 1M input @ $30/M + 500K output @ $60/M = $30 + $30 = $60
        expected = (1_000_000 / 1_000_000 * 30.0) + (500_000 / 1_000_000 * 60.0)
        assert cost == expected


class TestLocalProviderIntegration:
    """Integration tests with local (Ollama) provider."""

    @patch("requests.post")
    def test_local_unstructured_output(self, mock_post: Mock) -> None:
        """Test local provider with unstructured output."""

        # Mock Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Local model response"},
            "done": True,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Create provider and query
        provider = LocalProvider(
            api_key="",
            model="llama2",
            base_url="http://localhost:11434",
        )

        from loft.neural.llm_interface import LLMQuery

        query = LLMQuery(question="Test question")
        response = provider.query(query)

        assert isinstance(response.content, DefaultResponse)
        assert "Local model response" in response.raw_text
        assert response.metadata.cost_usd == 0.0
        assert response.metadata.provider == "local"

    @patch("requests.post")
    def test_local_structured_output(self, mock_post: Mock) -> None:
        """Test local provider with structured JSON output."""

        class SimpleResponse(BaseModel):
            answer: str
            confidence: float

        # Mock Ollama response with JSON
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": '```json\n{"answer": "Yes", "confidence": 0.7}\n```'
            },
            "done": True,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Create provider and query
        provider = LocalProvider(api_key="", model="llama2")

        from loft.neural.llm_interface import LLMQuery

        query = LLMQuery(question="Is this valid?", output_schema=SimpleResponse)
        response = provider.query(query, response_model=SimpleResponse)

        assert isinstance(response.content, SimpleResponse)
        assert response.content.answer == "Yes"
        assert response.content.confidence == 0.7


class TestPromptTemplateIntegration:
    """Integration tests using prompt templates."""

    def test_gap_identification_template(self) -> None:
        """Test using gap identification template."""
        template = get_template("gap_identification")

        # Render template
        prompt = template.render(
            {
                "question": "Is the contract enforceable?",
                "context": "Contract for land sale without writing",
            }
        )

        assert "Is the contract enforceable?" in prompt
        assert "land sale" in prompt
        assert "missing" in prompt.lower()

    def test_element_extraction_template(self) -> None:
        """Test element extraction template."""
        template = get_template("element_extraction")

        prompt = template.render(
            {
                "fact_pattern": "Alice sold her car to Bob for $5,000",
                "doctrine": "Contract Formation",
            }
        )

        assert "Alice sold her car" in prompt
        assert "Contract Formation" in prompt
        assert len(template.few_shot_examples) > 0

    def test_rule_proposal_template(self) -> None:
        """Test rule proposal template."""
        template = get_template("rule_proposal")

        prompt = template.render(
            {
                "scenario": "Online contract with electronic signature",
                "existing_rules": "rule1.\nrule2.\n",
            }
        )

        assert "electronic signature" in prompt
        assert "ASP" in prompt


class TestCostTracking:
    """Integration tests for cost tracking."""

    def test_multi_query_cost_tracking(self) -> None:
        """Test cost tracking across multiple queries."""
        provider = Mock()

        # Create interface
        interface = LLMInterface(provider, enable_cache=False)

        # Mock responses with different costs
        responses = []
        for i in range(5):
            metadata = ResponseMetadata(
                model="claude-3-5-sonnet-20241022",
                tokens_input=100 * (i + 1),
                tokens_output=50 * (i + 1),
                tokens_total=150 * (i + 1),
                latency_ms=1000.0,
                cost_usd=0.001 * (i + 1),
                timestamp=datetime.utcnow().isoformat(),
                provider="anthropic",
            )

            response = LLMResponse(
                content=DefaultResponse(response=f"Response {i}"),
                raw_text=f"Response {i}",
                confidence=0.8,
                metadata=metadata,
            )
            responses.append(response)

        provider.query.side_effect = responses

        # Make 5 queries
        for i in range(5):
            interface.query(f"Question {i}")

        # Check totals
        total_cost = interface.get_total_cost()
        # 0.001 + 0.002 + 0.003 + 0.004 + 0.005 = 0.015
        assert total_cost == 0.015

        total_tokens = interface.get_total_tokens()
        # 150 + 300 + 450 + 600 + 750 = 2250
        assert total_tokens == 2250


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_error_exception_types(self) -> None:
        """Test that error types are defined correctly."""
        from loft.neural.errors import (
            LLMError,
            LLMProviderError,
            LLMRateLimitError,
        )

        # Test inheritance
        error = LLMProviderError("test", provider="test", status_code=500)
        assert isinstance(error, LLMError)

        rate_limit = LLMRateLimitError("test", provider="test", retry_after=60)
        assert isinstance(rate_limit, LLMProviderError)
        assert rate_limit.retry_after == 60


class TestCaching:
    """Integration tests for response caching."""

    def test_cache_hit_reduces_cost(self) -> None:
        """Test that cache hits don't incur additional cost."""
        provider = Mock()
        interface = LLMInterface(provider, enable_cache=True)

        # Mock a single response
        metadata = ResponseMetadata(
            model="claude-3-5-sonnet-20241022",
            tokens_input=100,
            tokens_output=50,
            tokens_total=150,
            latency_ms=1000.0,
            cost_usd=0.001,
            timestamp=datetime.utcnow().isoformat(),
            provider="anthropic",
        )

        response = LLMResponse(
            content=DefaultResponse(response="Answer"),
            raw_text="Answer",
            confidence=0.9,
            metadata=metadata,
        )

        provider.query.return_value = response

        # First query - incurs cost
        interface.query("What is X?")
        assert interface.get_total_cost() == 0.001

        # Second identical query - should be cached, no additional cost
        interface.query("What is X?")
        assert interface.get_total_cost() == 0.001  # Still 0.001, not 0.002

        # Different query - incurs cost
        interface.query("What is Y?")
        assert interface.get_total_cost() == 0.002


class TestCompleteWorkflow:
    """End-to-end integration tests."""

    def test_legal_analysis_workflow(self) -> None:
        """Test complete legal analysis workflow."""

        # Define analysis schema
        class ContractAnalysis(BaseModel):
            is_enforceable: bool
            confidence: float
            reasoning: str
            missing_elements: list[str]

        # Mock provider
        provider = Mock()

        # Create mock response
        analysis = ContractAnalysis(
            is_enforceable=False,
            confidence=0.85,
            reasoning="Missing signed writing for land sale",
            missing_elements=["signed_writing"],
        )

        metadata = ResponseMetadata(
            model="claude-3-5-sonnet-20241022",
            tokens_input=200,
            tokens_output=100,
            tokens_total=300,
            latency_ms=1500.0,
            cost_usd=0.002,
            timestamp=datetime.utcnow().isoformat(),
            provider="anthropic",
        )

        mock_response = LLMResponse(
            content=analysis,
            raw_text="Analysis text",
            confidence=0.85,
            metadata=metadata,
        )

        provider.query.return_value = mock_response

        # Create interface
        interface = LLMInterface(provider)

        # Use template
        template = get_template("element_extraction")
        prompt = template.render(
            {
                "fact_pattern": "Oral agreement to sell land for $100,000",
                "doctrine": "Statute of Frauds",
            }
        )

        # Query with structured output
        response = interface.query(
            question=prompt,
            context={"domain": "contracts"},
            output_schema=ContractAnalysis,
            temperature=0.3,
        )

        # Verify results
        assert isinstance(response.content, ContractAnalysis)
        assert response.content.is_enforceable is False
        assert "signed_writing" in response.content.missing_elements
        assert response.metadata.cost_usd == 0.002
