"""
LLM provider implementations.

Supports Anthropic, OpenAI, and local models (via Ollama).
"""

from __future__ import annotations

import time
from typing import Any, Optional, Type, TypeVar
from datetime import datetime
from dataclasses import replace
from pydantic import BaseModel
import anthropic
import openai
from loguru import logger

from .llm_interface import LLMProvider, LLMQuery, LLMResponse, ResponseMetadata
from .errors import LLMProviderError, LLMParsingError, LLMRateLimitError, LLMTimeoutError


T = TypeVar("T", bound=BaseModel)


# Default response model when none provided
class DefaultResponse(BaseModel):
    """Default response model for unstructured output."""

    response: str
    confidence: float = 0.7


class AnthropicProvider(LLMProvider):
    """
    Anthropic (Claude) provider with structured output support.

    Uses instructor library for JSON schema enforcement.
    """

    # Cost per million tokens (as of Jan 2025)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)

    def query(
        self,
        llm_query: LLMQuery,
        response_model: Optional[Type[T]] = None,
    ) -> LLMResponse:
        """Query Anthropic API with optional structured output."""
        start_time = time.time()

        try:
            # Build messages
            messages = self._build_messages(llm_query)
            system_prompt = llm_query.system_prompt or self._get_default_system_prompt()

            # If structured output requested, use JSON mode
            if response_model or llm_query.output_schema:
                schema = response_model or llm_query.output_schema
                assert schema is not None  # Type narrowing
                response = self._query_structured(messages, system_prompt, schema, llm_query)
            else:
                response = self._query_unstructured(messages, system_prompt, llm_query)

            latency_ms = (time.time() - start_time) * 1000
            updated_metadata = replace(response.metadata, latency_ms=latency_ms)
            return replace(response, metadata=updated_metadata)

        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(
                str(e),
                provider="anthropic",
                retry_after=getattr(e, "retry_after", None),
            )
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(str(e), provider="anthropic")
        except anthropic.APIError as e:
            raise LLMProviderError(
                str(e), provider="anthropic", status_code=getattr(e, "status_code", None)
            )

    def _build_messages(self, llm_query: LLMQuery) -> list[dict[str, str]]:
        """Build message list from query."""
        messages = []

        # Add few-shot examples if provided
        for example in llm_query.few_shot_examples:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": "..."})

        # Add context if provided
        context_str = ""
        if llm_query.context:
            context_items = [f"{k}: {v}" for k, v in llm_query.context.items()]
            context_str = "\n\nContext:\n" + "\n".join(context_items) + "\n\n"

        # Add main question
        question = context_str + llm_query.question
        if llm_query.cot_enabled:
            question += "\n\nLet's approach this step-by-step:"

        messages.append({"role": "user", "content": question})
        return messages

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return (
            "You are a legal reasoning assistant. Provide accurate, well-reasoned "
            "responses based on the information provided. When uncertain, express "
            "that uncertainty clearly."
        )

    def _query_structured(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        schema: Type[T],
        llm_query: LLMQuery,
    ) -> LLMResponse:
        """Query with structured output using JSON mode."""
        try:
            # Use instructor for structured outputs
            import instructor  # type: ignore[import-not-found]

            client = instructor.from_anthropic(self.client)

            response = client.messages.create(
                model=self.model,
                max_tokens=llm_query.max_tokens,
                temperature=llm_query.temperature,
                system=system_prompt,
                messages=messages,
                response_model=schema,
            )

            # Extract metadata
            # Note: instructor wraps the response, so we need to access usage differently
            usage = getattr(response, "_raw_response", None)
            if usage and hasattr(usage, "usage"):
                tokens_input = usage.usage.input_tokens
                tokens_output = usage.usage.output_tokens
            else:
                tokens_input = 0
                tokens_output = 0

            metadata = ResponseMetadata(
                model=self.model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_input + tokens_output,
                latency_ms=0.0,  # Set by caller
                cost_usd=self.estimate_cost(tokens_input, tokens_output),
                timestamp=datetime.utcnow().isoformat(),
                provider="anthropic",
            )

            # Extract confidence if available
            confidence = getattr(response, "confidence", 0.8)

            return LLMResponse(
                content=response,
                raw_text=str(response),
                confidence=confidence,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            raise LLMParsingError(str(e), raw_response=str(messages))

    def _query_unstructured(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        llm_query: LLMQuery,
    ) -> LLMResponse:
        """Query without structured output."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=llm_query.max_tokens,
            temperature=llm_query.temperature,
            system=system_prompt,
            messages=messages,  # type: ignore[arg-type]
        )

        # Extract text
        content_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                content_text += block.text

        metadata = ResponseMetadata(
            model=self.model,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            tokens_total=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=0.0,  # Set by caller
            cost_usd=self.estimate_cost(response.usage.input_tokens, response.usage.output_tokens),
            timestamp=datetime.utcnow().isoformat(),
            provider="anthropic",
        )

        # Wrap in default response model
        default_content = DefaultResponse(response=content_text)

        return LLMResponse(
            content=default_content,
            raw_text=content_text,
            confidence=0.7,
            metadata=metadata,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        pricing = self.PRICING.get(self.model, {"input": 3.00, "output": 15.00})
        cost = (input_tokens / 1_000_000 * pricing["input"]) + (
            output_tokens / 1_000_000 * pricing["output"]
        )
        return cost

    def get_provider_name(self) -> str:
        return "anthropic"


class OpenAIProvider(LLMProvider):
    """
    OpenAI (GPT-4, GPT-3.5) provider with structured output support.
    """

    # Cost per million tokens (as of Jan 2025)
    PRICING = {
        "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        self.client = openai.OpenAI(api_key=api_key)

    def query(
        self,
        llm_query: LLMQuery,
        response_model: Optional[Type[T]] = None,
    ) -> LLMResponse:
        """Query OpenAI API with optional structured output."""
        start_time = time.time()

        try:
            messages = self._build_messages(llm_query)

            # If structured output requested, use JSON mode
            if response_model or llm_query.output_schema:
                schema = response_model or llm_query.output_schema
                assert schema is not None  # Type narrowing
                response = self._query_structured(messages, schema, llm_query)
            else:
                response = self._query_unstructured(messages, llm_query)

            latency_ms = (time.time() - start_time) * 1000
            updated_metadata = replace(response.metadata, latency_ms=latency_ms)
            return replace(response, metadata=updated_metadata)

        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e), provider="openai")
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e), provider="openai")
        except openai.APIError as e:
            raise LLMProviderError(str(e), provider="openai")

    def _build_messages(self, llm_query: LLMQuery) -> list[dict[str, str]]:
        """Build message list from query."""
        messages = []

        # Add system prompt
        system_prompt = llm_query.system_prompt or self._get_default_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # Add few-shot examples
        for example in llm_query.few_shot_examples:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": "..."})

        # Add context
        context_str = ""
        if llm_query.context:
            context_items = [f"{k}: {v}" for k, v in llm_query.context.items()]
            context_str = "\n\nContext:\n" + "\n".join(context_items) + "\n\n"

        question = context_str + llm_query.question
        if llm_query.cot_enabled:
            question += "\n\nLet's think step-by-step:"

        messages.append({"role": "user", "content": question})
        return messages

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return (
            "You are a legal reasoning assistant. Provide accurate, well-reasoned "
            "responses based on the information provided. Express uncertainty when appropriate."
        )

    def _query_structured(
        self,
        messages: list[dict[str, str]],
        schema: Type[T],
        llm_query: LLMQuery,
    ) -> LLMResponse:
        """Query with structured output using instructor."""
        try:
            import instructor  # type: ignore[import-not-found]

            client = instructor.from_openai(self.client)

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=llm_query.max_tokens,
                temperature=llm_query.temperature,
                response_model=schema,
            )

            # Extract metadata
            usage = getattr(response, "_raw_response", None)
            if usage and hasattr(usage, "usage"):
                tokens_input = usage.usage.prompt_tokens
                tokens_output = usage.usage.completion_tokens
            else:
                tokens_input = 0
                tokens_output = 0

            metadata = ResponseMetadata(
                model=self.model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_input + tokens_output,
                latency_ms=0.0,
                cost_usd=self.estimate_cost(tokens_input, tokens_output),
                timestamp=datetime.utcnow().isoformat(),
                provider="openai",
            )

            confidence = getattr(response, "confidence", 0.8)

            return LLMResponse(
                content=response,
                raw_text=str(response),
                confidence=confidence,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            raise LLMParsingError(str(e), raw_response=str(messages))

    def _query_unstructured(
        self,
        messages: list[dict[str, str]],
        llm_query: LLMQuery,
    ) -> LLMResponse:
        """Query without structured output."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=llm_query.max_tokens,
            temperature=llm_query.temperature,
        )

        content_text = response.choices[0].message.content or ""

        usage = response.usage
        assert usage is not None  # Type narrowing

        metadata = ResponseMetadata(
            model=self.model,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            tokens_total=usage.total_tokens,
            latency_ms=0.0,
            cost_usd=self.estimate_cost(usage.prompt_tokens, usage.completion_tokens),
            timestamp=datetime.utcnow().isoformat(),
            provider="openai",
        )

        default_content = DefaultResponse(response=content_text)

        return LLMResponse(
            content=default_content,
            raw_text=content_text,
            confidence=0.7,
            metadata=metadata,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        pricing = self.PRICING.get(self.model, {"input": 10.00, "output": 30.00})
        cost = (input_tokens / 1_000_000 * pricing["input"]) + (
            output_tokens / 1_000_000 * pricing["output"]
        )
        return cost

    def get_provider_name(self) -> str:
        return "openai"


class LocalProvider(LLMProvider):
    """
    Local model provider using Ollama.

    For testing and cost savings with local models.
    """

    def __init__(
        self, api_key: str, model: str, base_url: str = "http://localhost:11434", **kwargs: Any
    ):
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url

    def query(
        self,
        llm_query: LLMQuery,
        response_model: Optional[Type[T]] = None,
    ) -> LLMResponse:
        """Query local Ollama API."""
        import requests

        start_time = time.time()

        try:
            messages = self._build_messages(llm_query)

            # Ollama API call
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": llm_query.temperature,
                        "num_predict": llm_query.max_tokens,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            content_text = data.get("message", {}).get("content", "")
            latency_ms = (time.time() - start_time) * 1000

            # Parse structured output if requested
            if response_model or llm_query.output_schema:
                schema = response_model or llm_query.output_schema
                assert schema is not None  # Type narrowing
                content = self._parse_json_response(content_text, schema)
            else:
                content = DefaultResponse(response=content_text)

            # Local models don't have token counts, estimate
            tokens_estimate = len(content_text.split())
            metadata = ResponseMetadata(
                model=self.model,
                tokens_input=tokens_estimate,
                tokens_output=tokens_estimate,
                tokens_total=tokens_estimate * 2,
                latency_ms=latency_ms,
                cost_usd=0.0,  # Free for local
                timestamp=datetime.utcnow().isoformat(),
                provider="local",
            )

            return LLMResponse(
                content=content,
                raw_text=content_text,
                confidence=0.6,  # Lower confidence for local models
                metadata=metadata,
            )

        except requests.RequestException as e:
            raise LLMProviderError(str(e), provider="local")

    def _build_messages(self, llm_query: LLMQuery) -> list[dict[str, str]]:
        """Build message list."""
        messages = []

        system_prompt = llm_query.system_prompt or self._get_default_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        context_str = ""
        if llm_query.context:
            context_items = [f"{k}: {v}" for k, v in llm_query.context.items()]
            context_str = "\n\nContext:\n" + "\n".join(context_items) + "\n\n"

        messages.append({"role": "user", "content": context_str + llm_query.question})
        return messages

    def _get_default_system_prompt(self) -> str:
        return "You are a helpful legal reasoning assistant."

    def _parse_json_response(self, text: str, schema: Type[T]) -> T:
        """Parse JSON response into Pydantic model."""
        import json

        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            return schema(**data)
        except Exception as e:
            raise LLMParsingError(f"Failed to parse JSON: {e}", raw_response=text)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local models are free."""
        return 0.0

    def get_provider_name(self) -> str:
        return "local"
