"""
Neural (LLM) interface for LOFT.

Provides multi-provider LLM support with structured outputs, prompt templates,
and cost tracking.

Example usage:
    >>> from loft.neural import LLMInterface, AnthropicProvider, LLMQuery
    >>> from loft.config import config
    >>>
    >>> # Create provider
    >>> provider = AnthropicProvider(
    ...     api_key=config.llm.api_key,
    ...     model=config.llm.model
    ... )
    >>>
    >>> # Create interface
    >>> interface = LLMInterface(provider)
    >>>
    >>> # Query with structured output
    >>> from pydantic import BaseModel
    >>> class Analysis(BaseModel):
    ...     conclusion: str
    ...     confidence: float
    >>>
    >>> response = interface.query(
    ...     question="Is this contract enforceable?",
    ...     context={"facts": "..."},
    ...     output_schema=Analysis
    ... )
    >>> print(response.content.conclusion)
"""

from .llm_interface import (
    LLMQuery,
    LLMResponse,
    ResponseMetadata,
    LLMProvider,
    LLMInterface,
)
from .providers import (
    AnthropicProvider,
    OpenAIProvider,
    LocalProvider,
    DefaultResponse,
)
from .prompts import (
    PromptTemplate,
    get_template,
    list_templates,
    TEMPLATE_REGISTRY,
    GAP_IDENTIFICATION,
    ELEMENT_EXTRACTION,
    RULE_PROPOSAL,
    EXPLANATION_GENERATION,
    CONFIDENCE_ASSESSMENT,
    CONSISTENCY_CHECK,
    COT_LEGAL_REASONING,
    ANALOGICAL_REASONING,
)
from .errors import (
    LLMError,
    LLMProviderError,
    LLMParsingError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMValidationError,
)

__all__ = [
    # Core interfaces
    "LLMQuery",
    "LLMResponse",
    "ResponseMetadata",
    "LLMProvider",
    "LLMInterface",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
    "LocalProvider",
    "DefaultResponse",
    # Prompts
    "PromptTemplate",
    "get_template",
    "list_templates",
    "TEMPLATE_REGISTRY",
    "GAP_IDENTIFICATION",
    "ELEMENT_EXTRACTION",
    "RULE_PROPOSAL",
    "EXPLANATION_GENERATION",
    "CONFIDENCE_ASSESSMENT",
    "CONSISTENCY_CHECK",
    "COT_LEGAL_REASONING",
    "ANALOGICAL_REASONING",
    # Errors
    "LLMError",
    "LLMProviderError",
    "LLMParsingError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMValidationError",
]
