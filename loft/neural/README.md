# Neural: LLM Interface Layer

This module handles all interactions with Large Language Models (LLMs).

## Responsibilities

- **Multi-provider support** for Anthropic (Claude), OpenAI (GPT), and local models
- **Structured output generation** using Pydantic schemas and instructor library
- **Prompt template management** with versioning and tracking
- **Error handling and retries** with exponential backoff
- **Cost tracking** and optimization (model selection based on task complexity)
- **Logging** of all LLM interactions for analysis

## Key Components (to be implemented)

- `llm_interface.py` - Abstract interface for LLM providers
- `anthropic_client.py` - Anthropic/Claude implementation
- `openai_client.py` - OpenAI/GPT implementation
- `prompt_manager.py` - Template management and rendering
- `structured_output.py` - Schema-guided generation utilities

## Example Usage (planned)

```python
from loft.neural import LLMInterface
from loft.config import config

# Initialize LLM interface
llm = LLMInterface(provider=config.llm.provider)

# Query with structured output
from pydantic import BaseModel

class ContractAnalysis(BaseModel):
    is_enforceable: bool
    confidence: float
    reasoning: str

result = llm.query(
    prompt="Analyze this contract...",
    schema=ContractAnalysis
)
```

## Integration Points

- **Core** (`loft.core`): Receives queries from symbolic core
- **Translation** (`loft.translation`): Uses NL prompts, produces structured outputs
- **Config** (`loft.config`): API keys, model selection, parameters
