# LOFT Scripts

Utility scripts for testing and demonstrating LOFT functionality.

## Interactive LLM Test Script

The `test_llm_interactive.py` script provides an interactive way to test the LLM interface with real API keys.

### Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to `.env`:**
   ```bash
   # Required: Anthropic API key
   ANTHROPIC_API_KEY=sk-ant-...

   # Optional: OpenAI API key (for multi-provider tests)
   OPENAI_API_KEY=sk-...
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

### Usage

**Run all tests with Anthropic (default):**
```bash
python scripts/test_llm_interactive.py
```

**Run all tests with OpenAI:**
```bash
python scripts/test_llm_interactive.py --provider openai
```

**Run specific test:**
```bash
# Available tests: basic, structured, template, cache, cost, extraction, cot, multi
python scripts/test_llm_interactive.py --test structured
```

**Compare multiple providers:**
```bash
python scripts/test_llm_interactive.py --test multi
```

### Tests Included

1. **Basic Query**: Simple unstructured question answering
2. **Structured Output**: Pydantic schema enforcement with legal analysis
3. **Prompt Templates**: Using pre-built templates for gap identification
4. **Response Caching**: Demonstrating cost savings through caching
5. **Cost Tracking**: Tracking costs across multiple queries
6. **Element Extraction**: Using templates for legal element analysis
7. **Chain-of-Thought**: Step-by-step legal reasoning
8. **Multi-Provider**: Comparing Anthropic, OpenAI, and local models

### Example Output

```
╭──────────────────────────────────────────────────────────────╮
│ LOFT LLM Interface - Interactive Test Suite                 │
│ Testing neural interface with real API keys                 │
╰──────────────────────────────────────────────────────────────╯

Provider: anthropic
Model: claude-3-5-sonnet-20241022
Caching: Enabled

Test 1: Basic Query
================================================================================

Question: What is the statute of frauds?

Structured Output:
  conclusion: The statute of frauds requires certain contracts to be in writing...
  confidence: 0.95
  reasoning: This is a well-established legal doctrine...

Metadata:
  Provider: anthropic
  Model: claude-3-5-sonnet-20241022
  Tokens: 245 (52 in / 193 out)
  Cost: $0.003045
  Latency: 1423ms
  Confidence: 0.95
  Cache hit: False
```

### Environment Variables

The script uses these environment variables from `.env`:

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required for Anthropic provider)
- `OPENAI_API_KEY`: Your OpenAI API key (optional, for OpenAI provider)
- `LLM_PROVIDER`: Default provider (anthropic or openai)
- `LLM_MODEL`: Default model to use
- `LLM_TEMPERATURE`: Sampling temperature (0.0-2.0)
- `LLM_MAX_TOKENS`: Maximum tokens in response

### Cost Estimates

Approximate costs per test run (using Claude 3.5 Sonnet):

- **Single test**: ~$0.01 - $0.05
- **All tests**: ~$0.10 - $0.30
- **With caching**: ~50% cost savings on repeated queries

Using Claude 3.5 Haiku or GPT-3.5 Turbo will significantly reduce costs.

### Local Testing (No Cost)

If you have Ollama running locally, the multi-provider test will automatically include it:

```bash
# Start Ollama (in another terminal)
ollama serve

# Pull a model
ollama pull llama2

# Run tests
python scripts/test_llm_interactive.py --test multi
```

### Troubleshooting

**"No module named 'loft'"**
```bash
# Install in development mode
pip install -e .
```

**"ANTHROPIC_API_KEY not set"**
```bash
# Check your .env file exists and has the key
cat .env | grep ANTHROPIC_API_KEY

# Or set directly in terminal
export ANTHROPIC_API_KEY='your-key'
```

**"Rate limit exceeded"**
- The script includes automatic retry logic with exponential backoff
- If tests fail repeatedly, you may have hit your API quota
- Consider using a cheaper model (Haiku instead of Sonnet)

### Security Notes

- **Never commit `.env` file to git** (already in `.gitignore`)
- API keys in `.env` are loaded via `python-dotenv`
- Keys are never logged or displayed in output
- Use separate API keys for development and production
