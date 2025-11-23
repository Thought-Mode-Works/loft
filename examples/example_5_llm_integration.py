"""
Example 5: Context Enrichment for LLM Integration

This example demonstrates how to use the translation layer with the LLM interface
to query about ASP predicates using natural language.
"""

from loft.translation import enrich_context
from loft.symbolic import ASPCore
from loft.neural import LLMInterface, AnthropicProvider
from loft.config import config


def main():
    """Run the LLM integration example."""

    # Load ASP core with rules
    print("Loading ASP core...")
    asp_core = ASPCore()
    asp_core.load_rules()

    # Enrich query with relevant rules
    query = "enforceable(c1)?"
    print(f"\nOriginal ASP query: {query}")

    enriched_query = enrich_context(query, asp_core)
    print("\n=== Enriched Natural Language Query ===")
    print(enriched_query)
    print()

    # Check if API key is configured
    if not config.llm.api_key:
        print("⚠️  No API key configured. Set ANTHROPIC_API_KEY in .env file.")
        return

    # Setup LLM interface
    print("Setting up LLM interface...")
    provider = AnthropicProvider(
        api_key=config.llm.api_key,
        model=config.llm.model
    )
    llm_interface = LLMInterface(provider)

    # Query LLM with enriched context
    print("Querying LLM...\n")
    response = llm_interface.query(
        question=enriched_query,
        context={"domain": "contract_law"},
        max_tokens=500
    )

    print("=== LLM Response ===")
    print(response.raw_text)
    print()

    # Show metadata
    print("=== Response Metadata ===")
    print(f"Model: {response.metadata.model}")
    print(f"Tokens (input/output/total): {response.metadata.tokens_input}/{response.metadata.tokens_output}/{response.metadata.tokens_total}")
    print(f"Cost: ${response.metadata.cost_usd:.4f}")
    print(f"Latency: {response.metadata.latency_ms:.0f}ms")
    print(f"Confidence: {response.confidence}")


if __name__ == "__main__":
    main()
