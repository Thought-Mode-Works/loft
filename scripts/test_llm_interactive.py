#!/usr/bin/env python3
"""
Interactive test script for LLM interface.

Tests the neural (LLM) interface with environment-configured API keys.
Demonstrates structured outputs, prompt templates, cost tracking, and caching.

Usage:
    # Set API keys in environment
    export ANTHROPIC_API_KEY="your-key"
    export OPENAI_API_KEY="your-key"  # Optional

    # Run interactive tests
    python scripts/test_llm_interactive.py

    # Or run specific test
    python scripts/test_llm_interactive.py --test structured
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from loft.neural import (
    LLMInterface,
    AnthropicProvider,
    OpenAIProvider,
    LocalProvider,
    get_template,
    list_templates,
)

console = Console()


class LegalAnalysis(BaseModel):
    """Example structured output schema for legal analysis."""

    conclusion: str
    confidence: float
    reasoning: str
    missing_elements: list[str] = []


class ContractValidity(BaseModel):
    """Example schema for contract validity assessment."""

    is_valid: bool
    confidence: float
    issues: list[str] = []


def print_header(title: str) -> None:
    """Print a formatted header."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 80)


def print_response(response, show_metadata: bool = True) -> None:
    """Print LLM response in formatted output."""
    # Print content
    if hasattr(response.content, "model_dump"):
        # Structured output
        console.print("\n[bold green]Structured Output:[/bold green]")
        content_dict = response.content.model_dump()
        for key, value in content_dict.items():
            console.print(f"  [yellow]{key}:[/yellow] {value}")
    else:
        # Raw text
        console.print("\n[bold green]Response:[/bold green]")
        console.print(response.raw_text)

    if show_metadata:
        # Print metadata
        console.print("\n[bold blue]Metadata:[/bold blue]")
        console.print(f"  Provider: {response.metadata.provider}")
        console.print(f"  Model: {response.metadata.model}")
        console.print(
            f"  Tokens: {response.metadata.tokens_total} "
            f"({response.metadata.tokens_input} in / {response.metadata.tokens_output} out)"
        )
        console.print(f"  Cost: ${response.metadata.cost_usd:.6f}")
        console.print(f"  Latency: {response.metadata.latency_ms:.0f}ms")
        console.print(f"  Confidence: {response.confidence:.2f}")
        console.print(f"  Cache hit: {response.metadata.cache_hit}")


def test_basic_query(interface: LLMInterface) -> None:
    """Test 1: Basic unstructured query."""
    print_header("Test 1: Basic Query")

    question = "What is the statute of frauds?"
    console.print(f"\n[bold]Question:[/bold] {question}")

    response = interface.query(question, temperature=0.7)
    print_response(response)


def test_structured_output(interface: LLMInterface) -> None:
    """Test 2: Structured output with Pydantic schema."""
    print_header("Test 2: Structured Output")

    question = """
    Analyze this scenario under the statute of frauds:

    Alice verbally agreed to sell her land to Bob for $200,000.
    They shook hands but nothing was put in writing.
    """

    console.print(f"\n[bold]Question:[/bold] {question.strip()}")
    console.print(f"\n[bold]Output Schema:[/bold] {LegalAnalysis.__name__}")

    response = interface.query(question=question, output_schema=LegalAnalysis, temperature=0.3)

    print_response(response)


def test_prompt_template(interface: LLMInterface) -> None:
    """Test 3: Using prompt templates."""
    print_header("Test 3: Prompt Templates")

    # List available templates
    templates = list_templates()
    console.print(f"\n[bold]Available Templates:[/bold] {', '.join(templates)}")

    # Use gap identification template
    template = get_template("gap_identification")

    console.print(f"\n[bold]Using Template:[/bold] {template.name} v{template.version}")
    console.print(f"[italic]{template.description}[/italic]")

    prompt = template.render(
        {
            "question": "Is the contract enforceable?",
            "context": "Oral agreement for sale of goods worth $1,000. No written contract exists.",
        }
    )

    console.print(f"\n[bold]Rendered Prompt:[/bold]\n{prompt[:200]}...")

    response = interface.query(
        question=prompt, system_prompt=template.system_prompt, temperature=0.5
    )

    print_response(response)


def test_caching(interface: LLMInterface) -> None:
    """Test 4: Response caching."""
    print_header("Test 4: Response Caching")

    question = "What are the elements of a valid contract?"

    # First query
    console.print(f"\n[bold]First Query:[/bold] {question}")
    response1 = interface.query(question)
    cost1 = response1.metadata.cost_usd
    cache_hit1 = response1.metadata.cache_hit

    console.print(f"  Cost: ${cost1:.6f}")
    console.print(f"  Cache hit: {cache_hit1}")

    # Second identical query (should be cached)
    console.print(f"\n[bold]Second Identical Query:[/bold] {question}")
    response2 = interface.query(question)
    cost2 = response2.metadata.cost_usd
    cache_hit2 = response2.metadata.cache_hit

    console.print(f"  Cost: ${cost2:.6f}")
    console.print(f"  Cache hit: {cache_hit2}")

    # Show savings
    total_cost = interface.get_total_cost()
    console.print(f"\n[bold green]Total Cost:[/bold green] ${total_cost:.6f}")
    console.print(
        f"[bold green]Savings from cache:[/bold green] ${cost1:.6f} "
        f"(would be ${cost1 + cost2:.6f} without caching)"
    )


def test_cost_tracking(interface: LLMInterface) -> None:
    """Test 5: Cost tracking across multiple queries."""
    print_header("Test 5: Cost Tracking")

    questions = [
        "What is consideration in contract law?",
        "What is mutual assent?",
        "What is legal capacity?",
    ]

    for i, question in enumerate(questions, 1):
        console.print(f"\n[bold]Query {i}:[/bold] {question}")
        response = interface.query(question, temperature=0.7)
        console.print(f"  Cost: ${response.metadata.cost_usd:.6f}")
        console.print(f"  Tokens: {response.metadata.tokens_total}")

    # Show totals
    console.print(f"\n[bold green]Total Cost:[/bold green] ${interface.get_total_cost():.6f}")
    console.print(f"[bold green]Total Tokens:[/bold green] {interface.get_total_tokens()}")


def test_element_extraction(interface: LLMInterface) -> None:
    """Test 6: Element extraction template."""
    print_header("Test 6: Element Extraction Template")

    template = get_template("element_extraction")

    prompt = template.render(
        {
            "fact_pattern": """
        On June 1, 2024, Alice (age 25) and Bob (age 30) signed a written agreement.
        Alice agreed to sell her car to Bob for $15,000. Bob paid a $500 deposit.
        The car was to be delivered on July 1, 2024.
        """,
            "doctrine": "Contract Formation",
        }
    )

    console.print(f"\n[bold]Template:[/bold] {template.name}")
    console.print(f"\n[bold]Rendered Prompt (excerpt):[/bold]\n{prompt[:300]}...")

    response = interface.query(
        question=prompt, system_prompt=template.system_prompt, temperature=0.3
    )

    print_response(response)


def test_chain_of_thought(interface: LLMInterface) -> None:
    """Test 7: Chain-of-thought reasoning."""
    print_header("Test 7: Chain-of-Thought Reasoning")

    template = get_template("cot_legal_reasoning")

    prompt = template.render(
        {
            "question": "Is the contract enforceable against Bob?",
            "facts": """
        Bob (age 17) entered into a contract with Seller to purchase a motorcycle
        for $5,000. Bob signed the contract and paid $1,000 deposit. Before delivery,
        Bob changed his mind and wants a refund.
        """,
            "law": """
        Minors (under 18) lack legal capacity to enter binding contracts.
        Contracts with minors are voidable at the minor's option.
        """,
        }
    )

    console.print(f"\n[bold]Template:[/bold] {template.name}")

    response = interface.query(
        question=prompt, system_prompt=template.system_prompt, temperature=0.5, cot_enabled=True
    )

    print_response(response)


def test_multiple_providers() -> None:
    """Test 8: Compare multiple providers."""
    print_header("Test 8: Multiple Providers")

    question = "In one sentence, what is a contract?"

    providers_to_test = []

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        anthropic_model = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
        providers_to_test.append(
            ("Anthropic", AnthropicProvider(api_key=anthropic_key, model=anthropic_model))
        )

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        providers_to_test.append(("OpenAI", OpenAIProvider(api_key=openai_key, model=openai_model)))

    # Local (if available)
    try:
        import requests

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            providers_to_test.append(("Local (Ollama)", LocalProvider(api_key="", model="llama2")))
    except Exception:
        pass

    if not providers_to_test:
        console.print("[red]No providers available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.[/red]")
        return

    # Query each provider
    results = []
    for name, provider in providers_to_test:
        console.print(f"\n[bold]Testing {name}...[/bold]")
        interface = LLMInterface(provider, enable_cache=False)
        try:
            response = interface.query(question, temperature=0.7, max_tokens=100)
            results.append(
                {
                    "provider": name,
                    "model": response.metadata.model,
                    "response": response.raw_text[:100] + "..."
                    if len(response.raw_text) > 100
                    else response.raw_text,
                    "cost": response.metadata.cost_usd,
                    "tokens": response.metadata.tokens_total,
                    "latency": response.metadata.latency_ms,
                }
            )
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    # Display comparison table
    if results:
        console.print("\n[bold]Comparison:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Cost")
        table.add_column("Tokens")
        table.add_column("Latency")

        for result in results:
            table.add_row(
                result["provider"],
                result["model"],
                f"${result['cost']:.6f}",
                str(result["tokens"]),
                f"{result['latency']:.0f}ms",
            )

        console.print(table)


def run_all_tests(provider_name: str = "anthropic") -> None:
    """Run all interactive tests."""
    console.print(
        Panel.fit(
            "[bold cyan]LOFT LLM Interface - Interactive Test Suite[/bold cyan]\n"
            "Testing neural interface with real API keys",
            border_style="cyan",
        )
    )

    # Get API key from environment
    if provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red]Error: ANTHROPIC_API_KEY not set in environment[/red]")
            console.print("Set it with: export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)

        model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        provider = AnthropicProvider(api_key=api_key, model=model)
    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY not set in environment[/red]")
            console.print("Set it with: export OPENAI_API_KEY='your-key'")
            sys.exit(1)

        model = os.getenv("LLM_MODEL", "gpt-4")
        provider = OpenAIProvider(api_key=api_key, model=model)
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        sys.exit(1)

    # Create interface with caching enabled
    interface = LLMInterface(provider, enable_cache=True)

    console.print(f"\n[bold]Provider:[/bold] {provider.get_provider_name()}")
    console.print(f"[bold]Model:[/bold] {provider.model}")
    console.print("[bold]Caching:[/bold] Enabled")

    # Run tests
    try:
        test_basic_query(interface)

        test_structured_output(interface)

        test_prompt_template(interface)

        test_caching(interface)

        test_cost_tracking(interface)

        test_element_extraction(interface)

        test_chain_of_thought(interface)

        # Multiple providers test (standalone)
        test_multiple_providers()

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Tests interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n\n[red]Error during tests: {e}[/red]")
        import traceback

        traceback.print_exc()

    # Final summary
    print_header("Test Summary")
    console.print(f"\n[bold green]Total Cost:[/bold green] ${interface.get_total_cost():.6f}")
    console.print(f"[bold green]Total Tokens:[/bold green] {interface.get_total_tokens()}")
    console.print(f"[bold green]Queries Made:[/bold green] {len(interface._cache)} unique")

    console.print("\n[bold cyan]Tests completed successfully![/bold cyan]")


def main():
    """Main entry point."""
    import argparse

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Interactive LLM interface tests")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--test",
        choices=[
            "basic",
            "structured",
            "template",
            "cache",
            "cost",
            "extraction",
            "cot",
            "multi",
            "all",
        ],
        default="all",
        help="Specific test to run (default: all)",
    )

    args = parser.parse_args()

    if args.test == "all":
        run_all_tests(args.provider)
    elif args.test == "multi":
        test_multiple_providers()
    else:
        # Run single test
        api_key = os.getenv(
            "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        )
        if not api_key:
            console.print(f"[red]Error: API key not set for {args.provider}[/red]")
            sys.exit(1)

        model = os.getenv(
            "LLM_MODEL", "claude-3-5-sonnet-20241022" if args.provider == "anthropic" else "gpt-4"
        )
        if args.provider == "anthropic":
            provider = AnthropicProvider(api_key=api_key, model=model)
        else:
            provider = OpenAIProvider(api_key=api_key, model=model)

        interface = LLMInterface(provider, enable_cache=True)

        test_map = {
            "basic": test_basic_query,
            "structured": test_structured_output,
            "template": test_prompt_template,
            "cache": test_caching,
            "cost": test_cost_tracking,
            "extraction": test_element_extraction,
            "cot": test_chain_of_thought,
        }

        test_map[args.test](interface)


if __name__ == "__main__":
    main()
