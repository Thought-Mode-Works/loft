"""
Examples for NL → ASP Translation

This file demonstrates the Natural Language to ASP translation layer,
showing various use cases for extracting ASP facts and rules from natural language.
"""

from loft.translation import (
    nl_to_asp_facts,
    nl_to_asp_rule,
    nl_to_structured,
    NLToASPTranslator,
    ASPToNLTranslator,
    ContractFact,
    roundtrip_fidelity_test,
    quick_extract_facts,
)


def example_1_simple_facts():
    """Example 1: Extract simple ASP facts from natural language."""
    print("=" * 70)
    print("Example 1: Simple Fact Extraction")
    print("=" * 70)

    nl_text = "c1 is a contract"
    print(f"Input: {nl_text}")

    facts = nl_to_asp_facts(nl_text)
    print("\nExtracted ASP Facts:")
    for fact in facts:
        print(f"  - {fact}")


def example_2_land_sale_contract():
    """Example 2: Extract land sale contract with details."""
    print("\n" + "=" * 70)
    print("Example 2: Land Sale Contract")
    print("=" * 70)

    nl_text = "This is a land sale contract for $500,000"
    print(f"Input: {nl_text}")

    facts = nl_to_asp_facts(nl_text)
    print("\nExtracted ASP Facts:")
    for fact in facts:
        print(f"  - {fact}")


def example_3_contract_with_parties():
    """Example 3: Extract contract with parties."""
    print("\n" + "=" * 70)
    print("Example 3: Contract Between Parties")
    print("=" * 70)

    nl_text = "Contract between Alice and Bob for land sale"
    print(f"Input: {nl_text}")

    facts = nl_to_asp_facts(nl_text)
    print("\nExtracted ASP Facts:")
    for fact in facts:
        print(f"  - {fact}")


def example_4_signed_contract():
    """Example 4: Extract signed contract information."""
    print("\n" + "=" * 70)
    print("Example 4: Signed Contract")
    print("=" * 70)

    nl_text = "The contract was signed by John and has writing"
    print(f"Input: {nl_text}")

    facts = nl_to_asp_facts(nl_text)
    print("\nExtracted ASP Facts:")
    for fact in facts:
        print(f"  - {fact}")


def example_5_structured_extraction():
    """Example 5: Use Pydantic schema for structured extraction."""
    print("\n" + "=" * 70)
    print("Example 5: Structured Extraction with Pydantic")
    print("=" * 70)

    nl_text = (
        "John and Mary have a land sale contract for $750,000 with a signed writing"
    )
    print(f"Input: {nl_text}")

    # Extract using ContractFact schema
    contract = nl_to_structured(nl_text, ContractFact)
    print("\nExtracted Contract:")
    print(f"  ID: {contract.contract_id}")
    print(f"  Type: {contract.contract_type}")
    print(f"  Parties: {contract.parties}")
    print(f"  Sale Amount: ${contract.sale_amount}")
    print(f"  Has Writing: {contract.has_writing}")
    print(f"  Is Signed: {contract.is_signed}")

    # Convert to ASP facts
    asp_facts = contract.to_asp()
    print("\nASP Facts:")
    for fact in asp_facts:
        print(f"  - {fact}")


def example_6_rule_extraction():
    """Example 6: Extract ASP rules from natural language."""
    print("\n" + "=" * 70)
    print("Example 6: Rule Extraction")
    print("=" * 70)

    nl_rule = "A contract satisfies statute of frauds if it has a signed writing"
    print(f"Input: {nl_rule}")

    asp_rule = nl_to_asp_rule(nl_rule)
    print("\nExtracted ASP Rule:")
    print(f"  {asp_rule}")

    print("\n" + "-" * 70)
    nl_rule2 = "A contract is enforceable unless proven unenforceable"
    print(f"Input: {nl_rule2}")

    asp_rule2 = nl_to_asp_rule(nl_rule2)
    print("\nExtracted ASP Rule:")
    print(f"  {asp_rule2}")


def example_7_translator_class():
    """Example 7: Use NLToASPTranslator class."""
    print("\n" + "=" * 70)
    print("Example 7: Using NLToASPTranslator Class")
    print("=" * 70)

    translator = NLToASPTranslator()

    nl_text = "c1 is a land sale contract with consideration"
    print(f"Input: {nl_text}")

    result = translator.translate_to_facts(nl_text)

    print("\nTranslation Result:")
    print(f"  Extraction Method: {result.extraction_method}")
    print(f"  Confidence: {result.confidence:.2f}")
    print("  ASP Facts:")
    for fact in result.asp_facts:
        print(f"    - {fact}")


def example_8_roundtrip_testing():
    """Example 8: Test roundtrip translation fidelity."""
    print("\n" + "=" * 70)
    print("Example 8: Roundtrip Translation Testing")
    print("=" * 70)

    asp_original = "contract(c1)."
    print(f"Original ASP: {asp_original}")

    # Create translators
    asp_to_nl = ASPToNLTranslator(domain="legal")
    nl_to_asp = NLToASPTranslator()

    # Test roundtrip
    fidelity, explanation = roundtrip_fidelity_test(asp_original, asp_to_nl, nl_to_asp)

    print(f"\nFidelity Score: {fidelity:.2%}")
    print(explanation)


def example_9_quick_extraction():
    """Example 9: Quick extraction using pattern matching only."""
    print("\n" + "=" * 70)
    print("Example 9: Quick Pattern-Based Extraction")
    print("=" * 70)

    nl_text = "The land sale contract has consideration and mutual assent"
    print(f"Input: {nl_text}")

    facts = quick_extract_facts(nl_text)
    print("\nQuickly Extracted Facts:")
    for fact in facts:
        print(f"  - {fact}")


def example_10_complex_scenario():
    """Example 10: Complex real-world scenario."""
    print("\n" + "=" * 70)
    print("Example 10: Complex Contract Scenario")
    print("=" * 70)

    nl_text = """
    Alice and Bob entered into a land sale contract for $1,000,000.
    The contract has a written document that was signed by both parties.
    The agreement includes consideration and shows mutual assent.
    """

    print(f"Input: {nl_text.strip()}")

    translator = NLToASPTranslator()
    result = translator.translate_to_facts(nl_text)

    print("\nExtracted ASP Facts:")
    for fact in result.asp_facts:
        print(f"  - {fact}")

    print(f"\nConfidence: {result.confidence:.2f}")
    print(f"Extraction Method: {result.extraction_method}")


def main():
    """Run all examples."""
    print("\n")
    print("#" * 70)
    print("# NL → ASP Translation Examples")
    print("#" * 70)

    example_1_simple_facts()
    example_2_land_sale_contract()
    example_3_contract_with_parties()
    example_4_signed_contract()
    example_5_structured_extraction()
    example_6_rule_extraction()
    example_7_translator_class()
    example_8_roundtrip_testing()
    example_9_quick_extraction()
    example_10_complex_scenario()

    print("\n" + "#" * 70)
    print("# All examples completed successfully!")
    print("#" * 70)


if __name__ == "__main__":
    main()
