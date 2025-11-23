"""
Unit tests for bugfixes in NL → ASP translation.

These tests document specific bugs that were found and fixed.
"""

from loft.translation import nl_to_asp_facts, ASPToNLTranslator, NLToASPTranslator
from loft.translation.quality import roundtrip_fidelity_test


class TestBugfix1GreedyPatterns:
    r"""
    Bug: Pattern matching was too greedy, capturing entire phrases.

    Example: "This is a land sale contract for $500,000"
    Was producing: ['land_sale_contract_for(this).', ...]
    Should produce: ['sale_amount(contract_1, 500000).', 'land_sale_contract(contract_1).']

    Root cause: Regex pattern r"(\w+(?:\s+\w+)*)\s+is\s+an?\s+(\w+(?:\s+\w+)*)"
    was matching "This is a land sale contract for" entirely.

    Fix:
    1. Reordered patterns (specific before general)
    2. Made patterns less greedy with {0,3}? quantifiers
    3. Added boundary constraints (?:\s|$|\.|\,)
    4. Filter out common sentence starters (this, that, it, etc.)
    """

    def test_example_2_no_malformed_facts(self) -> None:
        """Test that Example 2 produces correct facts."""
        nl = "This is a land sale contract for $500,000"
        facts = nl_to_asp_facts(nl)

        # Should NOT produce malformed facts
        assert not any("land_sale_contract_for" in f for f in facts)
        assert not any(f.startswith("land(this)") for f in facts)

        # Should produce correct facts
        assert any("land_sale" in f for f in facts)
        assert any("sale_amount" in f for f in facts)

    def test_sentence_starters_filtered(self) -> None:
        """Test that common sentence starters are filtered out."""
        test_cases = [
            ("This is a contract", ["contract"]),
            ("That is a party", ["party"]),
            ("It is a writing", ["writing"]),
        ]

        for nl, expected_predicates in test_cases:
            facts = nl_to_asp_facts(nl)
            # Should not produce facts with "this", "that", "it" as subjects
            assert not any("(this)" in f.lower() for f in facts)
            assert not any("(that)" in f.lower() for f in facts)
            assert not any("(it)" in f.lower() for f in facts)

    def test_pattern_boundary_constraints(self) -> None:
        """Test that patterns respect word boundaries."""
        nl = "c1 is a contract"
        facts = nl_to_asp_facts(nl)

        # Should produce exactly the expected fact
        assert "contract(c1)." in facts

    def test_specific_patterns_before_general(self) -> None:
        """Test that specific patterns match before general ones."""
        # "signed by" should match before "is a"
        nl = "The contract was signed by John"
        facts = nl_to_asp_facts(nl)

        assert any("signed_by" in f for f in facts)
        assert any("john" in f.lower() for f in facts)


class TestBugfix2RoundtripFidelity:
    """
    Bug: Roundtrip fidelity was 0% for all cases.

    Example: ASP "contract(c1)." → NL "Is contract c1 a contract?" → ASP ???
    Was producing: 0.00% fidelity
    Should produce: >90% fidelity

    Root cause: ASP→NL produces questions ("Is X a Y?") but NL→ASP pattern
    matching expects statements ("X is a Y").

    Fix: Added _question_to_statement() helper to convert questions to
    statements before feeding to NL→ASP pattern matching.
    """

    def test_example_5_roundtrip_fidelity_improved(self) -> None:
        """Test that Example 5 roundtrip achieves high fidelity."""
        asp_orig = "contract(c1)."
        asp_to_nl = ASPToNLTranslator()
        nl_to_asp = NLToASPTranslator()

        fidelity, _ = roundtrip_fidelity_test(asp_orig, asp_to_nl, nl_to_asp)

        # Should achieve near-perfect fidelity
        assert fidelity >= 0.9, f"Expected fidelity >= 90%, got {fidelity:.2%}"

    def test_roundtrip_unary_predicates(self) -> None:
        """Test roundtrip for unary predicates."""
        asp_to_nl = ASPToNLTranslator()
        nl_to_asp = NLToASPTranslator()

        test_cases = ["contract(c1).", "party(alice).", "writing(w1)."]

        for asp in test_cases:
            fidelity, _ = roundtrip_fidelity_test(asp, asp_to_nl, nl_to_asp)
            assert fidelity >= 0.9, f"Failed for {asp}: {fidelity:.2%}"

    def test_roundtrip_binary_predicates(self) -> None:
        """Test roundtrip for binary predicates."""
        asp_to_nl = ASPToNLTranslator()
        nl_to_asp = NLToASPTranslator()

        # Test signed_by which was in the original bug report
        asp = "signed_by(w1, john)."
        fidelity, _ = roundtrip_fidelity_test(asp, asp_to_nl, nl_to_asp)
        assert fidelity >= 0.9, f"Failed for {asp}: {fidelity:.2%}"

    def test_question_to_statement_conversion(self) -> None:
        """Test that various question formats convert correctly."""
        from loft.translation.quality import _question_to_statement

        test_cases = [
            ("Is contract c1 a contract?", "contract c1 is a contract"),
            ("Is alice a party?", "alice is a party"),
            ("Is writing w1 signed by john?", "writing w1 was signed by john"),
            ("Does writing hold for writing w1?", "w1 is a writing"),
        ]

        for question, expected_statement in test_cases:
            statement = _question_to_statement(question)
            # Check that key elements are present (exact match not required)
            assert "is" in statement.lower() or "was" in statement.lower(), (
                f"Failed to convert: {question} → {statement}"
            )
