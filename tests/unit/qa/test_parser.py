"""
Unit tests for legal question parser.

Tests question parsing from natural language to ASP queries.

Issue #272: Legal Question Answering Interface
"""

import pytest

from loft.qa.question_parser import LegalQuestionParser
from loft.qa.schemas import ASPQuery


class TestLegalQuestionParser:
    """Unit tests for LegalQuestionParser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance without LLM (uses rule-based fallback)."""
        return LegalQuestionParser(llm_provider=None)

    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser is not None
        assert parser.llm_provider is None
        assert parser.temperature == 0.3

    def test_parse_simple_contract_question(self, parser):
        """Test parsing a simple contract question."""
        question = "Is a contract valid without consideration?"

        result = parser.parse(question)

        assert isinstance(result, ASPQuery)
        assert result.original_question == question
        assert result.domain in ["contracts", "general"]
        assert isinstance(result.facts, list)
        assert isinstance(result.query, str)
        assert 0.0 <= result.confidence <= 1.0

    def test_parse_contract_with_consideration(self, parser):
        """Test parsing contract question with consideration."""
        question = "Is a contract valid with offer, acceptance, and consideration?"

        result = parser.parse(question)

        assert result.domain in ["contracts", "general"]
        # Should extract consideration fact
        assert any("consideration" in f.lower() for f in result.facts)

    def test_parse_minor_contract_question(self, parser):
        """Test parsing question about minor contracts."""
        question = "Can a minor enter a contract?"

        result = parser.parse(question)

        assert result.domain in ["contracts", "general"]
        assert len(result.facts) > 0

    def test_parse_negligence_question(self, parser):
        """Test parsing negligence question."""
        question = "Is there negligence if duty and breach exist?"

        result = parser.parse(question)

        assert result.domain in ["torts", "general"]
        assert len(result.facts) > 0

    def test_parse_with_domain_hint(self, parser):
        """Test parsing with explicit domain hint."""
        question = "Is this enforceable?"
        domain = "contracts"

        result = parser.parse(question, domain=domain)

        assert result.domain == domain

    def test_infer_domain_contracts(self, parser):
        """Test domain inference for contracts questions."""
        domain = parser._infer_domain("Does this contract have consideration?")
        assert domain == "contracts"

    def test_infer_domain_torts(self, parser):
        """Test domain inference for torts questions."""
        domain = parser._infer_domain("Was there negligence in this case?")
        assert domain == "torts"

    def test_infer_domain_property(self, parser):
        """Test domain inference for property questions."""
        domain = parser._infer_domain("Who has ownership of the land?")
        assert domain == "property"

    def test_infer_domain_unknown(self, parser):
        """Test domain inference for unknown questions."""
        domain = parser._infer_domain("What is the answer to this question?")
        assert domain == "general"

    def test_extract_facts_without_consideration(self, parser):
        """Test fact extraction for negation patterns."""
        question = "Is a contract valid without consideration?"
        facts = parser._extract_facts_simple(question, "contracts")

        # Should have negated consideration fact
        assert any("not consideration" in f for f in facts)

    def test_extract_facts_with_minor(self, parser):
        """Test fact extraction for minor pattern."""
        question = "Can a minor sign a contract?"
        facts = parser._extract_facts_simple(question, "contracts")

        # Should extract minor-related facts
        assert any("minor" in f.lower() for f in facts)

    def test_generate_query_valid_contract(self, parser):
        """Test query generation for contract validity."""
        query = parser._generate_query_simple("Is the contract valid?", "contracts")
        assert "valid_contract" in query.lower()

    def test_generate_query_enforceable(self, parser):
        """Test query generation for enforceability."""
        query = parser._generate_query_simple("Is this enforceable?", "contracts")
        assert "enforceable" in query.lower()

    def test_generate_query_negligence(self, parser):
        """Test query generation for negligence."""
        query = parser._generate_query_simple("Was there negligence?", "torts")
        assert "negligence" in query.lower()

    def test_parse_batch(self, parser):
        """Test parsing multiple questions in batch."""
        questions = [
            "Is a contract valid?",
            "Was there negligence?",
            "Who owns the property?",
        ]

        results = parser.parse_batch(questions)

        assert len(results) == 3
        assert all(isinstance(r, ASPQuery) for r in results)
        assert all(r.original_question in questions for r in results)

    def test_asp_query_to_program(self, parser):
        """Test converting ASPQuery to program string."""
        result = parser.parse("Is a contract valid?")
        program = result.to_asp_program()

        assert isinstance(program, str)
        assert len(program) > 0
        # Should contain query predicate
        assert "?-" in program

    def test_asp_query_string_representation(self, parser):
        """Test ASPQuery string representation."""
        result = parser.parse("Is a contract valid?")
        str_repr = str(result)

        assert "ASPQuery" in str_repr
        assert "facts" in str_repr
        assert "query" in str_repr


class TestQuestionParserEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return LegalQuestionParser(llm_provider=None)

    def test_parse_empty_question(self, parser):
        """Test parsing empty question."""
        result = parser.parse("")
        assert isinstance(result, ASPQuery)
        assert result.original_question == ""

    def test_parse_very_long_question(self, parser):
        """Test parsing very long question."""
        question = "Is a contract valid? " * 100
        result = parser.parse(question)
        assert isinstance(result, ASPQuery)

    def test_parse_special_characters(self, parser):
        """Test parsing question with special characters."""
        question = "Is contract #123 valid? (yes/no)"
        result = parser.parse(question)
        assert isinstance(result, ASPQuery)

    def test_parse_multiple_domains_mixed(self, parser):
        """Test parsing question mixing multiple domains."""
        question = "Is there negligence in the contract breach?"
        result = parser.parse(question)
        # Should infer one domain (likely torts due to negligence keyword)
        assert result.domain in ["torts", "contracts", "general"]
