"""
Legal question parser: Natural language to ASP queries.

Converts natural language legal questions into structured ASP queries
using LLM-based parsing.

Issue #272: Legal Question Answering Interface
"""

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from loft.neural.llm_interface import LLMQuery
from loft.neural.providers import AnthropicProvider
from loft.qa.prompts import (
    QUESTION_PARSING_SYSTEM_PROMPT,
    build_parsing_prompt,
)
from loft.qa.schemas import ASPQuery


class ParsedQuestion(BaseModel):
    """Structured output from LLM question parsing."""

    domain: str = Field(
        ..., description="Legal domain (contracts, torts, property, etc.)"
    )
    facts: list[str] = Field(
        ..., description="List of ASP facts extracted from question"
    )
    query: str = Field(..., description="ASP query goal to answer the question")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in parsing quality"
    )


class LegalQuestionParser:
    """
    Parse natural language legal questions into ASP queries.

    Uses LLM to convert questions into structured ASP facts and query goals.

    Example:
        parser = LegalQuestionParser()
        asp_query = parser.parse(
            "Is a contract valid without consideration?"
        )
        # Returns ASPQuery with facts and query goal
    """

    def __init__(
        self,
        llm_provider: Optional[AnthropicProvider] = None,
        temperature: float = 0.3,
    ):
        """
        Initialize question parser.

        Args:
            llm_provider: LLM provider for parsing (optional, uses default if None)
            temperature: LLM temperature for parsing (lower = more deterministic)
        """
        self.llm_provider = llm_provider
        self.temperature = temperature

    def parse(self, question: str, domain: Optional[str] = None) -> ASPQuery:
        """
        Parse a natural language question into an ASP query.

        Args:
            question: Natural language legal question
            domain: Optional domain hint (contracts, torts, etc.)

        Returns:
            ASPQuery with facts, query goal, and metadata

        Raises:
            ValueError: If question cannot be parsed
        """
        logger.info(f"Parsing question: {question[:100]}...")

        # Try LLM-based parsing first
        if self.llm_provider:
            try:
                return self._parse_with_llm(question, domain)
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}, falling back to rule-based")

        # Fallback to rule-based parsing
        return self._parse_rule_based(question, domain)

    def _parse_with_llm(
        self, question: str, domain_hint: Optional[str] = None
    ) -> ASPQuery:
        """
        Parse question using LLM with structured output.

        Args:
            question: Natural language question
            domain_hint: Optional domain hint

        Returns:
            Parsed ASPQuery
        """
        # Build parsing prompt
        prompt = build_parsing_prompt(question)

        # Create LLM query
        llm_query = LLMQuery(
            question=prompt,
            system_prompt=QUESTION_PARSING_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=1000,
            output_schema=ParsedQuestion,
        )

        # Query LLM
        response = self.llm_provider.query(llm_query, response_model=ParsedQuestion)

        # Extract parsed data
        parsed = response.content
        assert isinstance(parsed, ParsedQuestion)

        # Build ASP query
        asp_query = ASPQuery(
            facts=parsed.facts,
            query=parsed.query,
            domain=domain_hint or parsed.domain,
            original_question=question,
            confidence=parsed.confidence * response.confidence,
        )

        logger.info(
            f"LLM parsed to domain='{asp_query.domain}', query='{asp_query.query}'"
        )
        return asp_query

    def _parse_rule_based(
        self, question: str, domain_hint: Optional[str] = None
    ) -> ASPQuery:
        """
        Fallback rule-based parsing for simple questions.

        This is a simple baseline that handles common patterns.
        For production use, LLM parsing is recommended.

        Args:
            question: Natural language question
            domain_hint: Optional domain hint

        Returns:
            ASPQuery with basic parsing
        """
        logger.info("Using rule-based parsing (fallback)")

        # Infer domain from keywords
        domain = domain_hint or self._infer_domain(question)

        # Extract facts using simple patterns
        facts = self._extract_facts_simple(question, domain)

        # Generate query goal
        query = self._generate_query_simple(question, domain)

        return ASPQuery(
            facts=facts,
            query=query,
            domain=domain,
            original_question=question,
            confidence=0.6,  # Lower confidence for rule-based
        )

    def _infer_domain(self, question: str) -> str:
        """
        Infer legal domain from question keywords.

        Uses priority-based matching: more specific keywords take precedence.

        Args:
            question: Natural language question

        Returns:
            Inferred domain (contracts, torts, property, or general)
        """
        question_lower = question.lower()

        # Priority keywords that are highly specific to a domain
        priority_keywords = {
            "negligence": "torts",
            "tort": "torts",
            "liability": "torts",
            "contract": "contracts",
            "offer": "contracts",
            "acceptance": "contracts",
            "consideration": "contracts",
            "property": "property",
            "ownership": "property",
            "possession": "property",
            "crime": "criminal",
            "criminal": "criminal",
        }

        # Check priority keywords first
        for keyword, domain in priority_keywords.items():
            if keyword in question_lower:
                return domain

        # Fallback to general keyword matching
        from loft.qa.prompts import DOMAIN_HINTS

        for domain, hints in DOMAIN_HINTS.items():
            keywords = hints.get("keywords", [])
            if any(keyword.lower() in question_lower for keyword in keywords):
                return domain

        return "general"

    def _extract_facts_simple(self, question: str, domain: str) -> list[str]:
        """
        Extract ASP facts using simple pattern matching.

        Args:
            question: Natural language question
            domain: Legal domain

        Returns:
            List of ASP facts
        """
        facts = []
        question_lower = question.lower()

        # Handle negation patterns
        if "without" in question_lower or "no " in question_lower:
            # "without consideration" -> "not consideration(c1)."
            if "without consideration" in question_lower:
                facts.append("not consideration(c1).")
            if "no consideration" in question_lower:
                facts.append("not consideration(c1).")

        # Handle presence patterns
        if domain == "contracts":
            if "offer" in question_lower and "without offer" not in question_lower:
                facts.append("offer(c1).")
            if (
                "acceptance" in question_lower
                and "without acceptance" not in question_lower
            ):
                facts.append("acceptance(c1).")
            if (
                "consideration" in question_lower
                and "without consideration" not in question_lower
            ):
                facts.append("consideration(c1).")

        # Handle entity types
        if "minor" in question_lower:
            facts.append("party(c1, p1).")
            facts.append("minor(p1).")

        # Default fallback
        if not facts:
            facts.append("question(q1).")

        return facts

    def _generate_query_simple(self, question: str, domain: str) -> str:
        """
        Generate query goal from question.

        Args:
            question: Natural language question
            domain: Legal domain

        Returns:
            ASP query goal
        """
        question_lower = question.lower()

        # Common query patterns
        if "valid" in question_lower and domain == "contracts":
            return "valid_contract(c1)"
        if "enforceable" in question_lower and domain == "contracts":
            return "enforceable(c1)"
        if "voidable" in question_lower:
            return "voidable(c1)"
        if "negligence" in question_lower and domain == "torts":
            return "negligence(X)"
        if "liable" in question_lower or "liability" in question_lower:
            return "liable(X)"
        if "damages" in question_lower:
            return "can_recover_damages(X)"

        # Default fallback
        return "answer(X)"

    def parse_batch(self, questions: list[str]) -> list[ASPQuery]:
        """
        Parse multiple questions in batch.

        Args:
            questions: List of natural language questions

        Returns:
            List of parsed ASPQuery objects
        """
        return [self.parse(q) for q in questions]
