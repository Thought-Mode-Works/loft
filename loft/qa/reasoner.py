"""
Legal reasoner using ASP core and knowledge database.

Answers legal questions by retrieving relevant rules from the knowledge database
and reasoning with the ASP core.

Issue #272: Legal Question Answering Interface
"""

from typing import List, Optional

import clingo
from loguru import logger

from loft.knowledge.database import KnowledgeDatabase
from loft.symbolic.asp_core import ASPCore, QueryResult
from loft.qa.schemas import Answer, ASPQuery


class LegalReasoner:
    """
    Answer legal questions using ASP reasoning and accumulated rules.

    Combines rule retrieval from knowledge database with ASP-based reasoning
    to answer questions and generate explanations.

    Example:
        reasoner = LegalReasoner(knowledge_db=db)
        asp_query = ASPQuery(
            facts=["offer(c1).", "acceptance(c1).", "not consideration(c1)."],
            query="valid_contract(c1)"
        )
        answer = reasoner.answer(asp_query)
    """

    def __init__(
        self,
        knowledge_db: Optional[KnowledgeDatabase] = None,
        asp_core: Optional[ASPCore] = None,
        min_rule_confidence: float = 0.6,
    ):
        """
        Initialize legal reasoner.

        Args:
            knowledge_db: Knowledge database for rule retrieval
            asp_core: ASP core for reasoning (creates new if None)
            min_rule_confidence: Minimum confidence for retrieved rules
        """
        self.knowledge_db = knowledge_db
        self.asp_core = asp_core
        self.min_rule_confidence = min_rule_confidence

    def answer(
        self,
        asp_query: ASPQuery,
        domain: Optional[str] = None,
        max_rules: int = 100,
    ) -> Answer:
        """
        Answer a question using ASP reasoning.

        Process:
        1. Retrieve relevant rules from database
        2. Load rules and facts into ASP core
        3. Query ASP core
        4. Generate explanation

        Args:
            asp_query: Parsed ASP query
            domain: Optional domain filter for rules
            max_rules: Maximum rules to retrieve

        Returns:
            Answer with result, explanation, and citations
        """
        logger.info(f"Answering query: {asp_query.query}")

        # Retrieve relevant rules
        rules_used = []
        rules_text = []

        if self.knowledge_db:
            rules = self.knowledge_db.search_rules(
                domain=domain or asp_query.domain,
                is_active=True,
                min_confidence=self.min_rule_confidence,
                limit=max_rules,
            )
            logger.info(f"Retrieved {len(rules)} rules from database")
            rules_used = [r.rule_id for r in rules]
            rules_text = [r.asp_rule for r in rules]
        else:
            logger.warning("No knowledge database provided, using facts only")

        # Build ASP program with rules and facts
        asp_program_text = "\n".join(rules_text)
        if asp_query.facts:
            asp_program_text += "\n\n% Query facts\n" + "\n".join(asp_query.facts)

        # Create ASP core and load program
        asp_core = ASPCore()
        asp_core.control = clingo.Control()

        try:
            asp_core.control.add("base", [], asp_program_text)
            asp_core.control.ground([("base", [])])
            asp_core._loaded = True
        except RuntimeError as e:
            logger.error(f"Failed to load/ground ASP program: {e}")
            return Answer(
                answer="unknown",
                confidence=0.0,
                explanation=f"Failed to load/ground ASP program: {str(e)}",
                rules_used=rules_used,
                gaps_identified=["ASP syntax or grounding error"],
                asp_query=asp_query,
            )

        # Query ASP
        try:
            result = asp_core.query(
                query_predicate=self._extract_predicate(asp_query.query)
            )

            # Interpret result
            answer_str, confidence = self._interpret_result(result, asp_query)

            # Generate explanation
            explanation = self._generate_explanation(
                asp_query=asp_query,
                result=result,
                answer=answer_str,
                rules_count=len(rules_text),
            )

            # Identify gaps if answer is unknown
            gaps = []
            if answer_str == "unknown":
                gaps = self._identify_gaps(asp_query, result)

            return Answer(
                answer=answer_str,
                confidence=confidence,
                explanation=explanation,
                rules_used=rules_used,
                reasoning_trace=result.model_strings,
                gaps_identified=gaps,
                asp_query=asp_query,
            )

        except Exception as e:
            logger.error(f"ASP query failed: {e}")
            return Answer(
                answer="unknown",
                confidence=0.0,
                explanation=f"Failed to answer question: {str(e)}",
                rules_used=rules_used,
                gaps_identified=["ASP reasoning error"],
                asp_query=asp_query,
            )

    def _extract_predicate(self, query: str) -> str:
        """
        Extract predicate name from query.

        Args:
            query: Query string like "valid_contract(c1)"

        Returns:
            Predicate name like "valid_contract"
        """
        if "(" in query:
            return query.split("(")[0].strip()
        return query.strip()

    def _interpret_result(
        self, result: QueryResult, asp_query: ASPQuery
    ) -> tuple[str, float]:
        """
        Interpret ASP query result as yes/no/unknown.

        Args:
            result: Query result from ASP core
            asp_query: Original query

        Returns:
            Tuple of (answer_string, confidence)
        """
        # Check if satisfiable
        if not result.satisfiable:
            # Program has no answer sets - inconsistent or unsatisfiable
            return "no", 0.8

        # Check if query predicate appears in results
        if result.symbols:
            # Query succeeded - found matching symbols
            return "yes", 0.9
        else:
            # No matching symbols but program is satisfiable
            # This typically means the query is false
            return "no", 0.7

    def _generate_explanation(
        self,
        asp_query: ASPQuery,
        result: QueryResult,
        answer: str,
        rules_count: int,
    ) -> str:
        """
        Generate natural language explanation of answer.

        Args:
            asp_query: Original query
            result: ASP query result
            answer: Interpreted answer (yes/no/unknown)
            rules_count: Number of rules used

        Returns:
            Natural language explanation
        """
        explanation_parts = []

        # Start with the question context
        if asp_query.original_question:
            explanation_parts.append(f"Question: {asp_query.original_question}")

        # Explain the answer
        if answer == "yes":
            explanation_parts.append(
                f"\nThe query '{asp_query.query}' is satisfied based on the given facts and rules."
            )
            if result.symbols:
                symbols_str = ", ".join(str(s) for s in result.symbols[:5])
                if len(result.symbols) > 5:
                    symbols_str += f" (and {len(result.symbols) - 5} more)"
                explanation_parts.append(f"Matching results: {symbols_str}")
        elif answer == "no":
            explanation_parts.append(
                f"\nThe query '{asp_query.query}' is not satisfied based on the given facts and rules."
            )
            explanation_parts.append(
                "The ASP reasoner could not derive this conclusion."
            )
        else:  # unknown
            explanation_parts.append(
                f"\nCannot determine if '{asp_query.query}' holds due to insufficient knowledge."
            )

        # Mention rules used
        if rules_count > 0:
            explanation_parts.append(
                f"\nReasoned using {rules_count} rules from the knowledge base."
            )
        else:
            explanation_parts.append("\nNo relevant rules found in knowledge base.")

        # Add facts
        if asp_query.facts:
            facts_str = "; ".join(asp_query.facts[:3])
            if len(asp_query.facts) > 3:
                facts_str += f" (and {len(asp_query.facts) - 3} more)"
            explanation_parts.append(f"Given facts: {facts_str}")

        return "\n".join(explanation_parts)

    def _identify_gaps(self, asp_query: ASPQuery, result: QueryResult) -> List[str]:
        """
        Identify knowledge gaps when answer is unknown.

        Args:
            asp_query: Original query
            result: ASP query result

        Returns:
            List of identified gaps
        """
        gaps = []

        # Check if no rules were applied
        if result.answer_set_count == 0:
            gaps.append("No rules available for this domain")

        # Check if facts are insufficient
        if len(asp_query.facts) < 2:
            gaps.append("Insufficient facts provided in question")

        # Domain-specific gaps
        if asp_query.domain:
            gaps.append(f"Limited knowledge in {asp_query.domain} domain")

        if not gaps:
            gaps.append("Unknown knowledge gap")

        return gaps

    def batch_answer(
        self, asp_queries: List[ASPQuery], domain: Optional[str] = None
    ) -> List[Answer]:
        """
        Answer multiple questions in batch.

        Args:
            asp_queries: List of ASP queries
            domain: Optional domain filter

        Returns:
            List of answers
        """
        return [self.answer(query, domain=domain) for query in asp_queries]
