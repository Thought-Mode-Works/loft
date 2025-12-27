"""
Main legal question answering interface.

Provides the high-level API for asking legal questions and getting answers.

Issue #272: Legal Question Answering Interface
"""

from typing import List, Optional, Tuple

from loguru import logger

from loft.knowledge.database import KnowledgeDatabase
from loft.qa.question_parser import LegalQuestionParser
from loft.qa.reasoner import LegalReasoner
from loft.qa.schemas import Answer, ASPQuery, EvaluationReport, QuestionResult


class LegalQAInterface:
    """
    Main interface for legal question answering.

    Combines question parsing, reasoning, and answer generation into a
    unified interface.

    Example:
        qa = LegalQAInterface(knowledge_db=db)
        answer = qa.ask("Is a contract valid without consideration?")
        print(answer.to_natural_language())
    """

    def __init__(
        self,
        knowledge_db: Optional[KnowledgeDatabase] = None,
        parser: Optional[LegalQuestionParser] = None,
        reasoner: Optional[LegalReasoner] = None,
        log_questions: bool = True,
    ):
        """
        Initialize QA interface.

        Args:
            knowledge_db: Knowledge database for rules
            parser: Question parser (creates default if None)
            reasoner: Reasoner (creates default if None)
            log_questions: Whether to log questions to database
        """
        self.knowledge_db = knowledge_db
        self.parser = parser or LegalQuestionParser()
        self.reasoner = reasoner or LegalReasoner(knowledge_db=knowledge_db)
        self.log_questions = log_questions

    def ask(
        self,
        question: str,
        domain: Optional[str] = None,
        expected_answer: Optional[str] = None,
    ) -> Answer:
        """
        Ask a legal question and get an answer.

        Full pipeline:
        1. Parse question → ASP query
        2. Reason using knowledge base
        3. Generate natural language explanation
        4. Log question and answer (if enabled)

        Args:
            question: Natural language legal question
            domain: Optional domain hint (contracts, torts, etc.)
            expected_answer: Optional expected answer for evaluation

        Returns:
            Answer with result, explanation, and citations
        """
        logger.info(f"QA: {question[:100]}...")

        # Parse question to ASP
        try:
            asp_query = self.parser.parse(question, domain=domain)
        except Exception as e:
            logger.error(f"Failed to parse question: {e}")
            return Answer(
                answer="unknown",
                confidence=0.0,
                explanation=f"Failed to parse question: {str(e)}",
                gaps_identified=["Question parsing error"],
            )

        # Reason to get answer
        try:
            answer = self.reasoner.answer(
                asp_query=asp_query,
                domain=domain or asp_query.domain,
            )
        except Exception as e:
            logger.error(f"Failed to reason: {e}")
            return Answer(
                answer="unknown",
                confidence=0.0,
                explanation=f"Failed to reason about question: {str(e)}",
                gaps_identified=["Reasoning error"],
            )

        # Log to database if enabled
        if self.log_questions and self.knowledge_db:
            try:
                self._log_question(question, asp_query, answer, expected_answer)
            except Exception as e:
                logger.warning(f"Failed to log question: {e}")

        # Update rule usage statistics
        if self.knowledge_db:
            try:
                for rule_id in answer.rules_used:
                    self.knowledge_db.mark_rule_used(rule_id)
            except Exception as e:
                logger.warning(f"Failed to update rule usage: {e}")

        return answer

    def ask_with_feedback(
        self,
        question: str,
        correct_answer: str,
        domain: Optional[str] = None,
    ) -> Answer:
        """
        Ask a question and provide feedback on correctness.

        Updates rule performance based on whether answer was correct.

        Args:
            question: Natural language question
            correct_answer: Known correct answer (yes/no)
            domain: Optional domain hint

        Returns:
            Answer object
        """
        answer = self.ask(question, domain=domain, expected_answer=correct_answer)

        # Update rule performance
        if self.knowledge_db:
            correct = answer.is_correct(correct_answer)
            for rule_id in answer.rules_used:
                try:
                    self.knowledge_db.update_rule_performance(rule_id, success=correct)
                except Exception as e:
                    logger.warning(f"Failed to update rule performance: {e}")

        return answer

    def batch_eval(
        self,
        questions: List[Tuple[str, str]],  # (question, expected_answer)
        domain: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Evaluate QA system on a batch of questions.

        Args:
            questions: List of (question, expected_answer) tuples
            domain: Optional domain for all questions

        Returns:
            Evaluation report with accuracy and per-domain metrics
        """
        logger.info(f"Evaluating on {len(questions)} questions...")

        results = []
        for question, expected in questions:
            # Answer question
            answer = self.ask_with_feedback(question, expected, domain=domain)

            # Create result
            result = QuestionResult(
                question=question,
                expected_answer=expected,
                actual_answer=answer,
                domain=domain or answer.asp_query.domain if answer.asp_query else None,
            )
            results.append(result)

        # Create evaluation report
        report = EvaluationReport(results=results)
        logger.info(f"Evaluation complete: {report.accuracy:.1%} accuracy")

        return report

    def _log_question(
        self,
        question: str,
        asp_query: ASPQuery,
        answer: Answer,
        expected_answer: Optional[str] = None,
    ) -> None:
        """
        Log question and answer to knowledge database.

        Args:
            question: Original natural language question
            asp_query: Parsed ASP query
            answer: Generated answer
            expected_answer: Optional expected answer for validation
        """
        try:
            # Determine correctness
            correct = None
            if expected_answer is not None:
                correct = answer.is_correct(expected_answer)

            # Log to database
            self.knowledge_db.add_question(
                question_text=question,
                asp_query=asp_query.to_asp_program(),
                answer=answer.answer,
                reasoning=answer.explanation,
                rules_used=answer.rules_used,
                confidence=answer.confidence,
                correct=correct,
                domain=asp_query.domain,
            )

            logger.debug("Logged question to database")

        except Exception as e:
            logger.warning(f"Failed to log question: {e}")

    def get_performance_summary(self) -> dict:
        """
        Get performance summary from knowledge database.

        Returns:
            Dictionary with performance metrics
        """
        if not self.knowledge_db:
            return {"error": "No knowledge database configured"}

        try:
            stats = self.knowledge_db.get_database_stats()
            return {
                "total_rules": stats.total_rules,
                "active_rules": stats.active_rules,
                "total_questions": stats.total_questions,
                "domains": stats.domains,
                "avg_confidence": stats.avg_confidence,
                "coverage_by_domain": stats.coverage_by_domain,
            }
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
