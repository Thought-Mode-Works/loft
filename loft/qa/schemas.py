"""
Data schemas for legal question answering.

Defines data structures for questions, answers, and evaluations.

Issue #272: Legal Question Answering Interface
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ASPQuery:
    """
    Parsed legal question as ASP query.

    Represents a natural language legal question translated into
    ASP facts and a query goal.

    Example:
        Question: "Is a contract valid without consideration?"
        Facts: ["offer(c1).", "acceptance(c1).", "not consideration(c1)."]
        Query: "valid_contract(c1)"
        Domain: "contracts"
    """

    facts: List[str]
    query: str
    domain: Optional[str] = None
    original_question: str = ""
    confidence: float = 1.0

    def to_asp_program(self) -> str:
        """Convert to complete ASP program string."""
        program = "\n".join(self.facts)
        if self.query:
            program += f"\n?- {self.query}."
        return program

    def __str__(self) -> str:
        """String representation for logging."""
        facts_str = "; ".join(self.facts[:3])
        if len(self.facts) > 3:
            facts_str += f" ... ({len(self.facts)} total)"
        return f"ASPQuery(facts=[{facts_str}], query={self.query})"


@dataclass
class Answer:
    """
    Answer to a legal question with explanation.

    Attributes:
        answer: yes, no, or unknown
        confidence: Confidence score 0-1
        explanation: Natural language explanation
        rules_used: List of rule IDs that were applied
        reasoning_trace: Step-by-step reasoning
        gaps_identified: Missing knowledge areas
        timestamp: When answer was generated
        asp_query: Original ASP query
    """

    answer: str
    confidence: float
    explanation: str
    rules_used: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    asp_query: Optional[ASPQuery] = None

    def to_natural_language(self) -> str:
        """
        Convert answer to formatted natural language.

        Returns:
            Formatted string with answer, explanation, rules, and confidence
        """
        nl = f"Answer: {self.answer.capitalize()}\n\n"
        nl += f"Explanation:\n{self.explanation}\n"

        if self.rules_used:
            nl += "\nRules Applied:\n"
            for rule_id in self.rules_used:
                # Truncate long UUIDs for readability
                display_id = rule_id[:8] if len(rule_id) > 8 else rule_id
                nl += f"  - {display_id}\n"

        if self.gaps_identified:
            nl += "\nKnowledge Gaps:\n"
            for gap in self.gaps_identified:
                nl += f"  - {gap}\n"

        nl += f"\nConfidence: {self.confidence:.0%}"
        return nl

    def is_correct(self, expected: str) -> bool:
        """Check if answer matches expected answer."""
        return self.answer.lower() == expected.lower()


@dataclass
class QuestionResult:
    """Result of answering a single question in evaluation."""

    question: str
    expected_answer: Optional[str]
    actual_answer: Answer
    correct: Optional[bool] = None
    domain: Optional[str] = None

    def __post_init__(self):
        """Calculate correctness if expected answer provided."""
        if self.expected_answer is not None:
            self.correct = self.actual_answer.is_correct(self.expected_answer)


@dataclass
class EvaluationReport:
    """
    Report from batch evaluation of QA system.

    Attributes:
        results: List of question results
        total_questions: Total questions evaluated
        correct_count: Number of correct answers
        incorrect_count: Number of incorrect answers
        unknown_count: Number of unknown answers
        accuracy: Overall accuracy (correct / total)
        avg_confidence: Average confidence score
        by_domain: Accuracy breakdown by domain
    """

    results: List[QuestionResult]
    total_questions: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    unknown_count: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    by_domain: dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate summary statistics."""
        self.total_questions = len(self.results)

        if self.total_questions == 0:
            return

        # Count answers
        for result in self.results:
            if result.actual_answer.answer == "unknown":
                self.unknown_count += 1
            elif result.correct is True:
                self.correct_count += 1
            elif result.correct is False:
                self.incorrect_count += 1

            # Track by domain
            domain = result.domain or "unknown"
            if domain not in self.by_domain:
                self.by_domain[domain] = {"correct": 0, "total": 0}
            self.by_domain[domain]["total"] += 1
            if result.correct is True:
                self.by_domain[domain]["correct"] += 1

        # Calculate accuracy
        if self.total_questions > 0:
            self.accuracy = self.correct_count / self.total_questions

        # Calculate average confidence
        confidences = [r.actual_answer.confidence for r in self.results]
        self.avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Calculate domain accuracies
        for domain_stats in self.by_domain.values():
            total = domain_stats["total"]
            domain_stats["accuracy"] = (
                domain_stats["correct"] / total if total > 0 else 0.0
            )

    def to_string(self) -> str:
        """Format evaluation report as string."""
        report = "Evaluation Results\n"
        report += "=" * 50 + "\n\n"
        report += f"Total Questions: {self.total_questions}\n"
        report += f"Correct: {self.correct_count} ({self.correct_count/self.total_questions*100:.1f}%)\n"
        report += f"Incorrect: {self.incorrect_count} ({self.incorrect_count/self.total_questions*100:.1f}%)\n"
        report += f"Unknown: {self.unknown_count} ({self.unknown_count/self.total_questions*100:.1f}%)\n"
        report += f"Accuracy: {self.accuracy:.1%}\n"
        report += f"Average Confidence: {self.avg_confidence:.1%}\n\n"

        if self.by_domain:
            report += "By Domain:\n"
            for domain, stats in sorted(self.by_domain.items()):
                report += f"  {domain}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})\n"

        return report
