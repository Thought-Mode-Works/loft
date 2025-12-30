"""
Benchmark test suite loading and management.

Loads legal question test datasets from JSON files and provides
access to questions for evaluation.

Issue #277: Legal Question Test Suite
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuestion:
    """
    A single benchmark question.

    Attributes:
        id: Unique question identifier
        question: The question text
        expected_answer: Expected answer (yes/no/unknown)
        difficulty: easy/medium/hard
        topics: List of legal topics covered
        domain: Legal domain (contracts, torts, etc.)
        explanation: Explanation of correct answer
        reasoning_steps: Optional multi-step reasoning process
    """

    id: str
    question: str
    expected_answer: str
    difficulty: str
    topics: List[str]
    domain: str
    explanation: str = ""
    reasoning_steps: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate question data."""
        valid_answers = ["yes", "no", "unknown", "voidable"]
        if self.expected_answer.lower() not in valid_answers:
            logger.warning(
                f"Question {self.id} has unexpected answer: {self.expected_answer}"
            )

        valid_difficulties = ["easy", "medium", "hard"]
        if self.difficulty.lower() not in valid_difficulties:
            logger.warning(
                f"Question {self.id} has unexpected difficulty: {self.difficulty}"
            )


@dataclass
class BenchmarkSuite:
    """
    A collection of benchmark questions for a domain.

    Attributes:
        domain: Legal domain name
        description: Description of the test suite
        questions: List of benchmark questions
    """

    domain: str
    description: str
    questions: List[BenchmarkQuestion] = field(default_factory=list)

    @property
    def question_count(self) -> int:
        """Total number of questions."""
        return len(self.questions)

    def get_by_difficulty(self, difficulty: str) -> List[BenchmarkQuestion]:
        """Get questions by difficulty level."""
        return [q for q in self.questions if q.difficulty.lower() == difficulty.lower()]

    def get_by_topic(self, topic: str) -> List[BenchmarkQuestion]:
        """Get questions covering a specific topic."""
        return [q for q in self.questions if topic in q.topics]

    def get_by_id(self, question_id: str) -> Optional[BenchmarkQuestion]:
        """Get question by ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def __str__(self) -> str:
        """String representation."""
        return f"BenchmarkSuite(domain={self.domain}, questions={self.question_count})"


class BenchmarkLoader:
    """
    Loads benchmark test suites from JSON files.

    Supports loading individual domain files or entire directories
    of benchmark questions.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            data_dir: Directory containing benchmark JSON files.
                     Defaults to tests/data/legal_questions/
        """
        if data_dir is None:
            # Default to tests/data/legal_questions relative to project root
            self.data_dir = (
                Path(__file__).parent.parent.parent
                / "tests"
                / "data"
                / "legal_questions"
            )
        else:
            self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            logger.warning(f"Benchmark data directory not found: {self.data_dir}")

    def load_suite(self, filename: str) -> BenchmarkSuite:
        """
        Load a single benchmark suite from JSON file.

        Args:
            filename: Name of JSON file (with or without .json extension)

        Returns:
            BenchmarkSuite loaded from file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        # Add .json extension if not present
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")

        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")

        # Parse questions
        questions = []
        for q_data in data.get("questions", []):
            question = BenchmarkQuestion(
                id=q_data["id"],
                question=q_data["question"],
                expected_answer=q_data["expected_answer"],
                difficulty=q_data.get("difficulty", "medium"),
                topics=q_data.get("topics", []),
                domain=data.get("domain", "unknown"),
                explanation=q_data.get("explanation", ""),
                reasoning_steps=q_data.get("reasoning_steps", []),
            )
            questions.append(question)

        suite = BenchmarkSuite(
            domain=data.get("domain", "unknown"),
            description=data.get("description", ""),
            questions=questions,
        )

        logger.info(f"Loaded {suite.question_count} questions from {filename}")
        return suite

    def load_all_suites(self) -> Dict[str, BenchmarkSuite]:
        """
        Load all benchmark suites from data directory.

        Returns:
            Dictionary mapping domain names to BenchmarkSuites
        """
        suites = {}

        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return suites

        # Find all JSON files
        json_files = list(self.data_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir}")
            return suites

        for json_file in json_files:
            try:
                suite = self.load_suite(json_file.name)
                suites[suite.domain] = suite
            except Exception as e:
                logger.error(f"Failed to load {json_file.name}: {e}")

        logger.info(f"Loaded {len(suites)} benchmark suites")
        return suites

    def get_available_suites(self) -> List[str]:
        """
        Get list of available benchmark suite names.

        Returns:
            List of .json filenames (without extension)
        """
        if not self.data_dir.exists():
            return []

        return [f.stem for f in self.data_dir.glob("*.json")]
