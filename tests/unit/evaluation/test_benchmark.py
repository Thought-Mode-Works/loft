"""
Unit tests for benchmark loading.

Issue #277: Legal Question Test Suite
"""

import json
import pytest

from loft.evaluation.benchmark import BenchmarkLoader, BenchmarkQuestion, BenchmarkSuite


@pytest.fixture
def sample_benchmark_data():
    """Create sample benchmark data."""
    return {
        "domain": "test_domain",
        "description": "Test questions for unit testing",
        "questions": [
            {
                "id": "test_001",
                "question": "Is this a test question?",
                "expected_answer": "yes",
                "difficulty": "easy",
                "topics": ["testing", "examples"],
                "explanation": "This is a test question for unit testing.",
            },
            {
                "id": "test_002",
                "question": "Does testing matter?",
                "expected_answer": "yes",
                "difficulty": "medium",
                "topics": ["testing", "quality"],
                "explanation": "Testing is essential for quality software.",
            },
        ],
    }


@pytest.fixture
def sample_benchmark_file(tmp_path, sample_benchmark_data):
    """Create sample benchmark JSON file."""
    file_path = tmp_path / "test_domain.json"
    with open(file_path, "w") as f:
        json.dump(sample_benchmark_data, f)
    return file_path


class TestBenchmarkQuestion:
    """Test BenchmarkQuestion dataclass."""

    def test_question_initialization(self):
        """Test creating a benchmark question."""
        q = BenchmarkQuestion(
            id="test_001",
            question="Test question?",
            expected_answer="yes",
            difficulty="easy",
            topics=["test"],
            domain="test",
            explanation="Test explanation",
        )

        assert q.id == "test_001"
        assert q.question == "Test question?"
        assert q.expected_answer == "yes"
        assert q.difficulty == "easy"
        assert "test" in q.topics
        assert q.domain == "test"

    def test_question_with_reasoning_steps(self):
        """Test question with multi-step reasoning."""
        q = BenchmarkQuestion(
            id="test_002",
            question="Complex question?",
            expected_answer="yes",
            difficulty="hard",
            topics=["complex"],
            domain="test",
            reasoning_steps=["Step 1", "Step 2", "Step 3"],
        )

        assert len(q.reasoning_steps) == 3
        assert "Step 1" in q.reasoning_steps


class TestBenchmarkSuite:
    """Test BenchmarkSuite."""

    def test_suite_initialization(self, sample_benchmark_data):
        """Test creating a benchmark suite."""
        suite = BenchmarkSuite(
            domain="test_domain", description="Test suite", questions=[]
        )

        assert suite.domain == "test_domain"
        assert suite.description == "Test suite"
        assert suite.question_count == 0

    def test_suite_with_questions(self, sample_benchmark_data):
        """Test suite with questions."""
        questions = [
            BenchmarkQuestion(
                id=q["id"],
                question=q["question"],
                expected_answer=q["expected_answer"],
                difficulty=q["difficulty"],
                topics=q["topics"],
                domain=sample_benchmark_data["domain"],
            )
            for q in sample_benchmark_data["questions"]
        ]

        suite = BenchmarkSuite(
            domain="test_domain", description="Test suite", questions=questions
        )

        assert suite.question_count == 2

    def test_get_by_difficulty(self):
        """Test filtering questions by difficulty."""
        questions = [
            BenchmarkQuestion(
                id="easy_1",
                question="Easy?",
                expected_answer="yes",
                difficulty="easy",
                topics=[],
                domain="test",
            ),
            BenchmarkQuestion(
                id="hard_1",
                question="Hard?",
                expected_answer="no",
                difficulty="hard",
                topics=[],
                domain="test",
            ),
            BenchmarkQuestion(
                id="easy_2",
                question="Also easy?",
                expected_answer="yes",
                difficulty="easy",
                topics=[],
                domain="test",
            ),
        ]

        suite = BenchmarkSuite(domain="test", description="Test", questions=questions)

        easy_questions = suite.get_by_difficulty("easy")
        assert len(easy_questions) == 2

        hard_questions = suite.get_by_difficulty("hard")
        assert len(hard_questions) == 1

    def test_get_by_topic(self):
        """Test filtering questions by topic."""
        questions = [
            BenchmarkQuestion(
                id="q1",
                question="Q1?",
                expected_answer="yes",
                difficulty="easy",
                topics=["contracts", "formation"],
                domain="test",
            ),
            BenchmarkQuestion(
                id="q2",
                question="Q2?",
                expected_answer="no",
                difficulty="easy",
                topics=["torts", "negligence"],
                domain="test",
            ),
            BenchmarkQuestion(
                id="q3",
                question="Q3?",
                expected_answer="yes",
                difficulty="easy",
                topics=["contracts", "breach"],
                domain="test",
            ),
        ]

        suite = BenchmarkSuite(domain="test", description="Test", questions=questions)

        contract_questions = suite.get_by_topic("contracts")
        assert len(contract_questions) == 2

        tort_questions = suite.get_by_topic("torts")
        assert len(tort_questions) == 1

    def test_get_by_id(self):
        """Test getting question by ID."""
        questions = [
            BenchmarkQuestion(
                id="test_123",
                question="Find me?",
                expected_answer="yes",
                difficulty="easy",
                topics=[],
                domain="test",
            ),
        ]

        suite = BenchmarkSuite(domain="test", description="Test", questions=questions)

        found = suite.get_by_id("test_123")
        assert found is not None
        assert found.question == "Find me?"

        not_found = suite.get_by_id("nonexistent")
        assert not_found is None


class TestBenchmarkLoader:
    """Test BenchmarkLoader."""

    def test_loader_initialization(self, tmp_path):
        """Test creating a loader."""
        loader = BenchmarkLoader(data_dir=tmp_path)
        assert loader.data_dir == tmp_path

    def test_load_suite(self, tmp_path, sample_benchmark_file):
        """Test loading a benchmark suite from file."""
        loader = BenchmarkLoader(data_dir=tmp_path)
        suite = loader.load_suite("test_domain.json")

        assert suite.domain == "test_domain"
        assert suite.question_count == 2
        assert suite.questions[0].id == "test_001"
        assert suite.questions[1].id == "test_002"

    def test_load_suite_without_extension(self, tmp_path, sample_benchmark_file):
        """Test loading suite without .json extension."""
        loader = BenchmarkLoader(data_dir=tmp_path)
        suite = loader.load_suite("test_domain")  # No .json extension

        assert suite.question_count == 2

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file raises error."""
        loader = BenchmarkLoader(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_suite("nonexistent.json")

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises error."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalid json }")

        loader = BenchmarkLoader(data_dir=tmp_path)

        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_suite("bad.json")

    def test_load_all_suites(self, tmp_path, sample_benchmark_data):
        """Test loading all suites from directory."""
        # Create multiple benchmark files
        for domain in ["contracts", "torts", "property"]:
            data = sample_benchmark_data.copy()
            data["domain"] = domain
            file_path = tmp_path / f"{domain}.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

        loader = BenchmarkLoader(data_dir=tmp_path)
        suites = loader.load_all_suites()

        assert len(suites) == 3
        assert "contracts" in suites
        assert "torts" in suites
        assert "property" in suites

    def test_get_available_suites(self, tmp_path):
        """Test getting list of available suites."""
        # Create some files
        (tmp_path / "contracts.json").write_text("{}")
        (tmp_path / "torts.json").write_text("{}")
        (tmp_path / "readme.txt").write_text("Not a benchmark")

        loader = BenchmarkLoader(data_dir=tmp_path)
        available = loader.get_available_suites()

        assert "contracts" in available
        assert "torts" in available
        assert "readme" not in available  # Not a .json file
        assert len(available) == 2

    def test_load_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        loader = BenchmarkLoader(data_dir=tmp_path)
        suites = loader.load_all_suites()

        assert len(suites) == 0
