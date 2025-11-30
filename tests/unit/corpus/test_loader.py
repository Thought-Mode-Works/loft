"""Tests for corpus loader."""

import json
import tempfile
from pathlib import Path

import pytest

from loft.corpus.loader import (
    CorpusLoader,
    LegalCase,
    get_corpus_stats,
    load_all_domains,
)


@pytest.fixture
def sample_case_data():
    """Sample case data for testing."""
    return {
        "id": "test_001",
        "domain": "contracts",
        "subdomain": "formation",
        "description": "Test contract case",
        "facts": ["Fact 1", "Fact 2"],
        "asp_facts": "contract(c1). offer(c1, yes). acceptance(c1, yes).",
        "question": "Is there a contract?",
        "ground_truth": "enforceable",
        "rationale": "All elements present",
        "legal_citations": ["Test Citation"],
        "difficulty": "easy",
    }


@pytest.fixture
def temp_corpus(sample_case_data):
    """Create a temporary corpus for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create contracts domain
        contracts_dir = base_path / "contracts"
        contracts_dir.mkdir()

        # Create multiple test cases
        for i in range(5):
            case_data = sample_case_data.copy()
            case_data["id"] = f"contract_{i:03d}"
            case_data["ground_truth"] = "enforceable" if i % 2 == 0 else "unenforceable"

            with open(contracts_dir / f"case_{i:03d}.json", "w") as f:
                json.dump(case_data, f)

        # Create torts domain
        torts_dir = base_path / "torts"
        torts_dir.mkdir()

        for i in range(3):
            case_data = {
                "id": f"tort_{i:03d}",
                "domain": "torts",
                "subdomain": "negligence",
                "description": f"Tort case {i}",
                "facts": ["Defendant breached duty"],
                "asp_facts": f"claim(c{i}). duty_owed(c{i}, yes).",
                "question": "Is defendant liable?",
                "ground_truth": "enforceable" if i % 2 == 0 else "unenforceable",
                "rationale": "Test rationale",
            }
            with open(torts_dir / f"tort_{i:03d}.json", "w") as f:
                json.dump(case_data, f)

        yield base_path


class TestLegalCase:
    """Tests for LegalCase dataclass."""

    def test_from_json(self, sample_case_data):
        """Test creating LegalCase from JSON data."""
        case = LegalCase.from_json(sample_case_data)

        assert case.id == "test_001"
        assert case.domain == "contracts"
        assert case.subdomain == "formation"
        assert case.ground_truth == "enforceable"
        assert len(case.facts) == 2

    def test_from_json_with_source_file(self, sample_case_data):
        """Test creating LegalCase with source file."""
        source = Path("/test/path.json")
        case = LegalCase.from_json(sample_case_data, source_file=source)

        assert case.source_file == source

    def test_to_dict(self, sample_case_data):
        """Test converting LegalCase to dictionary."""
        case = LegalCase.from_json(sample_case_data)
        result = case.to_dict()

        assert result["id"] == "test_001"
        assert result["domain"] == "contracts"
        assert "source_file" not in result  # Should not be in output

    def test_default_values(self):
        """Test default values for optional fields."""
        minimal_data = {
            "id": "min_001",
            "asp_facts": "test(x).",
            "question": "Test?",
            "ground_truth": "enforceable",
            "rationale": "Test",
        }
        case = LegalCase.from_json(minimal_data)

        assert case.difficulty == "medium"  # Default
        assert case.legal_citations == []  # Default
        assert case.subdomain is None


class TestCorpusLoader:
    """Tests for CorpusLoader class."""

    def test_load_directory(self, temp_corpus):
        """Test loading cases from a directory."""
        loader = CorpusLoader(temp_corpus)
        cases = loader.load_directory("contracts")

        assert len(cases) == 5
        assert all(c.domain == "contracts" for c in cases)

    def test_load_all_domains(self, temp_corpus):
        """Test loading all domains."""
        loader = CorpusLoader(temp_corpus)
        cases = loader.load_all_domains()

        # Should load contracts (5) + torts (3) = 8
        assert len(cases) == 8

    def test_get_cases_filter_by_domain(self, temp_corpus):
        """Test filtering cases by domain."""
        loader = CorpusLoader(temp_corpus)
        loader.load_all_domains()

        contracts = loader.get_cases(domain="contracts")
        torts = loader.get_cases(domain="torts")

        assert len(contracts) == 5
        assert len(torts) == 3

    def test_get_cases_filter_by_outcome(self, temp_corpus):
        """Test filtering cases by outcome."""
        loader = CorpusLoader(temp_corpus)
        loader.load_all_domains()

        enforceable = loader.get_cases(outcome="enforceable")
        unenforceable = loader.get_cases(outcome="unenforceable")

        # Contracts: 3 enforceable, 2 unenforceable
        # Torts: 2 enforceable, 1 unenforceable
        assert len(enforceable) == 5
        assert len(unenforceable) == 3

    def test_get_cases_multiple_filters(self, temp_corpus):
        """Test filtering with multiple criteria."""
        loader = CorpusLoader(temp_corpus)
        loader.load_all_domains()

        result = loader.get_cases(domain="contracts", outcome="enforceable")

        assert len(result) == 3
        assert all(c.domain == "contracts" for c in result)
        assert all(c.ground_truth == "enforceable" for c in result)

    def test_get_stats(self, temp_corpus):
        """Test getting corpus statistics."""
        loader = CorpusLoader(temp_corpus)
        loader.load_all_domains()

        stats = loader.get_stats()

        assert stats.total_cases == 8
        assert stats.cases_by_domain["contracts"] == 5
        assert stats.cases_by_domain["torts"] == 3
        assert "enforceable" in stats.cases_by_outcome
        assert len(stats.unique_predicates) > 0

    def test_iter_cases(self, temp_corpus):
        """Test iterating over cases."""
        loader = CorpusLoader(temp_corpus)
        loader.load_all_domains()

        cases = list(loader.iter_cases())

        assert len(cases) == 8

    def test_len(self, temp_corpus):
        """Test length of loader."""
        loader = CorpusLoader(temp_corpus)
        loader.load_all_domains()

        assert len(loader) == 8

    def test_empty_loader(self, temp_corpus):
        """Test empty loader before loading."""
        loader = CorpusLoader(temp_corpus)

        assert len(loader) == 0
        assert loader.get_stats().total_cases == 0

    def test_nonexistent_directory(self, temp_corpus):
        """Test loading from nonexistent directory."""
        loader = CorpusLoader(temp_corpus)
        cases = loader.load_directory("nonexistent")

        assert len(cases) == 0


class TestCorpusStats:
    """Tests for CorpusStats class."""

    def test_str_format(self, temp_corpus):
        """Test string formatting of stats."""
        loader = CorpusLoader(temp_corpus)
        loader.load_all_domains()

        stats = loader.get_stats()
        result = str(stats)

        assert "Total cases: 8" in result
        assert "contracts" in result
        assert "torts" in result
        assert "enforceable" in result


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_corpus_stats(self, temp_corpus):
        """Test get_corpus_stats function."""
        stats = get_corpus_stats(temp_corpus)

        assert stats.total_cases == 8

    def test_load_all_domains_function(self, temp_corpus):
        """Test load_all_domains function."""
        cases = load_all_domains(temp_corpus)

        assert len(cases) == 8
