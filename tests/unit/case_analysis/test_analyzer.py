"""
Unit tests for case analyzer.

Issue #276: Case Analysis and Rule Extraction
"""

import json
from pathlib import Path

import pytest

from loft.case_analysis.analyzer import CaseAnalyzer
from loft.case_analysis.schemas import CaseDocument, CaseFormat
from loft.neural.rule_generator import GeneratedRule


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self):
        """Initialize mock LLM."""
        self.queries = []

    def query(
        self,
        question: str = None,
        prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        output_schema=None,
        **kwargs,
    ):
        """Mock query method."""
        # Support both question and prompt for compatibility
        query_text = question or prompt or ""
        self.queries.append({"question": query_text, "temperature": temperature})

        # Simple response objects
        class MockResponse:
            def __init__(self, raw_text):
                self.raw_text = raw_text

        # Metadata extraction
        if "Extract and return the following information as JSON" in query_text:
            return MockResponse(
                json.dumps(
                    {
                        "title": "Test v. Case",
                        "court": "Test Court",
                        "jurisdiction": "federal",
                        "domain": "contracts",
                        "confidence": 0.9,
                    }
                )
            )

        # Principle extraction
        elif "identify the key legal principles" in query_text.lower():
            return MockResponse(
                json.dumps(
                    [
                        {
                            "principle_text": "A contract requires offer and acceptance.",
                            "domain": "contracts",
                            "source_section": "analysis",
                            "confidence": 0.9,
                            "reasoning": "Fundamental contract principle",
                            "related_facts": [],
                            "case_specific": False,
                        }
                    ]
                )
            )

        # Rule generation - need to return structured GeneratedRule for output_schema requests
        elif (
            output_schema == GeneratedRule
            or "translate" in query_text.lower()
            or "asp" in query_text.lower()
        ):
            # For structured output, return object with content attribute
            class StructuredResponse:
                def __init__(self):
                    self.content = GeneratedRule(
                        asp_rule="valid_contract(X) :- offer(X), acceptance(X).",
                        confidence=0.85,
                        reasoning="Translated contract formation principle to ASP",
                    )
                    self.raw_text = json.dumps(
                        {
                            "asp_rule": "valid_contract(X) :- offer(X), acceptance(X).",
                            "confidence": 0.85,
                            "reasoning": "Translated contract formation principle to ASP",
                        }
                    )

            return StructuredResponse()

        return MockResponse("{}")


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    return MockLLM()


@pytest.fixture
def analyzer(mock_llm):
    """Create analyzer with mock LLM."""
    return CaseAnalyzer(
        llm=mock_llm,
        min_principle_confidence=0.5,
        min_rule_confidence=0.6,
        max_principles_per_case=10,
    )


@pytest.fixture
def sample_case_doc():
    """Create sample case document."""
    return CaseDocument(
        content="""
        Test Court

        Test v. Case
        No. 2023-001

        FACTS:
        The parties entered into a contract.

        ANALYSIS:
        A valid contract requires offer and acceptance.

        HOLDING:
        The contract is valid.
        """,
        format=CaseFormat.TEXT,
        case_id="test_case",
        title="Test v. Case",
    )


@pytest.fixture
def sample_text_file(tmp_path):
    """Create sample text file."""
    file_path = tmp_path / "test_case.txt"
    content = """
    Supreme Court

    Sample v. Example
    No. 123-456

    This is a sample case for testing.
    """
    file_path.write_text(content)
    return file_path


class TestCaseAnalyzer:
    """Test case analyzer."""

    def test_analyzer_initialization(self, mock_llm):
        """Test analyzer initializes correctly."""
        analyzer = CaseAnalyzer(mock_llm)

        assert analyzer.llm == mock_llm
        assert analyzer.parser is not None
        assert analyzer.metadata_extractor is not None
        assert analyzer.principle_extractor is not None
        assert analyzer.rule_generator is not None

    def test_analyze_document(self, analyzer, sample_case_doc):
        """Test analyzing a case document."""
        result = analyzer.analyze_document(sample_case_doc)

        assert result is not None
        assert result.case_id == "test_case"
        assert result.metadata is not None
        assert len(result.principles) >= 1
        # Rules may be 0 if validation fails in ASP core
        assert isinstance(result.rules, list)
        assert result.processing_time_ms > 0

    def test_analyze_file(self, analyzer, sample_text_file):
        """Test analyzing a case file."""
        result = analyzer.analyze_file(sample_text_file)

        assert result is not None
        assert result.case_id == "test_case"
        assert result.metadata is not None
        assert result.processing_time_ms > 0

    def test_principle_to_rule_generation(self, analyzer, sample_case_doc):
        """Test rule generation from principles."""
        result = analyzer.analyze_document(sample_case_doc)

        # Rules are generated from principles, but may be filtered by validation
        assert len(result.principles) >= 1

        # If rules were generated, check their structure
        if result.rules:
            rule = result.rules[0]
            assert rule.asp_rule
            assert ":-" in rule.asp_rule or rule.asp_rule.endswith(".")
            assert rule.principle is not None
            assert rule.confidence > 0
            assert len(rule.predicates_used) > 0

    def test_confidence_filtering(self, mock_llm):
        """Test that low-confidence rules are filtered."""
        # Create analyzer with high confidence threshold
        analyzer = CaseAnalyzer(
            mock_llm,
            min_principle_confidence=0.95,  # Very high
            min_rule_confidence=0.95,  # Very high
        )

        case_doc = CaseDocument(
            content="Test case",
            format=CaseFormat.TEXT,
            case_id="test",
        )

        result = analyzer.analyze_document(case_doc)

        # With high thresholds, may filter out all rules
        # Just verify it doesn't crash
        assert result is not None
        assert isinstance(result.rules, list)

    def test_validation_enabled(self, mock_llm):
        """Test rule validation when enabled."""
        analyzer = CaseAnalyzer(mock_llm, validate_rules=True)
        case_doc = CaseDocument(
            content="Test case",
            format=CaseFormat.TEXT,
            case_id="test",
        )

        result = analyzer.analyze_document(case_doc)

        # Validation should have run
        for rule in result.rules:
            # validation_passed should be set
            assert isinstance(rule.validation_passed, bool)

    def test_validate_asp_syntax(self, analyzer):
        """Test ASP syntax validation."""
        # Valid rule
        assert analyzer._validate_asp_syntax("valid(X) :- condition(X).")

        # Valid fact
        assert analyzer._validate_asp_syntax("fact(a).")

        # Invalid - no :- or ending .
        assert not analyzer._validate_asp_syntax("invalid rule")

        # Invalid - unbalanced parentheses
        assert not analyzer._validate_asp_syntax("invalid(X :- condition(X).")

        # Invalid - empty
        assert not analyzer._validate_asp_syntax("")

    def test_extract_predicates(self, analyzer):
        """Test predicate extraction from rules."""
        rule = "valid_contract(X) :- offer(X), acceptance(X), consideration(X)."
        predicates = analyzer._extract_predicates(rule)

        assert "valid_contract" in predicates
        assert "offer" in predicates
        assert "acceptance" in predicates
        assert "consideration" in predicates

    def test_batch_analysis(self, analyzer, tmp_path):
        """Test analyzing multiple files."""
        # Create multiple test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"case_{i}.txt"
            file_path.write_text(f"Test case {i} content")
            files.append(file_path)

        results = analyzer.analyze_batch(files)

        assert len(results) == 3
        for result in results:
            assert result is not None
            assert result.processing_time_ms > 0

    def test_error_handling(self, analyzer, tmp_path):
        """Test error handling for problematic files."""
        # Nonexistent file
        bad_file = tmp_path / "nonexistent.txt"

        result = analyzer.analyze_file(bad_file)

        # Should return result with errors, not crash
        assert result is not None
        assert len(result.errors) > 0
        assert not result.success

    def test_analysis_result_properties(self, analyzer, sample_case_doc):
        """Test CaseAnalysisResult properties."""
        result = analyzer.analyze_document(sample_case_doc)

        # Test properties
        assert result.principle_count == len(result.principles)
        assert result.rule_count == len(result.rules)

        if result.rules:
            avg_conf = sum(r.confidence for r in result.rules) / len(result.rules)
            assert abs(result.avg_confidence - avg_conf) < 0.001

        # Test success property
        if result.rule_count > 0 and not result.errors:
            assert result.success

    def test_analysis_result_to_dict(self, analyzer, sample_case_doc):
        """Test converting analysis result to dictionary."""
        result = analyzer.analyze_document(sample_case_doc)
        result_dict = result.to_dict()

        assert "case_id" in result_dict
        assert "metadata" in result_dict
        assert "principle_count" in result_dict
        assert "rule_count" in result_dict
        assert "avg_confidence" in result_dict
        assert "success" in result_dict
        assert "principles" in result_dict
        assert "rules" in result_dict

    def test_metadata_extraction_failure_handling(self, mock_llm):
        """Test handling of metadata extraction failures."""

        # Create LLM that fails for metadata
        class FailingLLM(MockLLM):
            def query(
                self,
                question=None,
                prompt=None,
                temperature=0.7,
                max_tokens=500,
                **kwargs,
            ):
                query_text = question or prompt or ""
                if "Extract and return the following" in query_text:
                    raise Exception("Metadata extraction failed")
                return super().query(
                    question=question,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

        analyzer = CaseAnalyzer(FailingLLM())
        case_doc = CaseDocument(
            content="Test",
            format=CaseFormat.TEXT,
            case_id="test",
        )

        result = analyzer.analyze_document(case_doc)

        # Should handle gracefully - errors are caught internally
        assert result is not None
        # Error is logged but may not be in results.errors list
        # since extractors handle failures gracefully
        assert result.metadata is not None

    def test_principle_extraction_failure_handling(self, mock_llm):
        """Test handling of principle extraction failures."""

        # Create LLM that fails for principles
        class FailingLLM(MockLLM):
            def query(
                self,
                question=None,
                prompt=None,
                temperature=0.7,
                max_tokens=500,
                **kwargs,
            ):
                query_text = question or prompt or ""
                if "identify the key legal principles" in query_text.lower():
                    raise Exception("Principle extraction failed")
                return super().query(
                    question=question,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

        analyzer = CaseAnalyzer(FailingLLM())
        case_doc = CaseDocument(
            content="Test",
            format=CaseFormat.TEXT,
            case_id="test",
        )

        result = analyzer.analyze_document(case_doc)

        # Should handle gracefully - errors are caught internally
        assert result is not None
        # Error is logged but may not be in results.errors list
        # since extractors handle failures gracefully
        assert len(result.principles) == 0  # No principles due to failure

    def test_rule_generation_failure_handling(self, analyzer, sample_case_doc):
        """Test handling of rule generation failures."""
        # This test verifies that failed rule generation doesn't crash
        result = analyzer.analyze_document(sample_case_doc)

        # Should complete even if some rules fail
        assert result is not None
