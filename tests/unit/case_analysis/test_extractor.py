"""
Unit tests for principle and metadata extractors.

Issue #276: Case Analysis and Rule Extraction
"""

import json

import pytest

from loft.case_analysis.extractor import MetadataExtractor, PrincipleExtractor
from loft.case_analysis.schemas import CaseDocument, CaseFormat, CaseMetadata


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response_text: str = None):
        """Initialize with optional fixed response."""
        self.response_text = response_text
        self.queries = []  # Track queries made

    def query(
        self, question: str, temperature: float = 0.7, max_tokens: int = 500, **kwargs
    ):
        """Mock query method."""
        self.queries.append(
            {"question": question, "temperature": temperature, "max_tokens": max_tokens}
        )

        # Simple response object with raw_text attribute
        class MockResponse:
            def __init__(self, raw_text):
                self.raw_text = raw_text

        if self.response_text:
            return MockResponse(self.response_text)

        # Default responses based on question content
        if "Extract and return the following information as JSON" in question:
            # Metadata extraction
            return MockResponse(
                json.dumps(
                    {
                        "title": "Smith v. Jones",
                        "citation": "123 U.S. 456",
                        "court": "Supreme Court",
                        "jurisdiction": "federal",
                        "date_decided": "2023-01-15",
                        "parties_plaintiff": ["Smith"],
                        "parties_defendant": ["Jones"],
                        "judges": ["Chief Justice Roberts"],
                        "legal_citations": ["Marbury v. Madison"],
                        "statutes_cited": ["28 U.S.C. § 1331"],
                        "domain": "contracts",
                        "outcome": "affirmed",
                        "confidence": 0.9,
                    }
                )
            )
        elif (
            "Identify legal principles" in question
            or "identify the key legal principles" in question.lower()
        ):
            # Principle extraction
            return MockResponse(
                json.dumps(
                    [
                        {
                            "principle_text": "A valid contract requires offer, acceptance, and consideration.",
                            "domain": "contracts",
                            "source_section": "analysis",
                            "confidence": 0.95,
                            "reasoning": "This is a fundamental contract law principle",
                            "related_facts": [
                                "parties exchanged promises",
                                "consideration was present",
                            ],
                            "case_specific": False,
                        },
                        {
                            "principle_text": "Past consideration is generally insufficient.",
                            "domain": "contracts",
                            "source_section": "holding",
                            "confidence": 0.85,
                            "reasoning": "Well-established contract doctrine",
                            "related_facts": ["promise made after performance"],
                            "case_specific": False,
                        },
                    ]
                )
            )
        else:
            return MockResponse("{}")


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    return MockLLM()


@pytest.fixture
def sample_case_doc():
    """Create sample case document."""
    return CaseDocument(
        content="""
        SUPREME COURT OF THE UNITED STATES

        Smith v. Jones
        No. 12-345
        Decided: January 15, 2023

        FACTS:
        The plaintiff, Smith, entered into a contract with the defendant, Jones.

        ANALYSIS:
        The court finds that a valid contract requires offer, acceptance, and consideration.

        HOLDING:
        We hold that the contract is enforceable.
        """,
        format=CaseFormat.TEXT,
        case_id="smith_v_jones",
        title="Smith v. Jones",
    )


class TestPrincipleExtractor:
    """Test principle extractor."""

    def test_extractor_initialization(self, mock_llm):
        """Test extractor initializes correctly."""
        extractor = PrincipleExtractor(mock_llm, min_confidence=0.6, max_principles=5)

        assert extractor.llm == mock_llm
        assert extractor.min_confidence == 0.6
        assert extractor.max_principles == 5

    def test_extract_principles(self, mock_llm, sample_case_doc):
        """Test principle extraction from case."""
        extractor = PrincipleExtractor(mock_llm, min_confidence=0.5)
        principles = extractor.extract_principles(sample_case_doc)

        assert len(principles) == 2
        assert principles[0].principle_text.startswith("A valid contract")
        assert principles[0].domain == "contracts"
        assert principles[0].confidence == 0.95
        assert not principles[0].case_specific

    def test_extract_principles_with_metadata(self, mock_llm, sample_case_doc):
        """Test principle extraction with metadata."""
        extractor = PrincipleExtractor(mock_llm)
        metadata = CaseMetadata(
            case_id="smith_v_jones",
            title="Smith v. Jones",
            court="Supreme Court",
            domain="contracts",
        )

        principles = extractor.extract_principles(sample_case_doc, metadata)

        assert len(principles) >= 1
        # Should use metadata domain
        assert all(p.domain == "contracts" for p in principles)

    def test_filter_low_confidence_principles(self, mock_llm):
        """Test filtering of low-confidence principles."""
        # Set up LLM to return principles with varying confidence
        response_data = [
            {
                "principle_text": "High confidence principle",
                "domain": "contracts",
                "source_section": "analysis",
                "confidence": 0.9,
                "case_specific": False,
            },
            {
                "principle_text": "Low confidence principle",
                "domain": "contracts",
                "source_section": "analysis",
                "confidence": 0.3,
                "case_specific": False,
            },
        ]
        mock_llm.response_text = json.dumps(response_data)

        extractor = PrincipleExtractor(mock_llm, min_confidence=0.5)
        case_doc = CaseDocument(
            content="Test content", format=CaseFormat.TEXT, case_id="test"
        )

        principles = extractor.extract_principles(case_doc)

        # Should only get high-confidence principle
        assert len(principles) == 1
        assert principles[0].confidence == 0.9

    def test_max_principles_limit(self, mock_llm):
        """Test maximum principles limit."""
        # Create response with many principles
        many_principles = [
            {
                "principle_text": f"Principle {i}",
                "domain": "test",
                "source_section": "analysis",
                "confidence": 0.8,
                "case_specific": False,
            }
            for i in range(20)
        ]
        mock_llm.response_text = json.dumps(many_principles)

        extractor = PrincipleExtractor(mock_llm, max_principles=5)
        case_doc = CaseDocument(
            content="Test content", format=CaseFormat.TEXT, case_id="test"
        )

        principles = extractor.extract_principles(case_doc)

        # Should only get max_principles
        assert len(principles) == 5

    def test_text_truncation(self, mock_llm):
        """Test that very long text is truncated."""
        extractor = PrincipleExtractor(mock_llm)

        # Create very long text
        long_text = "Long content. " * 10000  # Much longer than 8000 chars

        truncated = extractor._truncate_text(long_text, max_length=1000)

        assert len(truncated) <= 1100  # Allow some buffer for truncation marker
        assert "truncated" in truncated.lower()

    def test_parse_json_with_markdown(self, mock_llm):
        """Test parsing JSON response with markdown code blocks."""
        extractor = PrincipleExtractor(mock_llm)

        # Response with markdown formatting
        response_with_markdown = """```json
{
    "test": "value"
}
```"""

        result = extractor._parse_json_response(response_with_markdown)
        assert result == {"test": "value"}

    def test_invalid_principle_data_handling(self, mock_llm):
        """Test handling of invalid principle data."""
        # Return malformed principle data
        mock_llm.response_text = json.dumps(
            [
                {
                    "principle_text": "Valid principle",
                    "domain": "test",
                    "source_section": "test",
                    "confidence": 0.8,
                },
                {"missing_required_fields": "invalid"},  # Missing required fields
                {
                    "principle_text": "Another valid",
                    "domain": "test",
                    "source_section": "test",
                    "confidence": 0.7,
                },
            ]
        )

        extractor = PrincipleExtractor(mock_llm, min_confidence=0.5)
        case_doc = CaseDocument(content="Test", format=CaseFormat.TEXT, case_id="test")

        principles = extractor.extract_principles(case_doc)

        # Should only get valid principles
        assert len(principles) == 2


class TestMetadataExtractor:
    """Test metadata extractor."""

    def test_extractor_initialization(self, mock_llm):
        """Test extractor initializes correctly."""
        extractor = MetadataExtractor(mock_llm)
        assert extractor.llm == mock_llm

    def test_extract_metadata(self, mock_llm, sample_case_doc):
        """Test metadata extraction."""
        extractor = MetadataExtractor(mock_llm)
        metadata = extractor.extract_metadata(sample_case_doc)

        assert metadata is not None
        assert metadata.case_id == "smith_v_jones"
        assert metadata.title == "Smith v. Jones"
        assert metadata.court == "Supreme Court"
        assert metadata.jurisdiction == "federal"
        assert metadata.domain == "contracts"

    def test_metadata_from_document_fields(self, mock_llm):
        """Test using existing document fields without LLM."""
        case_doc = CaseDocument(
            content="Short content",
            format=CaseFormat.TEXT,
            case_id="test-case",
            title="Test v. Case",
            court="Test Court",
            jurisdiction="test",
        )

        # Create metadata without complete info to trigger LLM
        extractor = MetadataExtractor(mock_llm)
        metadata = extractor.extract_metadata(case_doc)

        # Should use document fields as base
        assert metadata.case_id == "test-case"
        assert metadata.title == "Test v. Case"

    def test_metadata_merge(self, mock_llm):
        """Test merging document and LLM metadata."""
        extractor = MetadataExtractor(mock_llm)

        # Document has some fields
        doc_metadata = CaseMetadata(
            case_id="test-123",
            title="Doc Title",
            court=None,  # Missing
            jurisdiction=None,  # Missing
        )

        # LLM provides others
        llm_metadata = CaseMetadata(
            case_id="test-123",
            title=None,
            court="LLM Court",
            jurisdiction="federal",
        )

        merged = extractor._merge_metadata(doc_metadata, llm_metadata)

        # Should have fields from both
        assert merged.case_id == "test-123"
        assert merged.title == "Doc Title"  # From doc
        assert merged.court == "LLM Court"  # From LLM
        assert merged.jurisdiction == "federal"  # From LLM

    def test_text_truncation_for_metadata(self, mock_llm):
        """Test text truncation for metadata extraction."""
        extractor = MetadataExtractor(mock_llm)

        long_text = "A" * 10000
        truncated = extractor._truncate_for_metadata(long_text, max_length=1000)

        assert len(truncated) <= 1100  # Some buffer for markers
        assert "omitted" in truncated.lower() or len(truncated) <= 1000

    def test_parse_json_response(self, mock_llm):
        """Test JSON response parsing."""
        extractor = MetadataExtractor(mock_llm)

        # Valid JSON
        valid_json = '{"key": "value"}'
        result = extractor._parse_json_response(valid_json)
        assert result == {"key": "value"}

        # JSON with markdown
        markdown_json = '```json\n{"key": "value"}\n```'
        result = extractor._parse_json_response(markdown_json)
        assert result == {"key": "value"}

    def test_date_parsing(self, mock_llm, sample_case_doc):
        """Test date parsing in metadata extraction."""
        extractor = MetadataExtractor(mock_llm)
        metadata = extractor.extract_metadata(sample_case_doc)

        # Mock returns date in ISO format
        if metadata.date_decided:
            assert metadata.date_decided.year == 2023
            assert metadata.date_decided.month == 1
            assert metadata.date_decided.day == 15

    def test_incomplete_metadata_triggers_llm(self, mock_llm):
        """Test that incomplete metadata triggers LLM extraction."""
        case_doc = CaseDocument(
            content="Case content",
            format=CaseFormat.TEXT,
            case_id="test",
            title=None,  # Missing fields
            court=None,
        )

        extractor = MetadataExtractor(mock_llm)
        metadata = extractor.extract_metadata(case_doc)

        # Should have called LLM
        assert len(mock_llm.queries) > 0
        # Should have filled in missing fields
        assert metadata.title == "Smith v. Jones"  # From mock LLM
