"""
Unit tests for case document parser.

Issue #276: Case Analysis and Rule Extraction
"""

import json

import pytest

from loft.case_analysis.parser import CaseDocumentParser, DocumentParseError
from loft.case_analysis.schemas import CaseFormat


@pytest.fixture
def parser():
    """Create parser instance."""
    return CaseDocumentParser()


@pytest.fixture
def sample_text_file(tmp_path):
    """Create sample text file."""
    file_path = tmp_path / "sample_case.txt"
    content = """
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
    """
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_json_file(tmp_path):
    """Create sample JSON file."""
    file_path = tmp_path / "sample_case.json"
    data = {
        "id": "case-123",
        "title": "Smith v. Jones",
        "citation": "123 U.S. 456",
        "court": "Supreme Court",
        "jurisdiction": "federal",
        "content": "Full case opinion text here...",
    }
    file_path.write_text(json.dumps(data))
    return file_path


class TestCaseDocumentParser:
    """Test case document parser."""

    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser is not None
        assert isinstance(parser, CaseDocumentParser)

    def test_parse_text_file(self, parser, sample_text_file):
        """Test parsing text file."""
        doc = parser.parse_file(sample_text_file)

        assert doc is not None
        assert doc.format == CaseFormat.TEXT
        assert doc.case_id == "sample_case"
        assert "Smith v. Jones" in doc.content
        assert doc.source_file == sample_text_file

    def test_parse_json_file(self, parser, sample_json_file):
        """Test parsing JSON file."""
        doc = parser.parse_file(sample_json_file)

        assert doc is not None
        assert doc.format == CaseFormat.JSON
        assert doc.case_id == "case-123"
        assert doc.title == "Smith v. Jones"
        assert doc.citation == "123 U.S. 456"
        assert doc.court == "Supreme Court"
        assert doc.jurisdiction == "federal"
        assert "Full case opinion text" in doc.content

    def test_parse_raw_text(self, parser):
        """Test parsing raw text string."""
        text = "This is a case document."
        doc = parser.parse_text(text, case_id="test-case")

        assert doc is not None
        assert doc.format == CaseFormat.TEXT
        assert doc.case_id == "test-case"
        assert doc.content == text

    def test_parse_json_dict(self, parser):
        """Test parsing JSON dictionary."""
        data = {
            "id": "dict-case",
            "title": "Test Case",
            "content": "Case content here",
        }
        doc = parser.parse_json_dict(data)

        assert doc is not None
        assert doc.format == CaseFormat.JSON
        assert doc.case_id == "dict-case"
        assert doc.title == "Test Case"
        assert doc.content == "Case content here"

    def test_parse_nonexistent_file(self, parser, tmp_path):
        """Test parsing nonexistent file raises error."""
        nonexistent = tmp_path / "does_not_exist.txt"

        with pytest.raises(DocumentParseError, match="File not found"):
            parser.parse_file(nonexistent)

    def test_parse_unsupported_format(self, parser, tmp_path):
        """Test parsing unsupported file format raises error."""
        unsupported = tmp_path / "document.docx"
        unsupported.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_file(unsupported)

    def test_parse_json_with_alternative_fields(self, parser):
        """Test JSON parsing with alternative field names."""
        # Test "text" field
        data1 = {"text": "Content via text field"}
        doc1 = parser.parse_json_dict(data1)
        assert doc1.content == "Content via text field"

        # Test "opinion" field
        data2 = {"opinion": "Content via opinion field"}
        doc2 = parser.parse_json_dict(data2)
        assert doc2.content == "Content via opinion field"

    def test_parse_json_missing_content(self, parser):
        """Test JSON parsing with no content field raises error."""
        data = {"id": "test", "title": "Test"}

        with pytest.raises(DocumentParseError, match="no content field"):
            parser.parse_json_dict(data)

    def test_html_parsing_not_available(self, parser, tmp_path):
        """Test HTML parsing when BeautifulSoup not available."""
        html_file = tmp_path / "case.html"
        html_file.write_text("<html><body>Case content</body></html>")

        if not parser._html_available:
            with pytest.raises(DocumentParseError, match="HTML parsing not available"):
                parser.parse_file(html_file)
        else:
            # If BS4 is installed, parsing should work
            doc = parser.parse_file(html_file)
            assert doc is not None
            assert doc.format == CaseFormat.HTML

    def test_pdf_parsing_not_available(self, parser, tmp_path):
        """Test PDF parsing when pypdf not available."""
        # Create dummy PDF file (won't be valid PDF)
        pdf_file = tmp_path / "case.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n")

        if not parser._pdf_available:
            with pytest.raises(DocumentParseError, match="PDF parsing not available"):
                parser.parse_file(pdf_file)
        # Note: If pypdf IS available, this will fail because it's not a valid PDF
        # That's expected and not testing the "not available" scenario

    def test_case_document_string_representation(self, parser, sample_text_file):
        """Test CaseDocument string representation."""
        doc = parser.parse_file(sample_text_file)
        doc_str = str(doc)

        assert "CaseDocument" in doc_str
        assert doc.case_id in doc_str
        assert "text" in doc_str.lower()

    def test_parse_html_file_if_available(self, parser, tmp_path):
        """Test HTML file parsing if BeautifulSoup is available."""
        if not parser._html_available:
            pytest.skip("BeautifulSoup not available")

        html_file = tmp_path / "case.html"
        html_content = """
        <html>
        <head><title>Smith v. Jones</title></head>
        <body>
            <h1>Supreme Court Decision</h1>
            <article>
                <p>This is the case opinion text.</p>
                <p>Multiple paragraphs of legal reasoning.</p>
            </article>
        </body>
        </html>
        """
        html_file.write_text(html_content)

        doc = parser.parse_file(html_file)

        assert doc is not None
        assert doc.format == CaseFormat.HTML
        assert doc.case_id == "case"
        assert doc.title == "Smith v. Jones"
        assert "case opinion text" in doc.content.lower()

    def test_parse_pdf_file_if_available(self, parser, tmp_path):
        """Test PDF file parsing if pypdf is available."""
        if not parser._pdf_available:
            pytest.skip("pypdf not available")

        # This test would require creating a valid PDF
        # For now, just verify the method exists
        assert hasattr(parser, "_parse_pdf")

    def test_case_id_from_filename(self, parser, tmp_path):
        """Test case ID extraction from filename."""
        file_path = tmp_path / "brown_v_board.txt"
        file_path.write_text("Case content")

        doc = parser.parse_file(file_path)
        assert doc.case_id == "brown_v_board"

    def test_json_metadata_preservation(self, parser):
        """Test that JSON metadata is preserved in document."""
        data = {
            "id": "test-case",
            "content": "Case text",
            "extra_field": "Extra data",
            "custom_metadata": {"key": "value"},
        }

        doc = parser.parse_json_dict(data)

        assert doc.metadata == data
        assert doc.metadata["extra_field"] == "Extra data"
