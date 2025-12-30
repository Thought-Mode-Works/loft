"""
Case document parser for multiple formats.

Parses legal case documents from various formats (PDF, text, JSON, HTML)
and creates CaseDocument instances for analysis.

Issue #276: Case Analysis and Rule Extraction
"""

import importlib.util
import logging
from pathlib import Path
from typing import Optional

from loft.case_analysis.schemas import CaseDocument, CaseFormat

logger = logging.getLogger(__name__)


class DocumentParseError(Exception):
    """Raised when document parsing fails."""

    pass


class CaseDocumentParser:
    """
    Parser for legal case documents in multiple formats.

    Supports:
    - Plain text files (.txt)
    - PDF documents (.pdf)
    - JSON files (.json)
    - HTML files (.html)
    """

    def __init__(self):
        """Initialize parser."""
        self._pdf_available = self._check_pdf_support()
        self._html_available = self._check_html_support()

    def _check_pdf_support(self) -> bool:
        """Check if PDF parsing is available."""
        if importlib.util.find_spec("pypdf") is not None:
            return True
        else:
            logger.warning(
                "PyPDF not available. PDF parsing will be disabled. "
                "Install with: pip install pypdf"
            )
            return False

    def _check_html_support(self) -> bool:
        """Check if HTML parsing is available."""
        if importlib.util.find_spec("bs4") is not None:
            return True
        else:
            logger.warning(
                "BeautifulSoup not available. HTML parsing will be disabled. "
                "Install with: pip install beautifulsoup4"
            )
            return False

    def parse_file(self, file_path: Path) -> CaseDocument:
        """
        Parse a case document file.

        Automatically detects format from extension.

        Args:
            file_path: Path to case document

        Returns:
            Parsed CaseDocument

        Raises:
            DocumentParseError: If parsing fails
            ValueError: If file format is not supported
        """
        if not file_path.exists():
            raise DocumentParseError(f"File not found: {file_path}")

        # Detect format from extension
        ext = file_path.suffix.lower()
        format_map = {
            ".txt": CaseFormat.TEXT,
            ".pdf": CaseFormat.PDF,
            ".json": CaseFormat.JSON,
            ".html": CaseFormat.HTML,
            ".htm": CaseFormat.HTML,
        }

        if ext not in format_map:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {list(format_map.keys())}"
            )

        case_format = format_map[ext]

        # Parse based on format
        try:
            if case_format == CaseFormat.TEXT:
                return self._parse_text(file_path)
            elif case_format == CaseFormat.PDF:
                return self._parse_pdf(file_path)
            elif case_format == CaseFormat.JSON:
                return self._parse_json(file_path)
            elif case_format == CaseFormat.HTML:
                return self._parse_html(file_path)
            else:
                raise ValueError(f"Unsupported format: {case_format}")
        except Exception as e:
            raise DocumentParseError(f"Failed to parse {file_path}: {e}") from e

    def _parse_text(self, file_path: Path) -> CaseDocument:
        """
        Parse plain text file.

        Args:
            file_path: Path to text file

        Returns:
            CaseDocument with text content
        """
        return CaseDocument.from_text_file(file_path)

    def _parse_json(self, file_path: Path) -> CaseDocument:
        """
        Parse JSON file.

        Expected JSON structure:
        {
            "content": "full case text" or "text": "..." or "opinion": "...",
            "id": "case id",
            "title": "case name",
            "citation": "official citation",
            "court": "court name",
            "jurisdiction": "federal/state",
            ...
        }

        Args:
            file_path: Path to JSON file

        Returns:
            CaseDocument with extracted fields
        """
        return CaseDocument.from_json_file(file_path)

    def _parse_pdf(self, file_path: Path) -> CaseDocument:
        """
        Parse PDF file.

        Extracts text from all pages and combines.

        Args:
            file_path: Path to PDF file

        Returns:
            CaseDocument with extracted text

        Raises:
            DocumentParseError: If PDF support is not available
        """
        if not self._pdf_available:
            raise DocumentParseError(
                "PDF parsing not available. Install pypdf: pip install pypdf"
            )

        import pypdf

        try:
            reader = pypdf.PdfReader(str(file_path))
            pages = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

            content = "\n\n".join(pages)

            if not content.strip():
                raise DocumentParseError("PDF contains no extractable text")

            # Try to extract metadata from PDF
            metadata = {}
            if reader.metadata:
                metadata = {
                    "pdf_title": reader.metadata.get("/Title"),
                    "pdf_author": reader.metadata.get("/Author"),
                    "pdf_subject": reader.metadata.get("/Subject"),
                    "pdf_creator": reader.metadata.get("/Creator"),
                    "pdf_pages": len(reader.pages),
                }
                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}

            return CaseDocument(
                content=content,
                format=CaseFormat.PDF,
                case_id=file_path.stem,
                title=metadata.get("pdf_title"),
                source_file=file_path,
                metadata=metadata,
            )

        except Exception as e:
            raise DocumentParseError(f"Failed to parse PDF: {e}") from e

    def _parse_html(self, file_path: Path) -> CaseDocument:
        """
        Parse HTML file.

        Extracts text content from HTML, stripping tags.

        Args:
            file_path: Path to HTML file

        Returns:
            CaseDocument with extracted text

        Raises:
            DocumentParseError: If HTML support is not available
        """
        if not self._html_available:
            raise DocumentParseError(
                "HTML parsing not available. "
                "Install beautifulsoup4: pip install beautifulsoup4"
            )

        from bs4 import BeautifulSoup

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Try to extract title from HTML
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()

            # Try to find main content
            # Look for common case law HTML structures
            main_content = None
            for tag in [
                "article",
                "main",
                "div[class*='opinion']",
                "div[class*='case']",
            ]:
                content_elem = soup.select_one(tag)
                if content_elem:
                    main_content = content_elem
                    break

            # If no main content found, use body
            if main_content is None:
                main_content = soup.find("body")

            # If still no content, use whole document
            if main_content is None:
                main_content = soup

            # Extract text
            content = main_content.get_text(separator="\n", strip=True)

            if not content.strip():
                raise DocumentParseError("HTML contains no extractable text")

            # Try to extract metadata from HTML meta tags
            metadata = {}
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property")
                content_attr = meta.get("content")
                if name and content_attr:
                    metadata[f"html_{name}"] = content_attr

            return CaseDocument(
                content=content,
                format=CaseFormat.HTML,
                case_id=file_path.stem,
                title=title,
                source_file=file_path,
                metadata=metadata,
            )

        except Exception as e:
            raise DocumentParseError(f"Failed to parse HTML: {e}") from e

    def parse_text(self, text: str, case_id: Optional[str] = None) -> CaseDocument:
        """
        Parse raw text content.

        Args:
            text: Raw case text
            case_id: Optional case identifier

        Returns:
            CaseDocument with text content
        """
        return CaseDocument(
            content=text,
            format=CaseFormat.TEXT,
            case_id=case_id or "unknown",
        )

    def parse_json_dict(
        self, data: dict, case_id: Optional[str] = None
    ) -> CaseDocument:
        """
        Parse JSON data as dictionary.

        Args:
            data: Dictionary with case data
            case_id: Optional case identifier override

        Returns:
            CaseDocument with extracted fields
        """
        # Extract content from various possible fields
        content = data.get("content") or data.get("text") or data.get("opinion", "")

        if not content:
            raise DocumentParseError("JSON data contains no content field")

        return CaseDocument(
            content=content,
            format=CaseFormat.JSON,
            case_id=case_id or data.get("id") or data.get("case_id"),
            title=data.get("title") or data.get("case_name"),
            citation=data.get("citation"),
            court=data.get("court"),
            jurisdiction=data.get("jurisdiction"),
            metadata=data,
        )
