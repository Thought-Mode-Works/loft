"""
LLM-based principle and metadata extraction from case documents.

Uses language models to identify legal principles, extract metadata,
and understand case structure.

Issue #276: Case Analysis and Rule Extraction
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Optional

from loft.case_analysis.schemas import (
    CaseDocument,
    CaseMetadata,
    LegalPrinciple,
)
from loft.neural.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


# Prompts for case analysis
METADATA_EXTRACTION_PROMPT = """You are a legal document analyzer. Extract structured metadata from the following case document.

Case Text:
{case_text}

Extract and return the following information as JSON:
{{
    "title": "Full case name (e.g., 'Smith v. Jones')",
    "citation": "Official citation if present",
    "court": "Court name",
    "jurisdiction": "Jurisdiction (federal, state name, etc.)",
    "date_decided": "Date decided in YYYY-MM-DD format if available",
    "parties_plaintiff": ["List of plaintiffs"],
    "parties_defendant": ["List of defendants"],
    "judges": ["List of judges if mentioned"],
    "legal_citations": ["List of cited cases or statutes"],
    "statutes_cited": ["List of specific statutes cited"],
    "domain": "Legal domain (contracts, torts, property, criminal, constitutional, etc.)",
    "outcome": "Brief outcome (e.g., 'affirmed', 'reversed', 'granted', 'denied')",
    "confidence": 0.0-1.0 score for extraction quality
}}

If any field cannot be determined, use null. Be accurate and conservative with confidence scores.

Return only valid JSON, no other text."""

PRINCIPLE_EXTRACTION_PROMPT = """You are a legal expert analyzing case law. Identify the key legal principles from this case.

Case Information:
Title: {title}
Domain: {domain}
Court: {court}

Case Text:
{case_text}

Identify legal principles that:
1. Are general rules or holdings (not just case-specific facts)
2. Could apply to future cases
3. Represent the legal reasoning or holdings

For EACH principle, provide:
{{
    "principle_text": "Clear statement of the legal principle",
    "domain": "Legal domain (contracts, torts, property, etc.)",
    "source_section": "Which part of opinion (facts, analysis, holding, etc.)",
    "confidence": 0.0-1.0 confidence score,
    "reasoning": "Why this is a legal principle",
    "related_facts": ["Key facts that support this principle"],
    "case_specific": true/false (is this only for this case or a general rule?)
}}

Return a JSON array of principles:
[
    {{"principle_text": "...", "domain": "...", ...}},
    ...
]

Return only valid JSON array, no other text. If no principles found, return empty array []."""

SECTION_IDENTIFICATION_PROMPT = """You are a legal document analyzer. Identify the major sections of this case opinion.

Case Text:
{case_text}

Common sections include:
- Facts / Background
- Procedural History
- Legal Analysis / Discussion
- Holding / Conclusion
- Dissent (if any)

Return JSON mapping section names to their text content:
{{
    "facts": "text of facts section...",
    "analysis": "text of analysis section...",
    "holding": "text of holding/conclusion...",
    ...
}}

Return only valid JSON, no other text."""


class PrincipleExtractor:
    """
    Extracts legal principles from case documents using LLM.

    Uses language models to identify general legal rules and principles
    from case opinions.
    """

    def __init__(
        self,
        llm: LLMInterface,
        min_confidence: float = 0.5,
        max_principles: int = 10,
    ):
        """
        Initialize principle extractor.

        Args:
            llm: Language model interface
            min_confidence: Minimum confidence for principle acceptance
            max_principles: Maximum principles to extract per case
        """
        self.llm = llm
        self.min_confidence = min_confidence
        self.max_principles = max_principles

    def extract_principles(
        self,
        case_doc: CaseDocument,
        metadata: Optional[CaseMetadata] = None,
    ) -> List[LegalPrinciple]:
        """
        Extract legal principles from case document.

        Args:
            case_doc: Case document to analyze
            metadata: Optional pre-extracted metadata

        Returns:
            List of extracted legal principles
        """
        # Truncate very long documents for LLM
        case_text = self._truncate_text(case_doc.content, max_length=8000)

        # Use metadata if available, otherwise use doc fields
        title = (metadata.title if metadata else case_doc.title) or "Unknown Case"
        domain = (metadata.domain if metadata else None) or "general"
        court = (metadata.court if metadata else case_doc.court) or "Unknown Court"

        # Format prompt
        prompt = PRINCIPLE_EXTRACTION_PROMPT.format(
            title=title,
            domain=domain,
            court=court,
            case_text=case_text,
        )

        # Query LLM
        try:
            response = self.llm.query(question=prompt, temperature=0.3, max_tokens=2000)
            principles_data = self._parse_json_response(response.raw_text)

            if not isinstance(principles_data, list):
                logger.warning(
                    f"LLM returned non-list for principles: {type(principles_data)}"
                )
                return []

            # Convert to LegalPrinciple objects
            principles = []
            for data in principles_data[: self.max_principles]:
                try:
                    principle = LegalPrinciple(
                        principle_text=data["principle_text"],
                        domain=data.get("domain", domain),
                        source_section=data.get("source_section", "unknown"),
                        confidence=float(data.get("confidence", 0.5)),
                        reasoning=data.get("reasoning"),
                        related_facts=data.get("related_facts", []),
                        case_specific=data.get("case_specific", False),
                    )

                    # Filter by confidence
                    if principle.confidence >= self.min_confidence:
                        principles.append(principle)
                    else:
                        logger.debug(
                            f"Filtered low-confidence principle: {principle.confidence}"
                        )

                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse principle data: {e}")
                    continue

            logger.info(
                f"Extracted {len(principles)} principles from {case_doc.case_id}"
            )
            return principles

        except Exception as e:
            logger.error(f"Principle extraction failed: {e}")
            return []

    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to maximum length, trying to preserve structure.

        Args:
            text: Text to truncate
            max_length: Maximum character length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Try to truncate at paragraph boundary
        truncated = text[:max_length]
        last_para = truncated.rfind("\n\n")

        if last_para > max_length * 0.8:  # If we can keep >80%, use paragraph boundary
            return truncated[:last_para] + "\n\n[... truncated ...]"
        else:
            return truncated + "\n\n[... truncated ...]"

    def _parse_json_response(self, response_text: str):
        """
        Parse JSON from LLM response.

        Handles common formatting issues.

        Args:
            response_text: LLM response text

        Returns:
            Parsed JSON data
        """
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```"):
            # Remove opening ```json or ```
            text = re.sub(r"^```(?:json)?\s*\n", "", text)
            # Remove closing ```
            text = re.sub(r"\n```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse: {text[:200]}")
            raise


class MetadataExtractor:
    """
    Extracts structured metadata from case documents using LLM.

    Uses language models to identify case metadata like parties, court,
    citation, and domain.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initialize metadata extractor.

        Args:
            llm: Language model interface
        """
        self.llm = llm

    def extract_metadata(self, case_doc: CaseDocument) -> CaseMetadata:
        """
        Extract metadata from case document.

        Args:
            case_doc: Case document to analyze

        Returns:
            Extracted case metadata
        """
        # Start with document fields
        metadata = CaseMetadata(
            case_id=case_doc.case_id,
            title=case_doc.title,
            citation=case_doc.citation,
            court=case_doc.court,
            jurisdiction=case_doc.jurisdiction,
            date_decided=case_doc.date_decided,
        )

        # Use LLM to enhance/extract missing fields
        if not self._is_complete(metadata):
            llm_metadata = self._extract_with_llm(case_doc)
            metadata = self._merge_metadata(metadata, llm_metadata)

        return metadata

    def _is_complete(self, metadata: CaseMetadata) -> bool:
        """Check if metadata has all essential fields."""
        return all(
            [
                metadata.title,
                metadata.court,
                metadata.jurisdiction,
                metadata.domain,
            ]
        )

    def _extract_with_llm(self, case_doc: CaseDocument) -> CaseMetadata:
        """
        Extract metadata using LLM.

        Args:
            case_doc: Case document

        Returns:
            LLM-extracted metadata
        """
        # Truncate text for LLM (focus on header/beginning where metadata usually is)
        case_text = self._truncate_for_metadata(case_doc.content)

        prompt = METADATA_EXTRACTION_PROMPT.format(case_text=case_text)

        try:
            response = self.llm.query(question=prompt, temperature=0.1, max_tokens=1000)
            data = self._parse_json_response(response.raw_text)

            # Parse date if present
            date_decided = None
            if data.get("date_decided"):
                try:
                    date_decided = datetime.fromisoformat(data["date_decided"])
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse date: {data['date_decided']}")

            return CaseMetadata(
                case_id=case_doc.case_id,
                title=data.get("title"),
                citation=data.get("citation"),
                court=data.get("court"),
                jurisdiction=data.get("jurisdiction"),
                date_decided=date_decided,
                parties_plaintiff=data.get("parties_plaintiff", []),
                parties_defendant=data.get("parties_defendant", []),
                judges=data.get("judges", []),
                legal_citations=data.get("legal_citations", []),
                statutes_cited=data.get("statutes_cited", []),
                domain=data.get("domain"),
                outcome=data.get("outcome"),
                confidence=float(data.get("confidence", 0.5)),
            )

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            # Return minimal metadata
            return CaseMetadata(case_id=case_doc.case_id, confidence=0.0)

    def _truncate_for_metadata(self, text: str, max_length: int = 3000) -> str:
        """
        Truncate text for metadata extraction.

        Takes beginning and end of document where metadata usually appears.

        Args:
            text: Full case text
            max_length: Maximum length

        Returns:
            Truncated text focused on metadata sections
        """
        if len(text) <= max_length:
            return text

        # Take first 70% from beginning, 30% from end
        beginning_len = int(max_length * 0.7)
        end_len = int(max_length * 0.3)

        beginning = text[:beginning_len]
        end = text[-end_len:]

        return beginning + "\n\n[... middle content omitted ...]\n\n" + end

    def _merge_metadata(
        self, doc_metadata: CaseMetadata, llm_metadata: CaseMetadata
    ) -> CaseMetadata:
        """
        Merge document and LLM metadata, preferring non-null values.

        Args:
            doc_metadata: Metadata from document
            llm_metadata: Metadata from LLM

        Returns:
            Merged metadata
        """
        return CaseMetadata(
            case_id=doc_metadata.case_id or llm_metadata.case_id,
            title=doc_metadata.title or llm_metadata.title,
            citation=doc_metadata.citation or llm_metadata.citation,
            court=doc_metadata.court or llm_metadata.court,
            jurisdiction=doc_metadata.jurisdiction or llm_metadata.jurisdiction,
            date_decided=doc_metadata.date_decided or llm_metadata.date_decided,
            parties_plaintiff=(
                doc_metadata.parties_plaintiff or llm_metadata.parties_plaintiff
            ),
            parties_defendant=(
                doc_metadata.parties_defendant or llm_metadata.parties_defendant
            ),
            judges=doc_metadata.judges or llm_metadata.judges,
            legal_citations=doc_metadata.legal_citations
            or llm_metadata.legal_citations,
            statutes_cited=doc_metadata.statutes_cited or llm_metadata.statutes_cited,
            domain=doc_metadata.domain or llm_metadata.domain,
            outcome=doc_metadata.outcome or llm_metadata.outcome,
            confidence=max(doc_metadata.confidence, llm_metadata.confidence),
        )

    def _parse_json_response(self, response_text: str):
        """Parse JSON from LLM response."""
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n", "", text)
            text = re.sub(r"\n```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse: {text[:200]}")
            raise
