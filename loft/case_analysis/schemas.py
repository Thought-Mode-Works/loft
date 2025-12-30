"""
Data schemas for case analysis and rule extraction.

Defines data structures for case documents, extracted principles,
generated rules, and analysis results.

Issue #276: Case Analysis and Rule Extraction
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class CaseFormat(Enum):
    """Supported case document formats."""

    TEXT = "text"
    PDF = "pdf"
    JSON = "json"
    HTML = "html"


@dataclass
class CaseDocument:
    """
    A legal case document to be analyzed.

    Represents a raw case document from various sources (PDF, text, etc.)
    before analysis and rule extraction.
    """

    content: str  # Raw text content of the case
    format: CaseFormat  # Format of the document
    case_id: Optional[str] = None  # Identifier for the case
    title: Optional[str] = None  # Case title/name
    citation: Optional[str] = None  # Official citation
    court: Optional[str] = None  # Court that decided the case
    jurisdiction: Optional[str] = None  # Jurisdiction (federal, state)
    date_decided: Optional[datetime] = None  # Date case was decided
    source_file: Optional[Path] = None  # Original file path
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional metadata

    @classmethod
    def from_text_file(cls, file_path: Path) -> "CaseDocument":
        """
        Create a CaseDocument from a text file.

        Args:
            file_path: Path to text file

        Returns:
            CaseDocument instance
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return cls(
            content=content,
            format=CaseFormat.TEXT,
            source_file=file_path,
            case_id=file_path.stem,
        )

    @classmethod
    def from_json_file(cls, file_path: Path) -> "CaseDocument":
        """
        Create a CaseDocument from a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            CaseDocument instance
        """
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract structured fields if available
        content = data.get("content") or data.get("text") or data.get("opinion", "")

        return cls(
            content=content,
            format=CaseFormat.JSON,
            case_id=data.get("id") or data.get("case_id"),
            title=data.get("title") or data.get("case_name"),
            citation=data.get("citation"),
            court=data.get("court"),
            jurisdiction=data.get("jurisdiction"),
            source_file=file_path,
            metadata=data,
        )

    def __str__(self) -> str:
        """String representation."""
        id_str = self.case_id or "unknown"
        format_str = self.format.value
        content_preview = (
            self.content[:100] + "..." if len(self.content) > 100 else self.content
        )
        return f"CaseDocument({id_str}, {format_str}): {content_preview}"


@dataclass
class CaseMetadata:
    """
    Extracted metadata from a case document.

    Contains structured information about the case (parties, court, dates, etc.)
    extracted through parsing or LLM analysis.
    """

    case_id: Optional[str] = None
    title: Optional[str] = None
    citation: Optional[str] = None
    court: Optional[str] = None
    jurisdiction: Optional[str] = None
    date_decided: Optional[datetime] = None
    parties_plaintiff: List[str] = field(default_factory=list)
    parties_defendant: List[str] = field(default_factory=list)
    judges: List[str] = field(default_factory=list)
    legal_citations: List[str] = field(default_factory=list)
    statutes_cited: List[str] = field(default_factory=list)
    domain: Optional[str] = None  # Legal domain (contracts, torts, etc.)
    outcome: Optional[str] = None  # Decision outcome
    confidence: float = 0.0  # Confidence in extraction quality

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "title": self.title,
            "citation": self.citation,
            "court": self.court,
            "jurisdiction": self.jurisdiction,
            "date_decided": (
                self.date_decided.isoformat() if self.date_decided else None
            ),
            "parties_plaintiff": self.parties_plaintiff,
            "parties_defendant": self.parties_defendant,
            "judges": self.judges,
            "legal_citations": self.legal_citations,
            "statutes_cited": self.statutes_cited,
            "domain": self.domain,
            "outcome": self.outcome,
            "confidence": self.confidence,
        }


@dataclass
class LegalPrinciple:
    """
    A legal principle extracted from a case.

    Represents a general legal rule or principle identified in the
    case opinion, before conversion to ASP.
    """

    principle_text: str  # Natural language statement of principle
    domain: str  # Legal domain
    source_section: str  # Section of case where found
    confidence: float  # Confidence in extraction (0.0-1.0)
    reasoning: Optional[str] = None  # Why this was identified as a principle
    related_facts: List[str] = field(default_factory=list)  # Facts that support it
    case_specific: bool = False  # Whether specific to this case or general

    def __str__(self) -> str:
        """String representation."""
        preview = (
            self.principle_text[:80] + "..."
            if len(self.principle_text) > 80
            else self.principle_text
        )
        return f"Principle({self.domain}, conf={self.confidence:.2f}): {preview}"


@dataclass
class ExtractedRule:
    """
    An ASP rule extracted from a legal principle.

    Represents a formal ASP rule generated from a legal principle,
    along with provenance and quality metrics.
    """

    asp_rule: str  # The ASP rule text
    principle: LegalPrinciple  # Source principle
    domain: str  # Legal domain
    confidence: float  # Confidence in rule quality
    reasoning: str  # Explanation of rule
    predicates_used: List[str] = field(default_factory=list)  # ASP predicates in rule
    source_case_id: Optional[str] = None  # ID of source case
    validation_passed: bool = False  # Whether rule passed syntax validation

    def __str__(self) -> str:
        """String representation."""
        rule_preview = (
            self.asp_rule[:60] + "..." if len(self.asp_rule) > 60 else self.asp_rule
        )
        return f"Rule({self.domain}, conf={self.confidence:.2f}): {rule_preview}"


@dataclass
class CaseAnalysisResult:
    """
    Complete analysis result for a case document.

    Contains all extracted principles, generated rules, and metadata
    from analyzing a single case document.
    """

    case_id: str
    metadata: CaseMetadata
    principles: List[LegalPrinciple] = field(default_factory=list)
    rules: List[ExtractedRule] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def principle_count(self) -> int:
        """Number of principles extracted."""
        return len(self.principles)

    @property
    def rule_count(self) -> int:
        """Number of rules generated."""
        return len(self.rules)

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all rules."""
        if not self.rules:
            return 0.0
        return sum(r.confidence for r in self.rules) / len(self.rules)

    @property
    def success(self) -> bool:
        """Whether analysis was successful."""
        return len(self.rules) > 0 and len(self.errors) == 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "case_id": self.case_id,
            "metadata": self.metadata.to_dict(),
            "principle_count": self.principle_count,
            "rule_count": self.rule_count,
            "avg_confidence": self.avg_confidence,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "errors": self.errors,
            "principles": [
                {
                    "text": p.principle_text,
                    "domain": p.domain,
                    "confidence": p.confidence,
                }
                for p in self.principles
            ],
            "rules": [
                {
                    "asp_rule": r.asp_rule,
                    "domain": r.domain,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in self.rules
            ],
        }

    def __str__(self) -> str:
        """String representation."""
        status = "✓" if self.success else "✗"
        return (
            f"CaseAnalysis({self.case_id}) {status}: "
            f"{self.principle_count} principles → {self.rule_count} rules "
            f"(avg conf: {self.avg_confidence:.2f})"
        )


@dataclass
class AnalysisStatistics:
    """
    Statistics about case analysis across multiple cases.

    Tracks aggregate metrics for quality monitoring.
    """

    total_cases: int = 0
    successful_cases: int = 0
    total_principles: int = 0
    total_rules: int = 0
    avg_principles_per_case: float = 0.0
    avg_rules_per_case: float = 0.0
    avg_confidence: float = 0.0
    avg_processing_time_ms: float = 0.0
    domains_covered: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation."""
        success_rate = (
            self.successful_cases / self.total_cases * 100
            if self.total_cases > 0
            else 0
        )
        return (
            f"AnalysisStats: {self.total_cases} cases, "
            f"{success_rate:.1f}% success, "
            f"{self.total_rules} rules extracted"
        )
