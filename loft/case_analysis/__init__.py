"""
Case analysis and rule extraction module.

Provides tools for parsing legal case documents, extracting principles,
and generating formal rules for the knowledge base.

Issue #276: Case Analysis and Rule Extraction
"""

from loft.case_analysis.parser import CaseDocumentParser, DocumentParseError
from loft.case_analysis.schemas import (
    AnalysisStatistics,
    CaseAnalysisResult,
    CaseDocument,
    CaseFormat,
    CaseMetadata,
    ExtractedRule,
    LegalPrinciple,
)

__all__ = [
    # Parser
    "CaseDocumentParser",
    "DocumentParseError",
    # Schemas
    "CaseDocument",
    "CaseFormat",
    "CaseMetadata",
    "LegalPrinciple",
    "ExtractedRule",
    "CaseAnalysisResult",
    "AnalysisStatistics",
]
