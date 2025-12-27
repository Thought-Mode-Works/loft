"""
Legal Question Answering Interface.

Provides natural language question answering using accumulated ASP rules.

Issue #272: Legal Question Answering Interface
"""

from loft.qa.interface import LegalQAInterface
from loft.qa.question_parser import LegalQuestionParser
from loft.qa.reasoner import LegalReasoner
from loft.qa.schemas import Answer, ASPQuery, EvaluationReport

__all__ = [
    "LegalQAInterface",
    "LegalQuestionParser",
    "LegalReasoner",
    "Answer",
    "ASPQuery",
    "EvaluationReport",
]
