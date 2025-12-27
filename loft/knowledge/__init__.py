"""
Legal knowledge database for persistent rule storage and retrieval.

This module provides a persistent knowledge base for storing ASP rules
with rich metadata, enabling continuous knowledge accumulation and
efficient retrieval for legal question answering.

Issue #271: Persistent Legal Knowledge Database
Epic #270: Legal Reasoning Accumulation Engine
"""

from loft.knowledge.database import (
    KnowledgeDatabase,
    LegalQuestion,
    LegalRule,
    RuleVersion,
)
from loft.knowledge.schemas import (
    KnowledgeCoverageStats,
    RuleCreate,
    RuleFilter,
    RuleUpdate,
)

__all__ = [
    "KnowledgeDatabase",
    "LegalRule",
    "LegalQuestion",
    "RuleVersion",
    "RuleCreate",
    "RuleUpdate",
    "RuleFilter",
    "KnowledgeCoverageStats",
]
