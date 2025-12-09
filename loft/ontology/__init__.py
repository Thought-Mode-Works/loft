"""
Ontology module for LOFT canonical predicate vocabulary.

This module provides RDF-based ontology for cross-domain predicate translation,
enabling deterministic mapping between domain-specific predicates and canonical
legal concepts.

Includes hybrid translation combining canonical mappings with LLM fallback
for predicates that lack explicit ontology mappings.
"""

from loft.ontology.canonical_translator import CanonicalTranslator
from loft.ontology.hybrid_translator import (
    HybridTranslator,
    TranslationResult,
    TranslationStats,
)

__all__ = [
    "CanonicalTranslator",
    "HybridTranslator",
    "TranslationResult",
    "TranslationStats",
]
