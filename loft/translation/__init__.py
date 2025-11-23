"""
Translation: ASP ↔ Natural Language Bridge

This module implements the ontological bridge between symbolic (ASP) and
neural (NL) representations.
"""

from .asp_to_nl import (
    asp_to_nl,
    asp_rule_to_nl,
    asp_facts_to_nl,
    ASPToNLTranslator,
    enrich_context,
    extract_predicates,
    TranslationResult,
)
from .quality import (
    validate_translation_quality,
    compute_fidelity,
    compute_quality_metrics,
    QualityMetrics,
)

__all__ = [
    "asp_to_nl",
    "asp_rule_to_nl",
    "asp_facts_to_nl",
    "ASPToNLTranslator",
    "enrich_context",
    "extract_predicates",
    "TranslationResult",
    "validate_translation_quality",
    "compute_fidelity",
    "compute_quality_metrics",
    "QualityMetrics",
]
