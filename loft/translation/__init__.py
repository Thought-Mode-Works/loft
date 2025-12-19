"""
Translation: ASP ↔ Natural Language Bridge

This module implements the ontological bridge between symbolic (ASP) and
neural (NL) representations.
"""

# ASP → NL translation
from .asp_to_nl import (
    asp_to_nl,
    asp_rule_to_nl,
    asp_facts_to_nl,
    ASPToNLTranslator,
    enrich_context,
    extract_predicates,
    TranslationResult,
)

# NL → ASP translation
from .nl_to_asp import (
    nl_to_structured,
    nl_to_asp_facts,
    nl_to_asp_rule,
    NLToASPTranslator,
    NLToASPResult,
)

# Schemas for structured extraction
from .schemas import (
    ContractFact,
    Party,
    Writing,
    ExtractedEntities,
    LegalRelationship,
    LegalRule,
)

# Pattern matching
from .patterns import (
    pattern_based_extraction,
    quick_extract_facts,
)

# Grounding and validation
from .grounding import (
    ASPGrounder,
    AmbiguityHandler,
    validate_new_facts,
)

# Quality and fidelity metrics
from .quality import (
    validate_translation_quality,
    compute_fidelity,
    compute_quality_metrics,
    compute_asp_equivalence,
    roundtrip_fidelity_test,
    QualityMetrics,
)

# Context-preserving translation
from .context import TranslationContext
from .context_preserving_translator import ContextPreservingTranslator

# Fidelity tracking and pattern documentation
from .fidelity_tracker import (
    FidelityTracker,
    FidelitySnapshot,
    TranslationResult as FidelityTranslationResult,
    Regression,
)
from .pattern_documenter import (
    TranslationPatternDocumenter,
    TranslationPattern,
    PatternAnalysis,
)

__all__ = [
    # ASP → NL
    "asp_to_nl",
    "asp_rule_to_nl",
    "asp_facts_to_nl",
    "ASPToNLTranslator",
    "enrich_context",
    "extract_predicates",
    "TranslationResult",
    # NL → ASP
    "nl_to_structured",
    "nl_to_asp_facts",
    "nl_to_asp_rule",
    "NLToASPTranslator",
    "NLToASPResult",
    # Context-preserving translation
    "TranslationContext",
    "ContextPreservingTranslator",
    # Schemas
    "ContractFact",
    "Party",
    "Writing",
    "ExtractedEntities",
    "LegalRelationship",
    "LegalRule",
    # Patterns
    "pattern_based_extraction",
    "quick_extract_facts",
    # Grounding
    "ASPGrounder",
    "AmbiguityHandler",
    "validate_new_facts",
    # Quality
    "validate_translation_quality",
    "compute_fidelity",
    "compute_quality_metrics",
    "compute_asp_equivalence",
    "roundtrip_fidelity_test",
    "QualityMetrics",
    # Fidelity tracking and pattern documentation
    "FidelityTracker",
    "FidelitySnapshot",
    "FidelityTranslationResult",
    "Regression",
    "TranslationPatternDocumenter",
    "TranslationPattern",
    "PatternAnalysis",
]
