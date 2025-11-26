"""
Documentation generation for the self-modifying system.

Provides living documentation that automatically generates and maintains
human-readable documentation of the evolving ASP core.

Designed for future LinkedASP integration (see docs/MAINTAINABILITY.md):
- Exports structured metadata for RDF/SPARQL querying
- Supports provenance tracking for LLM-generated rules
- Prepares for genre-based code generation patterns
"""

from loft.documentation.living_document import LivingDocumentGenerator
from loft.documentation.linkedasp_metadata import (
    LinkedASPExporter,
    RuleMetadata,
    ModuleMetadata,
)

__all__ = [
    "LivingDocumentGenerator",
    "LinkedASPExporter",
    "RuleMetadata",
    "ModuleMetadata",
]
