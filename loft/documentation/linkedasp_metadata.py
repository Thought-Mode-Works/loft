"""
LinkedASP Metadata Export for Living Documents.

Provides structured metadata export compatible with the future LinkedASP
RDF-based queryable documentation system described in docs/MAINTAINABILITY.md.

This module prepares the living document system for seamless integration with:
- RDF/Turtle metadata for ASP rules
- SPARQL queries for dependency analysis
- Genre-based code generation
- Provenance tracking ontology
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class RuleMetadata:
    """
    Metadata for an ASP rule compatible with LinkedASP ontology.

    Maps to the loft:ASPRule class in the LinkedASP ontology.
    Designed for future RDF export and SPARQL querying.
    """

    # Core identifiers
    rule_id: str
    rule_text: str
    predicate_name: str

    # Stratification (loft:stratificationLevel)
    stratification_level: str  # constitutional, strategic, tactical, operational

    # Genre classification (loft:hasGenre)
    genre: Optional[str] = None  # e.g., "DisjunctiveRequirement", "Exception"

    # Legal sources (loft:legalSource, loft:jurisdiction)
    legal_source: Optional[str] = None
    jurisdiction: Optional[str] = None

    # Confidence and validation (loft:confidence, loft:successRate)
    confidence: float = 1.0
    success_rate: Optional[float] = None
    validated_by: List[str] = field(default_factory=list)

    # Provenance tracking (loft:sourceType, loft:sourceLLM, etc.)
    source_type: str = "manual"  # manual, llm_generated, gap_fill, case_law
    source_llm: Optional[str] = None  # e.g., "anthropic/claude-3-opus"
    source_prompt_version: Optional[str] = None
    source_text: Optional[str] = None

    # Timestamps (loft:generationTimestamp, loft:incorporationTimestamp)
    generation_timestamp: Optional[datetime] = None
    incorporation_timestamp: Optional[datetime] = None
    last_modified: Optional[datetime] = None

    # Incorporation tracking (loft:incorporatedBy, loft:validationReportId)
    incorporated_by: str = "system"  # human, autonomous_system, reviewed_by_human
    validation_report_id: Optional[str] = None
    snapshot_id: Optional[str] = None

    # Dependencies and relationships
    requires_elements: List[str] = field(default_factory=list)  # loft:requiresElement
    has_alternatives: List[str] = field(default_factory=list)  # loft:hasAlternative
    applies_to_requirement: Optional[str] = None  # loft:appliesToRequirement
    replaced_rule: Optional[str] = None  # loft:replacedRule
    replaced_by: Optional[str] = None  # loft:replacedBy

    # Module information
    module_name: Optional[str] = None
    domain: Optional[str] = None  # e.g., "contract_law", "tort_law"
    phase: Optional[str] = None  # ROADMAP.md phase

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key in ["generation_timestamp", "incorporation_timestamp", "last_modified"]:
            if data[key]:
                data[key] = data[key].isoformat()
        return data

    def to_turtle(self) -> str:
        """
        Generate RDF Turtle representation for future LinkedASP integration.

        Returns a Turtle snippet that can be embedded in ASP comments
        or exported to a separate RDF file.
        """
        lines = [
            "@prefix loft: <https://loft.legal/ontology/> .",
            "@prefix asp: <https://loft.legal/asp/> .",
            "@prefix legal: <https://loft.legal/patterns/> .",
            "",
            f"asp:{self.predicate_name} a asp:ASPRule ;",
            f'    rdfs:label "{self.rule_id}" ;',
            f'    loft:stratificationLevel "{self.stratification_level}" ;',
        ]

        if self.genre:
            lines.append(f"    loft:hasGenre legal:{self.genre} ;")

        if self.legal_source:
            lines.append(f'    loft:legalSource "{self.legal_source}" ;')

        if self.jurisdiction:
            lines.append(f'    loft:jurisdiction "{self.jurisdiction}" ;')

        lines.append(f"    loft:confidence {self.confidence} ;")

        if self.source_type:
            lines.append(f'    loft:sourceType "{self.source_type}" ;')

        if self.source_llm:
            lines.append(f'    loft:sourceLLM "{self.source_llm}" ;')

        if self.incorporated_by:
            lines.append(f'    loft:incorporatedBy "{self.incorporated_by}" ;')

        # Add dependencies
        for req in self.requires_elements:
            lines.append(f"    loft:requiresElement asp:{req} ;")

        for alt in self.has_alternatives:
            lines.append(f"    loft:hasAlternative asp:{alt} ;")

        # Remove trailing semicolon and add period
        if lines[-1].endswith(";"):
            lines[-1] = lines[-1][:-1] + " ."

        return "\n".join(lines)


@dataclass
class ModuleMetadata:
    """
    Metadata for an ASP module (collection of related rules).

    Maps to the loft:ASPModule class in the LinkedASP ontology.
    """

    module_name: str
    domain: str  # e.g., "contract_law"
    phase: str  # ROADMAP.md phase
    stratification_level: str
    description: str
    exports: List[str] = field(default_factory=list)  # Exported predicates
    imports: List[str] = field(default_factory=list)  # Imported predicates
    rules: List[RuleMetadata] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "module_name": self.module_name,
            "domain": self.domain,
            "phase": self.phase,
            "stratification_level": self.stratification_level,
            "description": self.description,
            "exports": self.exports,
            "imports": self.imports,
            "rules": [rule.to_dict() for rule in self.rules],
        }


class LinkedASPExporter:
    """
    Exports living document data in LinkedASP-compatible formats.

    Prepares for future integration with RDF/SPARQL-based querying
    as described in docs/MAINTAINABILITY.md.
    """

    def __init__(self):
        """Initialize the exporter."""
        self.rules: List[RuleMetadata] = []
        self.modules: List[ModuleMetadata] = []

    def add_rule(self, rule: RuleMetadata) -> None:
        """Add a rule to the export."""
        self.rules.append(rule)

    def add_module(self, module: ModuleMetadata) -> None:
        """Add a module to the export."""
        self.modules.append(module)

    def export_json_ld(self, output_path: str) -> str:
        """
        Export metadata as JSON-LD for future RDF processing.

        JSON-LD is a JSON format that can be converted to RDF triples,
        making it queryable via SPARQL once the LinkedASP system is implemented.
        """
        context = {
            "@context": {
                "@vocab": "https://loft.legal/ontology/",
                "loft": "https://loft.legal/ontology/",
                "asp": "https://loft.legal/asp/",
                "legal": "https://loft.legal/patterns/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            },
            "modules": [module.to_dict() for module in self.modules],
            "rules": [rule.to_dict() for rule in self.rules],
            "generated_at": datetime.now().isoformat(),
            "generator": "LOFT Living Document Generator v1.0",
        }

        json_content = json.dumps(context, indent=2)

        # Write to file
        Path(output_path).write_text(json_content)

        return json_content

    def export_turtle(self, output_path: str) -> str:
        """
        Export metadata as RDF Turtle format.

        This format will be directly consumable by SPARQL query engines
        once the LinkedASP system is implemented.
        """
        lines = [
            "# LinkedASP Metadata Export",
            f"# Generated: {datetime.now().isoformat()}",
            "# Compatible with docs/MAINTAINABILITY.md LinkedASP ontology",
            "",
            "@prefix loft: <https://loft.legal/ontology/> .",
            "@prefix asp: <https://loft.legal/asp/> .",
            "@prefix legal: <https://loft.legal/patterns/> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
        ]

        # Export modules
        for module in self.modules:
            lines.extend(
                [
                    "",
                    f"<{module.module_name}> a loft:ASPModule ;",
                    f'    rdfs:label "{module.module_name}" ;',
                    f'    loft:domain "{module.domain}" ;',
                    f'    loft:phase "{module.phase}" ;',
                    f'    loft:stratificationLevel "{module.stratification_level}" ;',
                    f'    rdfs:comment "{module.description}" .',
                    "",
                ]
            )

        # Export rules
        for rule in self.rules:
            lines.append("")
            lines.append(rule.to_turtle())

        turtle_content = "\n".join(lines)

        # Write to file
        Path(output_path).write_text(turtle_content)

        return turtle_content

    def generate_query_examples(self, output_path: str) -> str:
        """
        Generate example SPARQL queries for the exported metadata.

        These queries demonstrate how to use the LinkedASP system
        once it's implemented (Phase 1.5+).
        """
        queries = """# Example SPARQL Queries for LinkedASP Metadata
#
# These queries will work once the LinkedASP system is implemented
# (See docs/MAINTAINABILITY.md Phase 1.5+)

## Query 1: Find all rules of a specific genre
PREFIX legal: <https://loft.legal/patterns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?rule ?label
WHERE {
    ?rule a legal:DisjunctiveRequirement .
    ?rule rdfs:label ?label .
}

## Query 2: Find rules with low confidence (candidates for refinement)
PREFIX loft: <https://loft.legal/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?rule ?label ?confidence
WHERE {
    ?rule loft:confidence ?confidence .
    ?rule rdfs:label ?label .
    FILTER(?confidence < 0.8)
}
ORDER BY ?confidence

## Query 3: Analyze impact of modifying a predicate
PREFIX loft: <https://loft.legal/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?rule ?label ?level
WHERE {
    ?rule loft:requiresElement <https://loft.legal/asp/within_statute> .
    ?rule rdfs:label ?label .
    ?rule loft:stratificationLevel ?level .
}

## Query 4: Find all LLM-generated rules
PREFIX loft: <https://loft.legal/ontology/>

SELECT ?rule ?label ?llm ?timestamp
WHERE {
    ?rule loft:sourceType "llm_generated" .
    ?rule rdfs:label ?label .
    ?rule loft:sourceLLM ?llm .
    ?rule loft:generationTimestamp ?timestamp .
}

## Query 5: Detect stratification violations
PREFIX loft: <https://loft.legal/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?rule ?label ?dependency
WHERE {
    ?rule loft:stratificationLevel "constitutional" .
    ?rule loft:requiresElement ?dependency .
    ?dependency loft:stratificationLevel ?depLevel .
    FILTER(?depLevel IN ("strategic", "tactical", "operational"))
}

## Query 6: Track rule evolution over time
PREFIX loft: <https://loft.legal/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?rule ?replaced ?timestamp
WHERE {
    ?rule loft:replacedRule ?replaced .
    ?rule loft:incorporationTimestamp ?timestamp .
}
ORDER BY ?timestamp
"""

        Path(output_path).write_text(queries)
        return queries
