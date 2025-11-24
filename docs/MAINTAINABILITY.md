# ASP Maintainability Strategy: LinkedASP + Genre-Based Generation

## Problem Statement

As LOFT evolves from Phase 1 (statute of frauds) through Phase 8 (multi-domain expansion), the ASP symbolic core faces a critical maintainability challenge:

**Current State (Phase 1.5):**
- 210 lines of ASP for a single legal domain (statute of frauds)
- Flat predicate namespace with implicit dependencies
- Manual stratification through comments
- No compositional abstractions

**Projected State (Phase 8):**
- Potential 5,000+ lines of interconnected ASP rules
- Multiple legal domains (contracts, torts, constitutional, criminal law)
- Complex cross-domain interactions
- Risk of unmaintainable "ASP spaghetti code"

**Risk Identified in ROADMAP.md:**
> Risk: ASP program complexity may be hard to maintain
> Mitigation: Comprehensive documentation, clear stratification, extensive comments

This document proposes a **stronger mitigation strategy** inspired by creative synthesis of two external projects:

1. **G-Lisp**: Genre-based abstraction with meta-programming (`gexpand`)
2. **GraphFS**: Semantic metadata (LinkedDoc+RDF) for queryable code relationships

---

## Solution: LinkedASP + Genre-Based Code Generation

### Core Innovation

Instead of writing ASP rules directly, we:

1. **Define legal reasoning genres** (patterns like "requirement", "exception", "balancing test")
2. **Annotate ASP with rich RDF metadata** describing semantic structure
3. **Generate ASP code** from high-level genre-based compositions
4. **Query the ASP structure** using SPARQL to understand dependencies, detect violations, and guide LLM interactions

This approach makes the symbolic core **self-documenting, queryable, and compositional**—perfectly aligned with LOFT's vision of a self-reflexive system.

---

## RDF Ontology for Legal Reasoning Patterns

### Ontology Design

```turtle
# loft/ontology/legal_reasoning.ttl

@prefix loft: <https://loft.legal/ontology/> .
@prefix asp: <https://loft.legal/asp/> .
@prefix legal: <https://loft.legal/patterns/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ============================================
# Core Legal Reasoning Genre Classes
# ============================================

legal:ReasoningGenre a owl:Class ;
    rdfs:label "Legal Reasoning Genre" ;
    rdfs:comment "Abstract pattern of legal reasoning (inspired by G-Lisp's genre concept)" ;
    rdfs:seeAlso <https://github.com/justin4957/glisp> .

legal:Requirement a owl:Class ;
    rdfs:subClassOf legal:ReasoningGenre ;
    rdfs:label "Legal Requirement" ;
    rdfs:comment "A set of elements that must be satisfied for a legal conclusion" ;
    loft:compositionType "conjunctive" ;
    loft:expandsTo "asp:rule with conjunction of elements" .

legal:DisjunctiveRequirement a owl:Class ;
    rdfs:subClassOf legal:Requirement ;
    rdfs:label "Disjunctive Requirement" ;
    rdfs:comment "Satisfied by ANY of several alternatives" ;
    loft:compositionType "disjunctive" ;
    loft:expandsTo "multiple asp:rules with same head" .

legal:ConjunctiveRequirement a owl:Class ;
    rdfs:subClassOf legal:Requirement ;
    rdfs:label "Conjunctive Requirement" ;
    rdfs:comment "Requires ALL elements to be satisfied" ;
    loft:compositionType "conjunctive" ;
    loft:expandsTo "single asp:rule with conjoined body" .

legal:Exception a owl:Class ;
    rdfs:subClassOf legal:ReasoningGenre ;
    rdfs:label "Exception Pattern" ;
    rdfs:comment "Conditions that negate or modify a requirement" ;
    loft:expandsTo "asp:rule modifying parent requirement" .

legal:BalancingTest a owl:Class ;
    rdfs:subClassOf legal:ReasoningGenre ;
    rdfs:label "Multi-Factor Balancing Test" ;
    rdfs:comment "Weighted consideration of multiple factors (common in constitutional law)" ;
    loft:hasProperty legal:Factor ;
    loft:expandsTo "asp:aggregation rules with weights" .

legal:Presumption a owl:Class ;
    rdfs:subClassOf legal:ReasoningGenre ;
    rdfs:label "Rebuttable Presumption" ;
    rdfs:comment "Default conclusion holds unless rebutted" ;
    loft:hasProperty legal:DefaultConclusion ;
    loft:hasProperty legal:RebuttalCondition ;
    loft:expandsTo "asp:default rule with negation-as-failure" .

legal:StandardOfProof a owl:Class ;
    rdfs:subClassOf legal:ReasoningGenre ;
    rdfs:label "Standard of Proof" ;
    rdfs:comment "Burden of proof requirement (preponderance, clear and convincing, beyond reasonable doubt)" ;
    loft:hasProperty legal:ConfidenceThreshold .

legal:ScopingRule a owl:Class ;
    rdfs:subClassOf legal:ReasoningGenre ;
    rdfs:label "Scoping Rule" ;
    rdfs:comment "Defines when a legal domain or rule applies" ;
    loft:expandsTo "asp:conditional activation rules" .

# ============================================
# Properties for Describing ASP Structure
# ============================================

## Rule Composition Properties

loft:hasGenre a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range legal:ReasoningGenre ;
    rdfs:label "has legal reasoning genre" ;
    rdfs:comment "Identifies the legal reasoning pattern this rule implements" .

loft:requiresElement a owl:ObjectProperty ;
    rdfs:domain legal:Requirement ;
    rdfs:range asp:Predicate ;
    rdfs:label "requires element" ;
    rdfs:comment "An element that must be satisfied for this requirement" .

loft:hasAlternative a owl:ObjectProperty ;
    rdfs:domain legal:DisjunctiveRequirement ;
    rdfs:range asp:Predicate ;
    rdfs:label "has alternative" ;
    rdfs:comment "One of several alternatives that can satisfy this requirement" .

loft:hasFactor a owl:ObjectProperty ;
    rdfs:domain legal:BalancingTest ;
    rdfs:range legal:Factor ;
    rdfs:label "has factor" ;
    rdfs:comment "A factor to be weighed in this balancing test" .

loft:appliesToRequirement a owl:ObjectProperty ;
    rdfs:domain legal:Exception ;
    rdfs:range legal:Requirement ;
    rdfs:label "applies to requirement" ;
    rdfs:comment "The requirement that this exception modifies or negates" .

loft:defaultConclusion a owl:ObjectProperty ;
    rdfs:domain legal:Presumption ;
    rdfs:range asp:Predicate ;
    rdfs:label "default conclusion" ;
    rdfs:comment "The conclusion that holds by default" .

loft:rebuttalCondition a owl:ObjectProperty ;
    rdfs:domain legal:Presumption ;
    rdfs:range asp:Predicate ;
    rdfs:label "rebuttal condition" ;
    rdfs:comment "Condition that rebuts the presumption" .

## Stratification Properties

loft:stratificationLevel a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "stratification level" ;
    rdfs:comment "One of: constitutional, strategic, tactical, operational" .

loft:modificationAuthority a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "modification authority" ;
    rdfs:comment "Who/what can modify this rule: human, llm_with_validation, autonomous" .

## Legal Source Properties

loft:legalSource a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "legal source" ;
    rdfs:comment "Citation to statute, case law, regulation, or Restatement" .

loft:jurisdiction a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "jurisdiction" ;
    rdfs:comment "Legal jurisdiction where this rule applies (e.g., 'federal', 'california', 'common_law')" .

## Confidence and Validation Properties

loft:confidence a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:float ;
    rdfs:label "confidence score" ;
    rdfs:comment "System's confidence in this rule (0.0-1.0), updated through experiential learning" .

loft:validatedBy a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range loft:TestCase ;
    rdfs:label "validated by" ;
    rdfs:comment "Test case that validates this rule" .

loft:successRate a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:float ;
    rdfs:label "success rate" ;
    rdfs:comment "Empirical success rate on test cases (0.0-1.0)" .

loft:lastModified a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:dateTime ;
    rdfs:label "last modified" ;
    rdfs:comment "Timestamp of last modification (for version control integration)" .

## Module and Dependency Properties

loft:exports a owl:ObjectProperty ;
    rdfs:domain loft:ASPModule ;
    rdfs:range asp:Predicate ;
    rdfs:label "exports" ;
    rdfs:comment "Predicates that this module provides to other modules" .

loft:imports a owl:ObjectProperty ;
    rdfs:domain loft:ASPModule ;
    rdfs:range asp:Predicate ;
    rdfs:label "imports" ;
    rdfs:comment "Predicates that this module depends on from other modules" .

loft:domain a owl:DatatypeProperty ;
    rdfs:domain loft:ASPModule ;
    rdfs:range xsd:string ;
    rdfs:label "legal domain" ;
    rdfs:comment "Legal domain this module belongs to (e.g., 'contract_law', 'tort_law')" .

loft:phase a owl:DatatypeProperty ;
    rdfs:domain loft:ASPModule ;
    rdfs:range xsd:string ;
    rdfs:label "project phase" ;
    rdfs:comment "ROADMAP.md phase where this module was introduced" .

## Code Generation Properties

loft:expandsInto a owl:DatatypeProperty ;
    rdfs:domain legal:ReasoningGenre ;
    rdfs:range xsd:string ;
    rdfs:label "expands into" ;
    rdfs:comment "ASP predicate name(s) that this genre generates" .

loft:generatedFrom a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range legal:ReasoningGenre ;
    rdfs:label "generated from" ;
    rdfs:comment "Genre pattern that generated this ASP code" .

loft:expansionTemplate a owl:DatatypeProperty ;
    rdfs:domain legal:ReasoningGenre ;
    rdfs:range xsd:string ;
    rdfs:label "expansion template" ;
    rdfs:comment "Template string for generating ASP code from this genre" .

## Provenance Tracking Properties

loft:sourceType a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "source type" ;
    rdfs:comment "Type of source: manual, llm_generated, gap_fill, case_law, principle, human_revision" .

loft:sourceText a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "source text" ;
    rdfs:comment "Original text (natural language) that was used to generate this rule" .

loft:sourceLLM a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range loft:LLMProvider ;
    rdfs:label "source LLM" ;
    rdfs:comment "LLM provider and model that generated this rule (e.g., 'anthropic/claude-3-opus')" .

loft:sourcePromptVersion a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "source prompt version" ;
    rdfs:comment "Version of prompt template used (e.g., 'gap_fill_v1.1', 'principle_expansion_v2.0')" .

loft:generationTimestamp a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:dateTime ;
    rdfs:label "generation timestamp" ;
    rdfs:comment "When this rule was generated by LLM or created manually" .

loft:incorporationTimestamp a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:dateTime ;
    rdfs:label "incorporation timestamp" ;
    rdfs:comment "When this rule was incorporated into the ASP core" .

loft:incorporatedBy a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "incorporated by" ;
    rdfs:comment "Agent that incorporated this rule: 'human', 'autonomous_system', 'reviewed_by_human'" .

loft:validationReportId a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "validation report ID" ;
    rdfs:comment "ID of validation report that approved this rule for incorporation" .

loft:snapshotId a owl:DatatypeProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range xsd:string ;
    rdfs:label "snapshot ID" ;
    rdfs:comment "Version control snapshot ID when this rule was incorporated (enables rollback)" .

loft:replacedRule a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range asp:ASPRule ;
    rdfs:label "replaced rule" ;
    rdfs:comment "Previous rule that this rule replaces (if any). Forms replacement chain for tracking evolution." .

loft:replacedBy a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range asp:ASPRule ;
    rdfs:label "replaced by" ;
    rdfs:comment "Rule that replaced this one (if deprecated). Inverse of loft:replacedRule." .

loft:derivedFromTestCase a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range loft:TestCase ;
    rdfs:label "derived from test case" ;
    rdfs:comment "Test case failure that prompted generation of this rule (gap-fill scenario)" .

loft:modificationChain a owl:ObjectProperty ;
    rdfs:domain asp:ASPRule ;
    rdfs:range rdf:List ;
    rdfs:label "modification chain" ;
    rdfs:comment "Ordered list of modifications to this rule (complete lineage from origin to current state)" .

# ============================================
# Supporting Classes
# ============================================

loft:ASPModule a owl:Class ;
    rdfs:label "ASP Module" ;
    rdfs:comment "A cohesive collection of related ASP rules" .

asp:ASPRule a owl:Class ;
    rdfs:label "ASP Rule" ;
    rdfs:comment "A single Answer Set Programming rule" .

asp:Predicate a owl:Class ;
    rdfs:label "ASP Predicate" ;
    rdfs:comment "A predicate in the ASP language" .

loft:TestCase a owl:Class ;
    rdfs:label "Test Case" ;
    rdfs:comment "A test case for validating ASP rules" .

legal:Factor a owl:Class ;
    rdfs:label "Balancing Factor" ;
    rdfs:comment "A factor in a multi-factor balancing test" ;
    loft:hasProperty legal:weight ;
    loft:hasProperty legal:direction .

# ============================================
# Example Instances (for documentation)
# ============================================

# Example: Statute of Frauds as a Disjunctive Requirement
<https://loft.legal/domains/statute_of_frauds/within_statute> a legal:DisjunctiveRequirement ;
    rdfs:label "Contracts Within Statute of Frauds" ;
    rdfs:comment "A contract falls within the statute if it matches ANY category" ;
    loft:hasGenre legal:DisjunctiveRequirement ;
    loft:stratificationLevel "strategic" ;
    loft:hasAlternative <https://loft.legal/asp/land_sale_contract> ;
    loft:hasAlternative <https://loft.legal/asp/long_term_contract> ;
    loft:hasAlternative <https://loft.legal/asp/goods_over_500> ;
    loft:legalSource "UCC § 2-201, Restatement (Second) of Contracts §§ 110-150" ;
    loft:confidence 0.95 ;
    loft:expandsInto "within_statute/1" .

# Example: Part Performance as an Exception
<https://loft.legal/domains/statute_of_frauds/part_performance> a legal:Exception ;
    rdfs:label "Part Performance Exception" ;
    rdfs:comment "Writing not required if part performance shown" ;
    loft:hasGenre legal:Exception ;
    loft:appliesToRequirement <https://loft.legal/domains/statute_of_frauds/writing_requirement> ;
    loft:requiresElement <https://loft.legal/asp/substantial_actions_taken> ;
    loft:requiresElement <https://loft.legal/asp/detrimental_reliance> ;
    loft:legalSource "Restatement (Second) of Contracts § 129" ;
    loft:confidence 0.88 ;
    loft:expandsInto "exception_applies/1" .
```

### Ontology Design Principles

1. **Genre-Based Abstraction**: Legal reasoning patterns as first-class objects (inspired by G-Lisp's `:g` genre field)
2. **Explicit Relationships**: Dependencies, exceptions, alternatives explicitly modeled (inspired by GraphFS's RDF relationships)
3. **Stratification Support**: Metadata tracks modification authority and layer assignment
4. **Experiential Learning**: Confidence scores and success rates track rule performance over time
5. **Queryability**: SPARQL enables complex queries about ASP structure
6. **Code Generation**: Templates and expansion rules transform genres into ASP

---

## Genre-Based ASP Code Generation

### High-Level Genre Definitions

Instead of writing 210 lines of ASP manually, define legal structure at a higher level:

```python
# loft/symbolic/genre_based_asp.py

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class LegalGenre(Enum):
    """Legal reasoning pattern types."""
    DISJUNCTIVE_REQUIREMENT = "disjunctive_requirement"
    CONJUNCTIVE_REQUIREMENT = "conjunctive_requirement"
    EXCEPTION = "exception"
    BALANCING_TEST = "balancing_test"
    PRESUMPTION = "presumption"
    SCOPING_RULE = "scoping_rule"

@dataclass
class GenrePattern:
    """Base class for genre-based legal patterns."""
    name: str
    genre: LegalGenre
    stratification_level: str
    legal_source: str
    confidence: float
    description: str

    def expand_to_asp(self) -> str:
        """Generate ASP code from this pattern."""
        raise NotImplementedError

    def to_rdf_metadata(self) -> str:
        """Generate LinkedASP RDF metadata."""
        raise NotImplementedError

@dataclass
class DisjunctiveRequirement(GenrePattern):
    """Requirement satisfied by ANY alternative."""
    alternatives: List[str]

    def __init__(self, name: str, alternatives: List[str], **kwargs):
        super().__init__(
            name=name,
            genre=LegalGenre.DISJUNCTIVE_REQUIREMENT,
            **kwargs
        )
        self.alternatives = alternatives

    def expand_to_asp(self) -> str:
        """Generate ASP rules for disjunctive requirement."""
        rules = []
        for alt in self.alternatives:
            rules.append(f"{self.name}(X) :- {alt}(X).")
        return "\n".join(rules)

    def to_rdf_metadata(self) -> str:
        """Generate RDF metadata for this pattern."""
        alternatives_rdf = " ;\n    ".join(
            f"loft:hasAlternative asp:{alt}"
            for alt in self.alternatives
        )

        return f"""
%%% <!-- LinkedASP Metadata -->
%%% @prefix loft: <https://loft.legal/ontology/> .
%%% @prefix asp: <https://loft.legal/asp/> .
%%% @prefix legal: <https://loft.legal/patterns/> .
%%%
%%% asp:{self.name} a legal:DisjunctiveRequirement ;
%%%     rdfs:label "{self.description}" ;
%%%     loft:hasGenre legal:DisjunctiveRequirement ;
%%%     loft:stratificationLevel "{self.stratification_level}" ;
%%%     {alternatives_rdf} ;
%%%     loft:legalSource "{self.legal_source}" ;
%%%     loft:confidence {self.confidence} ;
%%%     loft:expandsInto "{self.name}/1" .
%%% <!-- End LinkedASP Metadata -->
"""

@dataclass
class ConjunctiveRequirement(GenrePattern):
    """Requirement requiring ALL elements."""
    elements: List[str]

    def __init__(self, name: str, elements: List[str], **kwargs):
        super().__init__(
            name=name,
            genre=LegalGenre.CONJUNCTIVE_REQUIREMENT,
            **kwargs
        )
        self.elements = elements

    def expand_to_asp(self) -> str:
        """Generate ASP rule with conjoined elements."""
        element_conjunction = ",\n    ".join(self.elements)
        return f"""{self.name}(X) :-
    instance(X),
    {element_conjunction}."""

    def to_rdf_metadata(self) -> str:
        elements_rdf = " ;\n    ".join(
            f"loft:requiresElement asp:{elem}"
            for elem in self.elements
        )

        return f"""
%%% <!-- LinkedASP Metadata -->
%%% asp:{self.name} a legal:ConjunctiveRequirement ;
%%%     rdfs:label "{self.description}" ;
%%%     {elements_rdf} ;
%%%     loft:legalSource "{self.legal_source}" ;
%%%     loft:confidence {self.confidence} .
%%% <!-- End LinkedASP Metadata -->
"""

@dataclass
class ExceptionPattern(GenrePattern):
    """Exception that modifies a requirement."""
    applies_to: str
    conditions: List[str]

    def __init__(self, name: str, applies_to: str, conditions: List[str], **kwargs):
        super().__init__(
            name=name,
            genre=LegalGenre.EXCEPTION,
            **kwargs
        )
        self.applies_to = applies_to
        self.conditions = conditions

    def expand_to_asp(self) -> str:
        """Generate exception rule."""
        condition_conjunction = ",\n    ".join(self.conditions)
        return f"""exception_applies(X) :-
    {self.applies_to}(X),
    {condition_conjunction}."""

    def to_rdf_metadata(self) -> str:
        conditions_rdf = " ;\n    ".join(
            f"loft:requiresElement asp:{cond}"
            for cond in self.conditions
        )

        return f"""
%%% <!-- LinkedASP Metadata -->
%%% asp:{self.name} a legal:Exception ;
%%%     rdfs:label "{self.description}" ;
%%%     loft:appliesToRequirement asp:{self.applies_to} ;
%%%     {conditions_rdf} ;
%%%     loft:legalSource "{self.legal_source}" ;
%%%     loft:confidence {self.confidence} .
%%% <!-- End LinkedASP Metadata -->
"""

@dataclass
class LegalDomain:
    """High-level legal domain composed of genre patterns."""
    name: str
    patterns: List[GenrePattern]
    module_metadata: dict

    def generate_asp_module(self) -> str:
        """Generate complete ASP module with metadata."""
        sections = []

        # Module header
        sections.append(self._generate_module_header())

        # Generate each pattern
        for pattern in self.patterns:
            sections.append(pattern.to_rdf_metadata())
            sections.append(pattern.expand_to_asp())
            sections.append("")  # Blank line

        return "\n".join(sections)

    def _generate_module_header(self) -> str:
        """Generate module-level RDF metadata."""
        return f"""%%% ============================================
%%% {self.module_metadata.get('label', self.name)}
%%% ============================================
%%%
%%% <!-- LinkedASP Metadata -->
%%% @prefix loft: <https://loft.legal/ontology/> .
%%%
%%% <this> a loft:ASPModule ;
%%%     rdfs:label "{self.module_metadata.get('label', self.name)}" ;
%%%     loft:domain "{self.module_metadata.get('domain', 'unknown')}" ;
%%%     loft:phase "{self.module_metadata.get('phase', 'unknown')}" ;
%%%     loft:stratificationLevel "{self.module_metadata.get('stratification', 'tactical')}" .
%%% <!-- End LinkedASP Metadata -->
%%%
"""
```

### Example: Statute of Frauds Generated from Genres

```python
# Example: Generate statute_of_frauds.lp from genre patterns

statute_of_frauds = LegalDomain(
    name="statute_of_frauds",
    module_metadata={
        "label": "Statute of Frauds",
        "domain": "contract_law",
        "phase": "1.5",
        "stratification": "strategic"
    },
    patterns=[
        # Within statute determination (disjunctive)
        DisjunctiveRequirement(
            name="within_statute",
            alternatives=[
                "land_sale_contract",
                "long_term_contract",
                "goods_over_500",
                "suretyship_contract",
                "marriage_consideration_contract",
                "executor_contract"
            ],
            stratification_level="strategic",
            legal_source="UCC § 2-201, Restatement (Second) of Contracts §§ 110-150",
            confidence=0.95,
            description="Contracts Within Statute of Frauds"
        ),

        # Writing requirement (conjunctive)
        ConjunctiveRequirement(
            name="sufficient_writing",
            elements=[
                "writing_exists(X, W)",
                "signed_by(W, P)",
                "party_to_contract(X, P)",
                "essential_terms(W)"
            ],
            stratification_level="tactical",
            legal_source="Restatement (Second) of Contracts § 131",
            confidence=0.92,
            description="Sufficient Writing Requirement"
        ),

        # Part performance exception
        ExceptionPattern(
            name="part_performance",
            applies_to="writing_requirement",
            conditions=[
                "substantial_actions_taken(X)",
                "detrimental_reliance(X)"
            ],
            stratification_level="tactical",
            legal_source="Restatement (Second) of Contracts § 129",
            confidence=0.88,
            description="Part Performance Exception"
        ),

        # Promissory estoppel exception
        ExceptionPattern(
            name="promissory_estoppel",
            applies_to="writing_requirement",
            conditions=[
                "clear_promise(X)",
                "reasonable_reliance(X)",
                "substantial_detriment(X)",
                "injustice_without_enforcement(X)"
            ],
            stratification_level="tactical",
            legal_source="Restatement (Second) of Contracts § 90",
            confidence=0.85,
            description="Promissory Estoppel Exception"
        ),
    ]
)

# Generate the ASP file
asp_code = statute_of_frauds.generate_asp_module()

# Write to file
with open("loft/legal/statute_of_frauds_generated.lp", "w") as f:
    f.write(asp_code)
```

This generates a well-structured ASP file with rich RDF metadata, maintainable at the genre level rather than individual rule level.

---

## LinkedASP Query and Analysis Tools

### SPARQL Queries for ASP Structure

```python
# loft/symbolic/linkedasp_queries.py

from rdflib import Graph, Namespace
from typing import List, Dict, Set

LOFT = Namespace("https://loft.legal/ontology/")
LEGAL = Namespace("https://loft.legal/patterns/")
ASP = Namespace("https://loft.legal/asp/")

class LinkedASPAnalyzer:
    """Query and analyze ASP structure using RDF metadata."""

    def __init__(self):
        self.graph = Graph()
        self.graph.bind("loft", LOFT)
        self.graph.bind("legal", LEGAL)
        self.graph.bind("asp", ASP)

    def load_asp_file(self, filepath: str) -> None:
        """Parse LinkedASP metadata from .lp file."""
        import re

        with open(filepath, 'r') as f:
            content = f.read()

        # Extract RDF blocks
        rdf_pattern = r'%%% <!-- LinkedASP Metadata -->(.*?)%%% <!-- End LinkedASP Metadata -->'
        matches = re.findall(rdf_pattern, content, re.DOTALL)

        for match in matches:
            # Remove comment markers
            rdf_text = re.sub(r'%%%\s*', '', match)
            # Parse as Turtle
            self.graph.parse(data=rdf_text, format='turtle')

    def find_rules_by_genre(self, genre: str) -> List[str]:
        """Find all rules of a specific genre type."""
        query = f"""
        PREFIX legal: <https://loft.legal/patterns/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?label
        WHERE {{
            ?rule a legal:{genre} .
            ?rule rdfs:label ?label .
        }}
        """
        results = self.graph.query(query)
        return [(str(row.rule), str(row.label)) for row in results]

    def impact_analysis(self, predicate: str) -> Dict[str, any]:
        """Analyze impact of modifying a predicate."""
        query = f"""
        PREFIX loft: <https://loft.legal/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?label ?level
        WHERE {{
            ?rule loft:requiresElement <https://loft.legal/asp/{predicate}> .
            ?rule rdfs:label ?label .
            ?rule loft:stratificationLevel ?level .
        }}
        """

        affected_rules = []
        stratification_levels = set()

        for row in self.graph.query(query):
            affected_rules.append({
                'uri': str(row.rule),
                'label': str(row.label),
                'level': str(row.level)
            })
            stratification_levels.add(str(row.level))

        return {
            'predicate': predicate,
            'directly_affected': len(affected_rules),
            'affected_rules': affected_rules,
            'stratification_levels': list(stratification_levels)
        }

    def detect_stratification_violations(self) -> List[str]:
        """Find rules that violate stratification hierarchy."""
        violations = []

        # Constitutional rules cannot depend on lower layers
        query = """
        PREFIX loft: <https://loft.legal/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?label ?dependency
        WHERE {
            ?rule loft:stratificationLevel "constitutional" .
            ?rule loft:requiresElement ?dependency .
            ?dependency loft:stratificationLevel ?depLevel .
            FILTER(?depLevel IN ("strategic", "tactical", "operational"))
        }
        """

        for row in self.graph.query(query):
            violations.append(
                f"Constitutional rule '{row.label}' depends on lower-level predicate '{row.dependency}'"
            )

        return violations

    def find_low_confidence_rules(self, threshold: float = 0.8) -> List[Dict]:
        """Find rules with low confidence scores (candidates for LLM refinement)."""
        query = f"""
        PREFIX loft: <https://loft.legal/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?label ?confidence
        WHERE {{
            ?rule loft:confidence ?confidence .
            ?rule rdfs:label ?label .
            FILTER(?confidence < {threshold})
        }}
        ORDER BY ?confidence
        """

        results = []
        for row in self.graph.query(query):
            results.append({
                'uri': str(row.rule),
                'label': str(row.label),
                'confidence': float(row.confidence)
            })

        return results
```

---

## Integration with LOFT's Self-Reflexive Architecture

### Self-Querying Symbolic Core

The LinkedASP approach enables LOFT's symbolic core to **reason about its own structure**:

```python
# loft/meta/self_reflection_linkedasp.py

from loft.symbolic.linkedasp_queries import LinkedASPAnalyzer
from loft.neural.llm_interface import LLMInterface

class SelfReflectiveASPCore:
    """Symbolic core that can query and modify its own structure."""

    def __init__(self, asp_core, llm_interface: LLMInterface):
        self.asp_core = asp_core
        self.llm = llm_interface
        self.analyzer = LinkedASPAnalyzer()

    def identify_knowledge_gaps(self) -> List[str]:
        """Find missing predicates using RDF metadata."""
        # Query: which required elements are not defined?
        query = """
        PREFIX loft: <https://loft.legal/ontology/>

        SELECT DISTINCT ?element
        WHERE {
            ?rule loft:requiresElement ?element .
            FILTER NOT EXISTS {
                ?element a ?type
            }
        }
        """
        results = self.analyzer.graph.query(query)
        return [str(row.element).split('/')[-1] for row in results]

    def suggest_refinements(self) -> List[Dict]:
        """Identify low-confidence rules for LLM-assisted improvement."""
        candidates = self.analyzer.find_low_confidence_rules(threshold=0.8)

        refinement_suggestions = []
        for candidate in candidates:
            # Get rich context from RDF
            context = self._get_rule_context(candidate['uri'])

            # Generate LLM query
            llm_query = self._generate_refinement_query(candidate, context)

            refinement_suggestions.append({
                'rule': candidate,
                'context': context,
                'llm_query': llm_query
            })

        return refinement_suggestions

    def _get_rule_context(self, rule_uri: str) -> Dict:
        """Extract full context for a rule from RDF graph."""
        # Get genre, dependencies, legal sources, etc.
        query = f"""
        PREFIX loft: <https://loft.legal/ontology/>
        PREFIX legal: <https://loft.legal/patterns/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?genre ?source ?element
        WHERE {{
            <{rule_uri}> a ?genre .
            OPTIONAL {{ <{rule_uri}> loft:legalSource ?source }}
            OPTIONAL {{ <{rule_uri}> loft:requiresElement ?element }}
        }}
        """

        results = self.analyzer.graph.query(query)

        context = {
            'genre': None,
            'legal_sources': [],
            'required_elements': []
        }

        for row in results:
            if row.genre:
                context['genre'] = str(row.genre).split('/')[-1]
            if row.source:
                context['legal_sources'].append(str(row.source))
            if row.element:
                context['required_elements'].append(str(row.element).split('/')[-1])

        return context

    def _generate_refinement_query(self, rule: Dict, context: Dict) -> str:
        """Generate LLM query for rule refinement."""
        return f"""
I need to refine the ASP rule '{rule['label']}' which currently has a confidence score of {rule['confidence']}.

Context from the knowledge graph:
- Legal reasoning genre: {context['genre']}
- Legal sources: {', '.join(context['legal_sources'])}
- Required elements: {', '.join(context['required_elements'])}

Please provide:
1. Analysis of potential issues with the current rule
2. Suggested refinements to improve accuracy
3. Additional test cases to validate the refined rule
4. Expected confidence score improvement

Generate the refined rule in ASP syntax and explain the changes.
"""
```

---

## Roadmap Integration Strategy

### Recommended Phase: **Phase 1.5+ (Tangential Route)**

**Timing**: After completing **Phase 2** (LLM Logic Generation + Validation), before starting **Phase 8** (Multi-Domain Expansion)

**Rationale**:
1. **Not immediately critical**: Current 210-line ASP is manageable; don't over-complicate Phase 1
2. **Becomes essential at scale**: By Phase 8 with multiple legal domains, maintainability crisis is real
3. **Synergy with Phase 2**: LLM logic generation benefits enormously from RDF-based context
4. **Natural trigger**: After implementing 2-3 legal domains, patterns will be clear

**Trigger Conditions**:
- ASP codebase exceeds 500 lines
- OR 3+ legal domains implemented
- OR circular dependency detected
- OR manual maintenance becomes painful

**Implementation Phases**:
1. **Phase 1.5a (Weeks 6-7)**: Design RDF ontology, define core genres
2. **Phase 1.5b (Weeks 8-9)**: Implement LinkedASP parser and analyzer
3. **Phase 1.5c (Weeks 10-11)**: Refactor existing domains to use genres
4. **Phase 1.5d (Week 12)**: Integrate with self-reflexive meta-reasoning

---

## Benefits for LOFT's Mission

This LinkedASP approach directly supports LOFT's core architectural principles:

### 1. Self-Reflexive Symbolic Core (Principle #1)
- Core can **query its own structure** using SPARQL
- Identifies gaps, low-confidence rules, and refinement opportunities
- Generates **context-aware LLM queries** from RDF metadata

### 2. Validation Oversight (Principle #2)
- **Automated validation** of stratification rules
- **Dependency analysis** before modifications
- **Confidence tracking** through experiential learning

### 3. Ontological Bridge Integrity (Principle #3)
- RDF provides **semantic grounding** for symbolic-neural translation
- LLMs receive **rich context** from RDF metadata
- Genre patterns enable **compositional reasoning** across domains

### 4. Composability & Reusability (Coding Standard)
- **Genre patterns reused** across legal domains
- **No duplication**: "Exception" pattern works for all exceptions
- **Transposability**: Patterns transfer beyond legal domain

---

## Comparison to Current Mitigation

| Aspect | Current Mitigation | LinkedASP Approach |
|--------|-------------------|-------------------|
| **Documentation** | Comments in ASP files | Rich RDF metadata, queryable |
| **Stratification** | Manual comments | Enforced by RDF validation |
| **Dependencies** | Implicit in code | Explicit in RDF graph |
| **Maintainability** | Manual code review | Automated analysis tools |
| **Scalability** | Linear complexity growth | Compositional, genre-based |
| **LLM Integration** | Ad-hoc context gathering | Semantic context from RDF |
| **Self-Reflexivity** | Limited | Core can query itself |

---

## Implementation Checklist

When implementing LinkedASP (Phase 1.5+):

- [ ] Define complete RDF ontology (`loft/ontology/legal_reasoning.ttl`)
- [ ] Implement genre pattern classes (`loft/symbolic/genre_based_asp.py`)
- [ ] Build LinkedASP parser (`loft/symbolic/linkedasp_parser.py`)
- [ ] Create query/analysis tools (`loft/symbolic/linkedasp_queries.py`)
- [ ] Refactor statute of frauds to use genres
- [ ] Add SPARQL query CLI commands
- [ ] Integrate with meta-reasoning layer
- [ ] Add stratification validation to CI/CD
- [ ] Document genre pattern library
- [ ] Create visual dependency graph tool

---

## References

This approach synthesizes ideas from:

1. **G-Lisp** (../glisp-stuff/glisp): Genre-based abstraction (`{:g :v :m}`), `gexpand` meta-programming
2. **GraphFS** (../graphfs): LinkedDoc+RDF metadata, SPARQL queries, semantic code navigation
3. **LOFT ROADMAP.md**: Self-reflexive symbolic core, stratified architecture, ontological bridge
4. **LOFT CLAUDE.md**: Validation oversight, composability, self-modification safeguards

---

## Conclusion

The **LinkedASP + Genre-Based Generation** approach transforms ASP from a potential maintainability nightmare into a **self-documenting, queryable, and compositional** foundation for LOFT's self-reflexive architecture.

By embedding rich RDF metadata and generating code from high-level legal reasoning patterns, we:
- **Prevent complexity explosion** as we add legal domains
- **Enable self-reflexivity**: The core can reason about itself
- **Strengthen the ontological bridge**: Semantic metadata guides LLM interactions
- **Maintain validation oversight**: Automated detection of violations
- **Support composability**: Genre patterns reused across domains

This is not just a maintainability improvement—it's an **architectural enhancement** that makes LOFT's self-reflexive vision more achievable.

**Recommendation**: Implement this as a **tangential route in Phase 1.5+**, after basic LLM integration (Phase 2) but before multi-domain expansion becomes unwieldy (Phase 8).
