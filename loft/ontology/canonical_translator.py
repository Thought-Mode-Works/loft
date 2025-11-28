"""
Canonical predicate translator for cross-domain ASP rule translation.

This module provides deterministic translation between domain-specific
predicates and canonical legal concepts using RDF ontology mappings.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from rdflib import Graph, Namespace

    RDFLIB_AVAILABLE = True
    # Define namespaces (only if rdflib is available)
    CANON = Namespace("https://loft.legal/canonical/")
    LOFT = Namespace("https://loft.legal/ontology/")
except ImportError:
    RDFLIB_AVAILABLE = False
    Graph = None
    Namespace = None
    CANON = None
    LOFT = None


class CanonicalTranslator:
    """
    Translates ASP predicates between domains using canonical vocabulary.

    Uses RDF ontology to maintain semantic fidelity during cross-domain
    rule translation. Supports bidirectional mapping:
    - Domain predicate → Canonical predicate
    - Canonical predicate → Domain predicate

    Example:
        translator = CanonicalTranslator()
        translated = translator.translate_rule(
            rule="enforceable(X) :- occupation_continuous(X, yes).",
            source_domain="adverse_possession",
            target_domain="property_law"
        )
        # Result: "enforceable(X) :- use_continuous(X, yes)."
    """

    def __init__(self, ontology_path: Optional[Path] = None):
        """
        Initialize translator with ontology.

        Args:
            ontology_path: Path to canonical_predicates.ttl file.
                          If None, uses default location in package.
        """
        if not RDFLIB_AVAILABLE:
            raise ImportError(
                "rdflib is required for CanonicalTranslator. Install with: pip install rdflib"
            )

        if ontology_path is None:
            ontology_path = Path(__file__).parent / "canonical_predicates.ttl"

        self.ontology_path = ontology_path
        self.graph = Graph()

        # Mapping tables
        # domain -> predicate_name -> canonical_predicate
        self._domain_to_canonical: Dict[str, Dict[str, str]] = {}
        # canonical_predicate -> domain -> predicate_name
        self._canonical_to_domain: Dict[str, Dict[str, str]] = {}
        # canonical_predicate -> metadata (arity, argTypes, legalCategory)
        self._canonical_metadata: Dict[str, Dict] = {}

        self._load_ontology()

    def _load_ontology(self) -> None:
        """Load and parse the RDF ontology file."""
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")

        self.graph.parse(self.ontology_path, format="turtle")
        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build bidirectional mapping tables from RDF graph."""
        # Query for domain predicates and their canonical mappings
        query = """
        PREFIX canon: <https://loft.legal/canonical/>
        PREFIX loft: <https://loft.legal/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?domain_pred ?domain ?canonical
        WHERE {
            ?domain_pred a canon:DomainPredicate ;
                        canon:mapsTo ?canonical ;
                        loft:domain ?domain .
        }
        """

        for row in self.graph.query(query):
            domain_pred_uri = str(row.domain_pred)
            domain = str(row.domain)
            canonical_uri = str(row.canonical)

            # Extract predicate name from URI
            # e.g., "https://loft.legal/domains/adverse_possession/claim" -> "claim"
            pred_name = domain_pred_uri.split("/")[-1]
            canonical_name = canonical_uri.split("/")[-1]

            # Build domain -> canonical mapping
            if domain not in self._domain_to_canonical:
                self._domain_to_canonical[domain] = {}
            self._domain_to_canonical[domain][pred_name] = canonical_name

            # Build canonical -> domain mapping
            if canonical_name not in self._canonical_to_domain:
                self._canonical_to_domain[canonical_name] = {}
            self._canonical_to_domain[canonical_name][domain] = pred_name

        # Load canonical predicate metadata
        metadata_query = """
        PREFIX canon: <https://loft.legal/canonical/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?pred ?arity ?argTypes ?category ?label
        WHERE {
            ?pred a canon:CanonicalPredicate .
            OPTIONAL { ?pred canon:arity ?arity }
            OPTIONAL { ?pred canon:argTypes ?argTypes }
            OPTIONAL { ?pred canon:legalCategory ?category }
            OPTIONAL { ?pred rdfs:label ?label }
        }
        """

        for row in self.graph.query(metadata_query):
            pred_uri = str(row.pred)
            pred_name = pred_uri.split("/")[-1]

            self._canonical_metadata[pred_name] = {
                "arity": int(row.arity) if row.arity else None,
                "arg_types": str(row.argTypes) if row.argTypes else None,
                "legal_category": str(row.category) if row.category else None,
                "label": str(row.label) if row.label else pred_name,
            }

    def get_domains(self) -> List[str]:
        """Return list of all domains defined in ontology."""
        return list(self._domain_to_canonical.keys())

    def get_canonical_predicates(self) -> List[str]:
        """Return list of all canonical predicates."""
        return list(self._canonical_metadata.keys())

    def get_domain_predicates(self, domain: str) -> List[str]:
        """Return list of predicates for a specific domain."""
        return list(self._domain_to_canonical.get(domain, {}).keys())

    def to_canonical(self, predicate: str, domain: str) -> Optional[str]:
        """
        Translate domain predicate to canonical form.

        Args:
            predicate: Domain-specific predicate name
            domain: Source domain

        Returns:
            Canonical predicate name, or None if no mapping exists
        """
        domain_map = self._domain_to_canonical.get(domain, {})
        return domain_map.get(predicate)

    def from_canonical(self, canonical: str, domain: str) -> Optional[str]:
        """
        Translate canonical predicate to domain-specific form.

        Args:
            canonical: Canonical predicate name
            domain: Target domain

        Returns:
            Domain-specific predicate name, or None if no mapping exists
        """
        canonical_map = self._canonical_to_domain.get(canonical, {})
        return canonical_map.get(domain)

    def translate_predicate(
        self, predicate: str, source_domain: str, target_domain: str
    ) -> Optional[str]:
        """
        Translate predicate from source domain to target domain.

        Args:
            predicate: Predicate name in source domain
            source_domain: Source domain name
            target_domain: Target domain name

        Returns:
            Translated predicate name, or None if translation not possible
        """
        # First translate to canonical
        canonical = self.to_canonical(predicate, source_domain)
        if canonical is None:
            return None

        # Then translate to target domain
        return self.from_canonical(canonical, target_domain)

    def translate_rule(
        self,
        rule: str,
        source_domain: str,
        target_domain: str,
        strict: bool = False,
    ) -> Tuple[str, List[str], List[str]]:
        """
        Translate an ASP rule from source domain to target domain.

        Args:
            rule: ASP rule string
            source_domain: Source domain name
            target_domain: Target domain name
            strict: If True, fail on untranslatable predicates

        Returns:
            Tuple of:
            - Translated rule string
            - List of successfully translated predicates
            - List of untranslatable predicates (kept as-is)
        """
        translated_rule = rule
        translated_predicates = []
        untranslatable_predicates = []

        # Find all predicate names in the rule
        # Pattern matches: predicate_name(
        predicate_pattern = r"([a-z][a-z0-9_]*)\s*\("
        predicates_found = set(re.findall(predicate_pattern, rule))

        for pred in predicates_found:
            target_pred = self.translate_predicate(pred, source_domain, target_domain)

            if target_pred is not None and target_pred != pred:
                # Replace predicate in rule (with word boundary to avoid partial matches)
                pattern = rf"\b{re.escape(pred)}\s*\("
                replacement = f"{target_pred}("
                translated_rule = re.sub(pattern, replacement, translated_rule)
                translated_predicates.append(f"{pred} -> {target_pred}")
            elif target_pred is None:
                untranslatable_predicates.append(pred)

        if strict and untranslatable_predicates:
            raise ValueError(f"Cannot translate predicates: {untranslatable_predicates}")

        return translated_rule, translated_predicates, untranslatable_predicates

    def get_canonical_metadata(self, canonical: str) -> Optional[Dict]:
        """
        Get metadata for a canonical predicate.

        Args:
            canonical: Canonical predicate name

        Returns:
            Dictionary with arity, arg_types, legal_category, label
        """
        return self._canonical_metadata.get(canonical)

    def find_common_canonical(self, domain1: str, domain2: str) -> Set[str]:
        """
        Find canonical predicates that have mappings in both domains.

        Args:
            domain1: First domain
            domain2: Second domain

        Returns:
            Set of canonical predicate names present in both domains
        """
        preds1 = set(self._domain_to_canonical.get(domain1, {}).values())
        preds2 = set(self._domain_to_canonical.get(domain2, {}).values())
        return preds1.intersection(preds2)

    def get_translation_coverage(self, source_domain: str, target_domain: str) -> Dict:
        """
        Calculate translation coverage between two domains.

        Args:
            source_domain: Source domain
            target_domain: Target domain

        Returns:
            Dictionary with coverage statistics
        """
        source_preds = set(self._domain_to_canonical.get(source_domain, {}).keys())
        target_preds = set(self._domain_to_canonical.get(target_domain, {}).keys())

        common = self.find_common_canonical(source_domain, target_domain)

        # Count how many source predicates can be translated
        translatable = 0
        for pred in source_preds:
            canonical = self.to_canonical(pred, source_domain)
            if canonical in common:
                translatable += 1

        coverage = translatable / len(source_preds) if source_preds else 0.0

        return {
            "source_predicates": len(source_preds),
            "target_predicates": len(target_preds),
            "common_canonical": len(common),
            "translatable_count": translatable,
            "coverage_ratio": coverage,
            "common_predicates": list(common),
        }

    def to_dict(self) -> Dict:
        """Export mappings as dictionary for serialization."""
        return {
            "domain_to_canonical": self._domain_to_canonical,
            "canonical_to_domain": self._canonical_to_domain,
            "canonical_metadata": self._canonical_metadata,
        }

    def __repr__(self) -> str:
        domains = self.get_domains()
        canonical_count = len(self._canonical_metadata)
        return f"CanonicalTranslator(domains={domains}, canonical_predicates={canonical_count})"
