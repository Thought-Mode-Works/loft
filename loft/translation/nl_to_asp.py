"""
Natural Language to ASP Translation.

This module implements the reverse translation layer that converts LLM
natural language responses back into ASP facts and rules.
"""

from typing import List, Optional, Type, TypeVar, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel

from .schemas import ContractFact, ExtractedEntities, LegalRule
from .patterns import (
    quick_extract_facts,
    extract_parties,
    extract_contract_type,
    extract_essential_elements,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class NLToASPResult:
    """Result of NL → ASP translation."""

    asp_facts: List[str]
    source_nl: str
    confidence: float = 0.7
    extraction_method: str = "pattern"  # "pattern", "llm", "hybrid"
    ambiguities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def nl_to_structured(
    nl_response: str,
    schema: Type[T],
    llm_interface: Optional[Any] = None,
) -> T:
    """
    Parse LLM natural language response into structured format.

    Uses Pydantic schema to guide parsing and validation.
    The structured data can then be converted to ASP facts.

    Args:
        nl_response: Natural language text to parse
        schema: Pydantic model class to parse into
        llm_interface: Optional LLM interface for extraction (if None, uses patterns)

    Returns:
        Instance of the Pydantic schema with extracted data

    Example:
        >>> from loft.translation.schemas import ContractFact
        >>> nl = "John and Mary have a land sale contract for $500,000"
        >>> contract = nl_to_structured(nl, ContractFact)
        >>> print(contract.contract_id)
        'contract_1'
        >>> print(contract.sale_amount)
        500000.0
    """
    if llm_interface is not None:
        # Use LLM for extraction with structured output
        response = llm_interface.query(
            question=f"Extract legal entities from this text: {nl_response}",
            output_schema=schema,
        )
        return response.content

    # Fallback: parse using patterns and heuristics
    if schema == ContractFact:
        return _parse_contract_fact_from_nl(nl_response)
    elif schema == ExtractedEntities:
        return _parse_extracted_entities_from_nl(nl_response)

    raise ValueError(f"Unsupported schema for pattern-based parsing: {schema}")


def _parse_contract_fact_from_nl(nl_text: str) -> ContractFact:
    """Parse ContractFact from natural language using patterns."""
    contract_type = extract_contract_type(nl_text)
    parties = extract_parties(nl_text)
    elements = extract_essential_elements(nl_text)

    # Extract sale amount if mentioned
    import re

    sale_match = re.search(r"\$?([\d,]+(?:\.\d{2})?)", nl_text)
    sale_amount = None
    if sale_match:
        amount_str = sale_match.group(1).replace(",", "")
        sale_amount = float(amount_str)

    return ContractFact(
        contract_id="contract_1",
        contract_type=contract_type if contract_type != "general" else None,
        parties=parties,
        has_writing=elements["has_writing"],
        is_signed=elements["is_signed"],
        sale_amount=sale_amount,
        has_consideration=elements["has_consideration"],
        has_mutual_assent=elements["has_mutual_assent"],
    )


def _parse_extracted_entities_from_nl(nl_text: str) -> ExtractedEntities:
    """Parse ExtractedEntities from natural language using patterns."""
    contract = _parse_contract_fact_from_nl(nl_text)

    from .schemas import Party

    parties = [Party(name=p) for p in extract_parties(nl_text)]

    return ExtractedEntities(
        contracts=[contract],
        parties=parties,
        writings=[],
        relationships=[],
    )


def nl_to_asp_facts(
    nl_text: str,
    llm_interface: Optional[Any] = None,
    use_llm: bool = False,
) -> List[str]:
    """
    Extract ASP facts from natural language.

    Args:
        nl_text: Natural language text to parse
        llm_interface: Optional LLM interface for extraction
        use_llm: Whether to use LLM (if available) or pattern matching

    Returns:
        List of ASP facts

    Example:
        >>> nl = "The contract was signed by John and includes all essential terms."
        >>> facts = nl_to_asp_facts(nl)
        >>> print(facts)
        ['signed_by(the_contract, john).', 'contains(the_contract, all_essential_terms).']

        >>> nl = "This is a land sale contract for $500,000."
        >>> facts = nl_to_asp_facts(nl)
        >>> print(facts)
        ['land_sale_contract(contract_1).', 'sale_amount(contract_1, 500000).']
    """
    if use_llm and llm_interface is not None:
        # Use LLM for structured extraction
        entities = nl_to_structured(nl_text, ExtractedEntities, llm_interface)
        return entities.to_asp()

    # Use pattern-based extraction
    return quick_extract_facts(nl_text)


def nl_to_asp_rule(
    nl_rule: str,
    llm_interface: Optional[Any] = None,
) -> str:
    """
    Convert natural language rule to ASP rule.

    This is a basic version for Phase 1. Full implementation in Phase 2.

    Args:
        nl_rule: Natural language description of a rule
        llm_interface: Optional LLM interface for extraction

    Returns:
        ASP rule string

    Example:
        >>> rule = "A contract satisfies statute of frauds if it has a signed writing."
        >>> asp_rule = nl_to_asp_rule(rule)
        >>> print(asp_rule)
        'satisfies_statute_of_frauds(C) :- has_writing(C, W), signed_by(W, _).'

        >>> rule = "A contract is enforceable by default unless proven otherwise."
        >>> asp_rule = nl_to_asp_rule(rule)
        >>> print(asp_rule)
        'enforceable(C) :- contract(C), not unenforceable(C).'
    """
    if llm_interface is not None:
        # Use LLM for rule extraction
        legal_rule = nl_to_structured(nl_rule, LegalRule, llm_interface)
        return legal_rule.to_asp()

    # Basic pattern-based rule extraction
    return _parse_rule_from_nl_basic(nl_rule)


def _parse_rule_from_nl_basic(nl_rule: str) -> str:
    """
    Basic pattern-based rule extraction.

    Handles simple cases for Phase 1 MVP.
    """
    import re

    nl_lower = nl_rule.lower()

    # Pattern: "A X is Y if Z"
    # Example: "A contract satisfies statute of frauds if it has a signed writing"
    match = re.search(r"a\s+(\w+)\s+([\w\s]+?)\s+if\s+(.*)", nl_lower)
    if match:
        entity_var = match.group(1)[0].upper()  # "contract" → "C"
        head_predicate = match.group(2).strip().replace(" ", "_")
        body_text = match.group(3).strip()

        # Parse body conditions
        body_conditions = []

        # Handle "it has X"
        if "it has" in body_text:
            # Extract what it has
            has_match = re.search(r"it has (?:a\s+)?(\w+(?:\s+\w+)*)", body_text)
            if has_match:
                what = has_match.group(1).replace(" ", "_")
                body_conditions.append(f"has_{what}({entity_var}, _)")

        # Handle "it is signed" or "signed"
        if "signed" in body_text:
            if "has_writing" in str(body_conditions):
                # Add signed_by condition
                body_conditions.append("signed_by(_, _)")
            else:
                body_conditions.append(f"signed({entity_var})")

        # Default: try to extract predicates
        if not body_conditions:
            words = body_text.replace(" and ", " ").split()
            for word in words:
                if word.isalpha() and len(word) > 2:
                    body_conditions.append(f"{word}({entity_var})")

        body = ", ".join(body_conditions) if body_conditions else f"{entity_var}={entity_var}"

        return f"{head_predicate}({entity_var}) :- {body}."

    # Pattern: "A X is Y unless Z" (negation-as-failure)
    # Example: "A contract is enforceable unless proven unenforceable"
    match = re.search(r"a\s+(\w+)\s+is\s+([\w\s]+?)\s+unless\s+(.*)", nl_lower)
    if match:
        entity_var = match.group(1)[0].upper()
        head_predicate = match.group(2).strip().replace(" ", "_")
        negated_condition = match.group(3).strip()

        # Extract negated predicate
        neg_pred = negated_condition.replace("proven ", "").replace(" ", "_")

        return f"{head_predicate}({entity_var}) :- {match.group(1)}({entity_var}), not {neg_pred}({entity_var})."

    # Fallback: return as comment
    return f"% Could not parse rule: {nl_rule}"


class NLToASPTranslator:
    """
    Main interface for NL → ASP translation.

    Handles entity extraction, rule parsing, and validation.
    """

    def __init__(
        self,
        llm_interface: Optional[Any] = None,
        use_llm_by_default: bool = False,
    ):
        """
        Initialize translator.

        Args:
            llm_interface: Optional LLM interface for extraction
            use_llm_by_default: Whether to use LLM by default (vs patterns)
        """
        self.llm_interface = llm_interface
        self.use_llm = use_llm_by_default and llm_interface is not None

    def translate_to_facts(self, nl_text: str) -> NLToASPResult:
        """
        Translate natural language to ASP facts.

        Args:
            nl_text: Natural language text

        Returns:
            NLToASPResult with extracted facts and metadata
        """
        facts = nl_to_asp_facts(nl_text, self.llm_interface, self.use_llm)

        return NLToASPResult(
            asp_facts=facts,
            source_nl=nl_text,
            confidence=0.8 if self.use_llm else 0.6,
            extraction_method="llm" if self.use_llm else "pattern",
        )

    def translate_to_rule(self, nl_rule: str) -> NLToASPResult:
        """
        Translate natural language rule to ASP.

        Args:
            nl_rule: Natural language rule description

        Returns:
            NLToASPResult with extracted rule
        """
        asp_rule = nl_to_asp_rule(nl_rule, self.llm_interface)

        return NLToASPResult(
            asp_facts=[asp_rule],
            source_nl=nl_rule,
            confidence=0.7 if self.use_llm else 0.5,
            extraction_method="llm" if self.use_llm else "pattern",
        )

    def translate(self, nl_text: str) -> NLToASPResult:
        """
        Convenience method for translation. Automatically chooses between
        translate_to_facts or translate_to_rule based on input.

        Args:
            nl_text: Natural language text or rule

        Returns:
            NLToASPResult with ASP code
        """
        # Simple heuristic: if it looks like a rule (has "if", "when", etc.), translate as rule
        rule_keywords = ["if ", "when ", "unless ", "provided that", "except"]
        is_rule = any(keyword in nl_text.lower() for keyword in rule_keywords)

        if is_rule:
            return self.translate_to_rule(nl_text)
        else:
            return self.translate_to_rule(nl_text)  # Default to rule for most statements

    def extract_entities(self, nl_text: str) -> ExtractedEntities:
        """
        Extract structured entities from natural language.

        Args:
            nl_text: Natural language text

        Returns:
            ExtractedEntities with all found entities
        """
        return nl_to_structured(nl_text, ExtractedEntities, self.llm_interface)
