"""
Natural Language to ASP Translation.

This module implements the reverse translation layer that converts LLM
natural language responses back into ASP facts and rules.
"""

import re
from typing import List, Optional, Type, TypeVar, Dict, Any, FrozenSet
from dataclasses import dataclass, field
from pydantic import BaseModel
from loguru import logger

from .schemas import ContractFact, ExtractedEntities, LegalRule
from .patterns import (
    quick_extract_facts,
    extract_parties,
    extract_contract_type,
    extract_essential_elements,
)
from .utils import (
    CANONICAL_PREDICATES,
    PREDICATE_ALIASES,
    LEGAL_CONCEPT_TO_PREDICATE,
    normalize_predicate,
    extract_predicates_from_asp,
    validate_predicates,
    find_closest_predicate,
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

    @property
    def asp_code(self) -> str:
        """Get ASP code as a single string (convenience property for tests)."""
        return " ".join(self.asp_facts) if self.asp_facts else ""

    @property
    def predicates_used(self) -> List[str]:
        """Extract predicates used in ASP facts."""
        # Simple extraction - get predicate names from facts
        predicates = []
        for fact in self.asp_facts:
            # Extract predicate name (text before '(')
            if "(" in fact:
                pred = fact.split("(")[0].strip()
                if pred and pred not in predicates:
                    predicates.append(pred)
        return predicates


def nl_to_structured(
    nl_response: str,
    schema: Type[T],
    llm_interface: Optional[Any] = None,
) -> T | List[T]:
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
        # Determine prompt based on schema
        from .schemas import LegalRule
        
        is_rule_schema = False
        try:
            # Check if schema is LegalRule or List[LegalRule]
            if schema == LegalRule or (hasattr(schema, "__origin__") and schema.__origin__ is list and schema.__args__[0] == LegalRule):
                is_rule_schema = True
        except (AttributeError, IndexError):
            pass

        if is_rule_schema:
            prompt = (
                f"Extract legal rules from this text into ASP (Answer Set Programming) structure: {nl_response}\n"
                "Ensure 'head_predicate' and 'body_conditions' use valid ASP syntax (lowercase, underscores, variables like C, X).\n"
                "For negation, use 'not predicate(X)' in body_conditions. "
                "For negative heads, use 'minus_predicate(X)' or 'unenforceable(X)' as appropriate.\n"
                "IMPORTANT: You MUST use specific predicates for contract types:\n"
                "- 'land_sale_contract(C)' for land sale contracts\n"
                "- 'suretyship_agreement(C)' for suretyship agreements\n"
                "- 'goods_contract(C)' for sale of goods\n"
                "Do not just use 'contract(C)' for these.\n"
                "For writing requirements, use 'requires_writing(X)' as the head predicate."
            )
        else:
            prompt = f"Extract legal entities from this text: {nl_response}"

        # Use LLM for extraction with structured output
        response = llm_interface.query(
            question=prompt,
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
) -> List[str]:
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
        legal_rules = nl_to_structured(nl_rule, List[LegalRule], llm_interface)
        if isinstance(legal_rules, list):
            return [rule.to_asp() for rule in legal_rules]
        return [legal_rules.to_asp()]

    # Basic pattern-based rule extraction
    return [_parse_rule_from_nl_basic(nl_rule)]


def _parse_rule_from_nl_basic(nl_rule: str) -> str:
    """
    Basic pattern-based rule extraction.

    Handles simple cases for Phase 1 MVP.
    """
    import re

    nl_lower = nl_rule.lower()
    logger.debug(f"nl_lower: {nl_lower}")

    # Pattern: "A X is not Y if Z"
    pattern1 = r"a\s+(\w+)\s+is\s+not\s+([\w\s]+?)\s+if\s+(.*)"
    logger.debug(f"pattern1: {pattern1}")
    match = re.search(pattern1, nl_lower)
    logger.debug(f"match1: {match}")
    if match:
        entity_var = match.group(1)[0].upper()
        head_predicate = f"unenforceable({entity_var})"
        body_text = match.group(3).strip()
        body_conditions = []
        if "not in writing" in body_text:
            body_conditions.append(f"not has_writing({entity_var})")
        else:
            words = body_text.replace(" and ", " ").split()
            for word in words:
                if word.isalpha() and len(word) > 2:
                    body_conditions.append(f"{word}({entity_var})")
        body = ", ".join(body_conditions)
        return f"{head_predicate} :- {body}."

    # Pattern: "A X is Y if Z"
    # Example: "A contract satisfies statute of frauds if it has a signed writing"
    pattern2 = r"a\s+(\w+)\s+([\w\s]+?)\s+if\s+(.*)"
    logger.debug(f"pattern2: {pattern2}")
    match = re.search(pattern2, nl_lower)
    logger.debug(f"match2: {match}")
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

        body = (
            ", ".join(body_conditions)
            if body_conditions
            else f"{entity_var}={entity_var}"
        )

        return f"{head_predicate}({entity_var}) :- {body}."

    # Pattern: "A X is Y unless Z" (negation-as-failure)
    # Example: "A contract is enforceable unless proven unenforceable"
    pattern3 = r"a\s+(\w+)\s+is\s+([\w\s]+?)\s+unless\s+(.*)"
    logger.debug(f"pattern3: {pattern3}")
    match = re.search(pattern3, nl_lower)
    logger.debug(f"match3: {match}")
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
        asp_rules = nl_to_asp_rule(nl_rule, self.llm_interface)

        return NLToASPResult(
            asp_facts=asp_rules,
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
            return self.translate_to_rule(
                nl_text
            )  # Default to rule for most statements

    def extract_entities(self, nl_text: str) -> ExtractedEntities:
        """
        Extract structured entities from natural language.

        Args:
            nl_text: Natural language text

        Returns:
            ExtractedEntities with all found entities
        """
        return nl_to_structured(nl_text, ExtractedEntities, self.llm_interface)


@dataclass
class PredicateValidationResult:
    """Result of predicate validation and correction."""

    original_asp: str
    corrected_asp: str
    valid_predicates: List[str]
    invalid_predicates: List[str]
    corrections_made: Dict[str, str]  # Maps invalid -> corrected
    compliance_rate: float  # Percentage of predicates that were valid
    is_fully_compliant: bool

    def __post_init__(self):
        """Calculate compliance rate if not set."""
        total = len(self.valid_predicates) + len(self.invalid_predicates)
        if total > 0 and self.compliance_rate == 0.0:
            self.compliance_rate = len(self.valid_predicates) / total


class ConstrainedNLToASPTranslator:
    """
    NL→ASP translator that enforces a constrained predicate vocabulary.

    Forces the LLM to use only canonical predicates, improving round-trip
    translation fidelity by ensuring ASP→NL templates can match.

    Attributes:
        llm_interface: LLM interface for translation
        vocabulary: Set of allowed predicates
        auto_correct: Whether to automatically correct invalid predicates
        strict_mode: If True, reject translations with uncorrectable predicates
    """

    def __init__(
        self,
        llm_interface: Optional[Any] = None,
        vocabulary: FrozenSet[str] = CANONICAL_PREDICATES,
        auto_correct: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize constrained translator.

        Args:
            llm_interface: LLM interface for translation
            vocabulary: Set of allowed predicates (default: CANONICAL_PREDICATES)
            auto_correct: Whether to auto-correct invalid predicates
            strict_mode: If True, raise error for uncorrectable predicates
        """
        self.llm_interface = llm_interface
        self.vocabulary = vocabulary
        self.auto_correct = auto_correct
        self.strict_mode = strict_mode

        # Fallback to base translator for pattern-based extraction
        self._base_translator = NLToASPTranslator(llm_interface)

    def _build_constrained_prompt(self, nl_text: str) -> str:
        """
        Build a prompt that constrains LLM to use canonical predicates.

        Args:
            nl_text: Natural language text to translate

        Returns:
            Prompt string with vocabulary constraints
        """
        sorted_predicates = sorted(self.vocabulary)
        predicate_list = ", ".join(sorted_predicates)

        return f"""Translate this legal statement to ASP (Answer Set Programming) using ONLY these predicates:

ALLOWED PREDICATES:
{predicate_list}

RULES:
1. DO NOT invent new predicates - only use predicates from the list above
2. Use snake_case for all predicates (e.g., has_offer, not HasOffer)
3. Variables should be single uppercase letters (X, Y, C, W)
4. Map legal concepts to the closest canonical predicate
5. If no exact predicate exists, use the closest semantic match

CONCEPT MAPPINGS:
- "offer" → has_offer(X)
- "acceptance" → has_acceptance(X)
- "consideration" → has_consideration(X)
- "must be in writing" → requires_writing(X)
- "land sale contract" → land_sale(X)
- "statute of frauds" → satisfies_sof(X)
- "part performance" → part_performance(X)

STATEMENT TO TRANSLATE:
{nl_text}

ASP OUTPUT (use only allowed predicates):"""

    def translate(self, nl_text: str) -> NLToASPResult:
        """
        Translate NL to ASP with vocabulary constraints.

        Args:
            nl_text: Natural language text to translate

        Returns:
            NLToASPResult with validated/corrected ASP code
        """
        # If no LLM, use pattern-based translation
        if self.llm_interface is None:
            base_result = self._base_translator.translate(nl_text)
            validation = self._validate_and_correct(base_result.asp_code)

            return NLToASPResult(
                asp_facts=[validation.corrected_asp],
                source_nl=nl_text,
                confidence=base_result.confidence * validation.compliance_rate,
                extraction_method="pattern_constrained",
                metadata={
                    "validation": {
                        "valid_predicates": validation.valid_predicates,
                        "invalid_predicates": validation.invalid_predicates,
                        "corrections": validation.corrections_made,
                        "compliance_rate": validation.compliance_rate,
                    }
                },
            )

        # Use LLM with constrained prompt
        prompt = self._build_constrained_prompt(nl_text)

        try:
            response = self.llm_interface.query(prompt)
            asp_code = (
                response.content if hasattr(response, "content") else str(response)
            )
        except Exception as e:
            logger.warning(f"LLM translation failed: {e}, falling back to patterns")
            base_result = self._base_translator.translate(nl_text)
            asp_code = base_result.asp_code

        # Validate and correct
        validation = self._validate_and_correct(asp_code)

        # If not fully compliant and auto_correct is off, try LLM correction
        if not validation.is_fully_compliant and self.llm_interface is not None:
            validation = self._request_llm_correction(asp_code, validation)

        # Check strict mode
        if self.strict_mode and not validation.is_fully_compliant:
            raise ValueError(
                f"Translation contains invalid predicates that could not be corrected: "
                f"{validation.invalid_predicates}"
            )

        return NLToASPResult(
            asp_facts=[validation.corrected_asp],
            source_nl=nl_text,
            confidence=0.9 * validation.compliance_rate,
            extraction_method="llm_constrained",
            metadata={
                "validation": {
                    "valid_predicates": validation.valid_predicates,
                    "invalid_predicates": validation.invalid_predicates,
                    "corrections": validation.corrections_made,
                    "compliance_rate": validation.compliance_rate,
                    "original_asp": validation.original_asp,
                }
            },
        )

    def _validate_and_correct(self, asp_code: str) -> PredicateValidationResult:
        """
        Validate ASP code predicates and attempt automatic correction.

        Args:
            asp_code: ASP code to validate

        Returns:
            PredicateValidationResult with corrections applied
        """
        # Extract predicates from ASP
        predicates = extract_predicates_from_asp(asp_code)

        # Validate against vocabulary
        valid, invalid = validate_predicates(predicates, self.vocabulary)

        # Attempt corrections
        corrections: Dict[str, str] = {}
        corrected_asp = asp_code

        if self.auto_correct and invalid:
            for inv_pred in invalid:
                # First try normalization
                normalized = normalize_predicate(inv_pred)
                if normalized in self.vocabulary:
                    corrections[inv_pred] = normalized
                    # Replace in ASP code (word boundary match)
                    corrected_asp = re.sub(
                        rf"\b{re.escape(inv_pred)}\s*\(",
                        f"{normalized}(",
                        corrected_asp,
                    )
                else:
                    # Try to find closest match
                    closest = find_closest_predicate(inv_pred, self.vocabulary)
                    if closest:
                        corrections[inv_pred] = closest
                        corrected_asp = re.sub(
                            rf"\b{re.escape(inv_pred)}\s*\(",
                            f"{closest}(",
                            corrected_asp,
                        )

        # Recalculate valid/invalid after corrections
        corrected_predicates = extract_predicates_from_asp(corrected_asp)
        final_valid, final_invalid = validate_predicates(
            corrected_predicates, self.vocabulary
        )

        total = len(final_valid) + len(final_invalid)
        compliance = len(final_valid) / total if total > 0 else 1.0

        return PredicateValidationResult(
            original_asp=asp_code,
            corrected_asp=corrected_asp,
            valid_predicates=final_valid,
            invalid_predicates=final_invalid,
            corrections_made=corrections,
            compliance_rate=compliance,
            is_fully_compliant=len(final_invalid) == 0,
        )

    def _request_llm_correction(
        self,
        original_asp: str,
        validation: PredicateValidationResult,
    ) -> PredicateValidationResult:
        """
        Request LLM to correct invalid predicates.

        Args:
            original_asp: Original ASP code
            validation: Current validation result

        Returns:
            Updated PredicateValidationResult after LLM correction
        """
        if not validation.invalid_predicates or self.llm_interface is None:
            return validation

        correction_prompt = f"""The following ASP code contains invalid predicates that are not in the allowed vocabulary.

INVALID PREDICATES: {', '.join(validation.invalid_predicates)}

ALLOWED PREDICATES:
{', '.join(sorted(self.vocabulary))}

ORIGINAL ASP:
{original_asp}

Rewrite the ASP code using ONLY allowed predicates. Replace each invalid predicate with the closest semantic match from the allowed list.

CORRECTED ASP:"""

        try:
            response = self.llm_interface.query(correction_prompt)
            corrected_code = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Re-validate the corrected code
            return self._validate_and_correct(corrected_code)
        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")
            return validation

    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the predicate vocabulary.

        Returns:
            Dictionary with vocabulary statistics
        """
        return {
            "total_predicates": len(self.vocabulary),
            "categories": {
                "contract_formation": len(
                    [
                        p
                        for p in self.vocabulary
                        if "offer" in p or "accept" in p or "consider" in p
                    ]
                ),
                "statute_of_frauds": len(
                    [
                        p
                        for p in self.vocabulary
                        if "sof" in p or "writing" in p or "land" in p
                    ]
                ),
                "exceptions": len(
                    [
                        p
                        for p in self.vocabulary
                        if "exception" in p or "performance" in p
                    ]
                ),
                "enforceability": len(
                    [
                        p
                        for p in self.vocabulary
                        if "enforce" in p or "valid" in p or "void" in p
                    ]
                ),
            },
            "aliases_count": len(PREDICATE_ALIASES),
            "concept_mappings_count": len(LEGAL_CONCEPT_TO_PREDICATE),
        }


def get_predicate_compliance_rate(
    asp_code: str,
    vocabulary: FrozenSet[str] = CANONICAL_PREDICATES,
) -> float:
    """
    Calculate predicate compliance rate for ASP code.

    Args:
        asp_code: ASP code to analyze
        vocabulary: Canonical vocabulary to check against

    Returns:
        Compliance rate as float between 0 and 1

    Examples:
        >>> rate = get_predicate_compliance_rate("has_offer(X) :- contract(X).")
        >>> print(f"{rate:.2%}")
        '100.00%'
    """
    predicates = extract_predicates_from_asp(asp_code)
    if not predicates:
        return 1.0

    valid, invalid = validate_predicates(predicates, vocabulary)
    return len(valid) / (len(valid) + len(invalid))
