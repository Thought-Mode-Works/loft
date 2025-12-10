"""
Utilities for translation.
"""

import re
from typing import List, Optional, Dict, FrozenSet, Set

# =============================================================================
# Canonical Predicate Vocabulary
# =============================================================================
#
# This vocabulary defines the set of allowed predicates for NL→ASP translation.
# Using a constrained vocabulary ensures bidirectional template matching between
# NL→ASP and ASP→NL translations, improving round-trip fidelity.
#
# Categories:
#   - Contract Formation: Core elements of valid contracts
#   - Statute of Frauds: Writing requirements and triggers
#   - Exceptions: Ways to satisfy/bypass SOF requirements
#   - Signatures: Electronic and traditional signature predicates
#   - Enforceability: Contract validity states
#   - Parties & Entities: Contract participants
#   - UCC Specific: Uniform Commercial Code predicates

CANONICAL_PREDICATES: FrozenSet[str] = frozenset(
    {
        # Contract Formation (core elements)
        "contract",
        "contract_valid",
        "has_offer",
        "has_acceptance",
        "has_consideration",
        "has_mutual_assent",
        "legal_capacity",
        "has_writing",
        "signed",
        "signed_by",
        # Statute of Frauds (writing requirement triggers)
        "requires_writing",
        "satisfies_sof",
        "land_sale",
        "land_sale_contract",
        "suretyship",
        "suretyship_agreement",
        "goods_over_500",
        "cannot_perform_within_year",
        "within_one_year",
        "marriage_promise",
        "executor_promise",
        # Exceptions (ways to satisfy/bypass SOF)
        "part_performance",
        "promissory_estoppel",
        "merchant_confirmation",
        "specially_manufactured",
        "specially_manufactured_goods",
        "court_admission",
        "admission_in_court",
        "written_memorandum",
        "exception",
        "exception_applies",
        # Signatures
        "electronic_signature",
        "valid_signature",
        "esign_applies",
        "ueta_applies",
        # Enforceability states
        "enforceable",
        "unenforceable",
        "valid",
        "invalid",
        "void",
        "voidable",
        "binding",
        # Parties and entities
        "party",
        "merchant",
        "buyer",
        "seller",
        "promisor",
        "promisee",
        "involved_in",
        # UCC specific
        "ucc_applies",
        "goods",
        "sale_of_goods",
        "price",
        "quantity",
        # Additional common predicates
        "offer",
        "acceptance",
        "consideration",
        "writing",
        "memorandum",
    }
)

# Predicate aliases for normalization (maps variations to canonical form)
PREDICATE_ALIASES: Dict[str, str] = {
    # CamelCase to snake_case
    "ContractValid": "contract_valid",
    "HasOffer": "has_offer",
    "HasAcceptance": "has_acceptance",
    "HasConsideration": "has_consideration",
    "HasMutualAssent": "has_mutual_assent",
    "LegalCapacity": "legal_capacity",
    "HasWriting": "has_writing",
    "RequiresWriting": "requires_writing",
    "SatisfiesSof": "satisfies_sof",
    "SatisfiesStatuteOfFrauds": "satisfies_sof",
    "LandSale": "land_sale",
    "LandSaleContract": "land_sale_contract",
    "Suretyship": "suretyship",
    "SuretyshipAgreement": "suretyship_agreement",
    "GoodsOver500": "goods_over_500",
    "CannotPerformWithinYear": "cannot_perform_within_year",
    "PartPerformance": "part_performance",
    "PromissoryEstoppel": "promissory_estoppel",
    "MerchantConfirmation": "merchant_confirmation",
    "SpeciallyManufactured": "specially_manufactured",
    "CourtAdmission": "court_admission",
    "ElectronicSignature": "electronic_signature",
    "ValidSignature": "valid_signature",
    "EsignApplies": "esign_applies",
    "Enforceable": "enforceable",
    "Unenforceable": "unenforceable",
    "Valid": "valid",
    "Invalid": "invalid",
    "Void": "void",
    "Voidable": "voidable",
    "UccApplies": "ucc_applies",
    # Common variations
    "satisfy_sof": "satisfies_sof",
    "statute_of_frauds": "satisfies_sof",
    "writing_required": "requires_writing",
    "must_be_in_writing": "requires_writing",
    "is_valid": "valid",
    "is_enforceable": "enforceable",
    "contract_enforceable": "enforceable",
    "contract_valid_if": "contract_valid",
}

# Concept to predicate mapping for legal terms
LEGAL_CONCEPT_TO_PREDICATE: Dict[str, str] = {
    # Contract elements
    "offer": "has_offer",
    "acceptance": "has_acceptance",
    "consideration": "has_consideration",
    "mutual assent": "has_mutual_assent",
    "meeting of minds": "has_mutual_assent",
    "capacity": "legal_capacity",
    "written": "has_writing",
    "writing": "has_writing",
    "in writing": "has_writing",
    "signed": "signed",
    "signature": "signed",
    # Contract types
    "land sale": "land_sale",
    "real estate": "land_sale",
    "real property": "land_sale",
    "sale of land": "land_sale",
    "suretyship": "suretyship",
    "guaranty": "suretyship",
    "guarantee": "suretyship",
    "goods over $500": "goods_over_500",
    "goods over 500": "goods_over_500",
    "ucc": "ucc_applies",
    # Statute of frauds
    "statute of frauds": "satisfies_sof",
    "sof": "satisfies_sof",
    # Exceptions
    "part performance": "part_performance",
    "partial performance": "part_performance",
    "promissory estoppel": "promissory_estoppel",
    "estoppel": "promissory_estoppel",
    "specially manufactured": "specially_manufactured",
    "custom goods": "specially_manufactured",
    "merchant confirmation": "merchant_confirmation",
    "confirmatory memo": "merchant_confirmation",
    "court admission": "court_admission",
    "admitted in court": "court_admission",
    # Electronic
    "electronic signature": "electronic_signature",
    "e-signature": "electronic_signature",
    "esign": "esign_applies",
    "ueta": "ueta_applies",
}


def normalize_predicate(predicate: str) -> str:
    """
    Normalize a predicate name to canonical form.

    Handles:
    - CamelCase to snake_case conversion
    - Known alias mapping
    - Whitespace trimming

    Args:
        predicate: Raw predicate name from LLM output

    Returns:
        Normalized predicate name in snake_case

    Examples:
        >>> normalize_predicate("ContractValid")
        'contract_valid'
        >>> normalize_predicate("HasOffer")
        'has_offer'
        >>> normalize_predicate("statute_of_frauds")
        'satisfies_sof'
    """
    predicate = predicate.strip()

    # Check alias mapping first (handles both CamelCase and variations)
    if predicate in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[predicate]

    # Convert CamelCase to snake_case
    # Insert underscore before uppercase letters and convert to lowercase
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", predicate).lower()

    # Check alias after conversion
    if snake_case in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[snake_case]

    return snake_case


def extract_predicates_from_asp(asp_code: str) -> List[str]:
    """
    Extract all predicate names from ASP code.

    Args:
        asp_code: ASP code string (rules, facts, or queries)

    Returns:
        List of predicate names found

    Examples:
        >>> extract_predicates_from_asp("contract_valid(X) :- has_offer(X).")
        ['contract_valid', 'has_offer']
    """
    # Match predicate names: word followed by (
    pattern = r"(\w+)\s*\("
    matches = re.findall(pattern, asp_code)

    # Filter out ASP keywords and duplicates
    keywords = {"not", "if", "then", "else", "or", "and"}
    predicates = []
    seen: Set[str] = set()

    for pred in matches:
        if pred not in keywords and pred not in seen:
            predicates.append(pred)
            seen.add(pred)

    return predicates


def validate_predicates(
    predicates: List[str],
    canonical_vocabulary: FrozenSet[str] = CANONICAL_PREDICATES,
) -> tuple:
    """
    Validate predicates against canonical vocabulary.

    Args:
        predicates: List of predicate names to validate
        canonical_vocabulary: Set of allowed predicates

    Returns:
        Tuple of (valid_predicates, invalid_predicates)

    Examples:
        >>> valid, invalid = validate_predicates(["has_offer", "UnknownPred"])
        >>> print(valid)
        ['has_offer']
        >>> print(invalid)
        ['UnknownPred']
    """
    valid = []
    invalid = []

    for pred in predicates:
        normalized = normalize_predicate(pred)
        if normalized in canonical_vocabulary:
            valid.append(normalized)
        else:
            invalid.append(pred)

    return valid, invalid


def find_closest_predicate(
    predicate: str,
    vocabulary: FrozenSet[str] = CANONICAL_PREDICATES,
) -> Optional[str]:
    """
    Find the closest matching canonical predicate for an invalid one.

    Uses simple substring and word matching heuristics.

    Args:
        predicate: Invalid predicate name
        vocabulary: Set of canonical predicates

    Returns:
        Closest matching canonical predicate, or None if no good match

    Examples:
        >>> find_closest_predicate("ContractIsValid")
        'contract_valid'
        >>> find_closest_predicate("RequireWriting")
        'requires_writing'
    """
    normalized = normalize_predicate(predicate).lower()

    # Direct match after normalization
    if normalized in vocabulary:
        return normalized

    # Check concept mapping
    for concept, canonical in LEGAL_CONCEPT_TO_PREDICATE.items():
        if concept in normalized or normalized in concept:
            return canonical

    # Try substring matching
    candidates = []
    for canonical in vocabulary:
        # Score based on common substrings
        score = 0
        pred_words = set(normalized.replace("_", " ").split())
        canon_words = set(canonical.replace("_", " ").split())

        # Count common words
        common = pred_words & canon_words
        score = len(common)

        # Bonus for prefix match
        if canonical.startswith(normalized[:4]) or normalized.startswith(canonical[:4]):
            score += 2

        # Bonus for key terms
        key_terms = ["valid", "offer", "accept", "consider", "write", "sign", "enforce"]
        for term in key_terms:
            if term in normalized and term in canonical:
                score += 3

        if score > 0:
            candidates.append((canonical, score))

    if candidates:
        # Return highest scoring match
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None
