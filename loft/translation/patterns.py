"""
Pattern matching for NL → ASP translation.

This module provides regex-based pattern matching to extract ASP facts
from natural language text.
"""

import re
from typing import List, Callable, Dict, Match


def normalize_identifier(text: str) -> str:
    """Normalize text to valid ASP identifier."""
    # Convert to lowercase and replace spaces/special chars with underscores
    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9_]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)  # Collapse multiple underscores
    normalized = normalized.strip("_")
    return normalized


def extract_dollar_amount(text: str) -> int:
    """Extract dollar amount from text."""
    match = re.search(r"\$?([\d,]+(?:\.\d{2})?)", text)
    if match:
        amount_str = match.group(1).replace(",", "")
        return int(float(amount_str))
    return 0


# Pattern matching functions
def is_a_pattern(match: Match) -> str:
    """Convert 'X is a Y' to Y(X)."""
    x = normalize_identifier(match.group(1))
    y = normalize_identifier(match.group(2))

    # Skip common sentence starters that aren't entities
    if x.lower() in ("this", "that", "it", "there", "here"):
        return ""

    return f"{y}({x})."


def has_pattern(match: Match) -> str:
    """Convert 'X has Y' to has_Y(X)."""
    x = normalize_identifier(match.group(1))
    y = normalize_identifier(match.group(2))
    return f"has_{y}({x})."


def signed_by_pattern(match: Match) -> str:
    """Convert 'X is/was signed by Y' to signed_by(X, Y)."""
    x = normalize_identifier(match.group(1))
    y = normalize_identifier(match.group(2))
    return f"signed_by({x}, {y})."


def includes_pattern(match: Match) -> str:
    """Convert 'X includes/contains Y' to contains(X, Y)."""
    x = normalize_identifier(match.group(1))
    y = normalize_identifier(match.group(2))
    return f"contains({x}, {y})."


def between_pattern(match: Match) -> str:
    """Convert 'contract between X and Y' to party relationships."""
    contract = normalize_identifier(match.group(1) or "contract_1")
    party1 = normalize_identifier(match.group(2))
    party2 = normalize_identifier(match.group(3))

    facts = []
    facts.append(f"contract({contract}).")
    facts.append(f"party({party1}).")
    facts.append(f"party({party2}).")
    facts.append(f"party_to_contract({contract}, {party1}).")
    facts.append(f"party_to_contract({contract}, {party2}).")

    return "\n".join(facts)


def sale_amount_pattern(match: Match) -> str:
    """Convert 'for $X' or 'sale price of $X' to sale_amount."""
    amount = extract_dollar_amount(match.group(1))
    return f"sale_amount(contract_1, {amount})."


# Pattern registry: maps regex patterns to converter functions
# Order matters! More specific patterns should come first
NL_TO_ASP_PATTERNS: Dict[str, Callable[[Match], str]] = {
    # "contract between X and Y" → party relationships (must come before "is a")
    r"(?:(\w+)\s+)?(?:contract|agreement)\s+between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)": between_pattern,
    # "X is/was signed by Y" → signed_by(X, Y). (must come before "is a")
    r"(\w+(?:\s+\w+)*)\s+(?:is|was)\s+signed\s+by\s+(\w+(?:\s+\w+)*)": signed_by_pattern,
    # "X is a Y" → Y(X). (use non-greedy and limit to avoid matching too much)
    r"(\w+)\s+is\s+an?\s+(\w+(?:\s+\w+){0,3}?)(?:\s|$|\.|\,)": is_a_pattern,
    # "X has Y" → has_Y(X). (limit to avoid matching too much)
    r"(\w+)\s+has\s+(?:a\s+)?(\w+(?:\s+\w+){0,2}?)(?:\s|$|\.|\,)": has_pattern,
    # "X includes/contains Y" → contains(X, Y).
    r"(\w+(?:\s+\w+)*)\s+(?:includes|contains)\s+(\w+(?:\s+\w+)*)": includes_pattern,
    # "for $X" or "sale price of $X" → sale_amount (use word boundary to avoid partial matches)
    r"\b(?:for|sale\s+price\s+of)\s+\$?([\d,]+(?:\.\d{2})?)": sale_amount_pattern,
}


def pattern_based_extraction(nl_text: str) -> List[str]:
    """
    Use regex patterns to extract ASP facts from natural language.

    Args:
        nl_text: Natural language text to parse

    Returns:
        List of ASP facts extracted using pattern matching
    """
    asp_facts = []

    for pattern, converter in NL_TO_ASP_PATTERNS.items():
        for match in re.finditer(pattern, nl_text, re.IGNORECASE):
            try:
                asp_fact = converter(match)
                # Skip empty results (e.g., filtered out by converter)
                if not asp_fact:
                    continue
                # Handle multi-line facts (e.g., from between_pattern)
                if "\n" in asp_fact:
                    asp_facts.extend(asp_fact.split("\n"))
                else:
                    asp_facts.append(asp_fact)
            except Exception:
                # Skip patterns that fail to convert
                continue

    # Deduplicate facts while preserving order
    seen = set()
    unique_facts = []
    for fact in asp_facts:
        if fact not in seen:
            seen.add(fact)
            unique_facts.append(fact)

    return unique_facts


def extract_contract_type(nl_text: str) -> str:
    """
    Extract contract type from natural language.

    Returns contract type like 'land_sale', 'goods_sale', 'service', etc.
    """
    nl_lower = nl_text.lower()

    # Check for specific contract types
    if "land sale" in nl_lower or "real estate" in nl_lower:
        return "land_sale"
    elif "goods sale" in nl_lower or "sale of goods" in nl_lower:
        return "goods_sale"
    elif "service" in nl_lower and "contract" in nl_lower:
        return "service"
    elif "employment" in nl_lower:
        return "employment"
    elif "lease" in nl_lower or "rental" in nl_lower:
        return "lease"

    return "general"


def extract_essential_elements(nl_text: str) -> Dict[str, bool]:
    """
    Extract whether essential contract elements are mentioned.

    Returns:
        Dictionary with boolean flags for essential elements
    """
    nl_lower = nl_text.lower()

    return {
        "has_consideration": "consideration" in nl_lower,
        "has_mutual_assent": (
            "mutual assent" in nl_lower
            or "meeting of the minds" in nl_lower
            or "agreement" in nl_lower
        ),
        "has_writing": ("writing" in nl_lower or "written" in nl_lower or "document" in nl_lower),
        "is_signed": ("signed" in nl_lower or "signature" in nl_lower),
    }


def extract_parties(nl_text: str) -> List[str]:
    """
    Extract party names from natural language.

    Uses simple heuristics to identify proper nouns that might be parties.
    """
    # Look for "between X and Y" pattern
    between_match = re.search(
        r"between\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        nl_text,
    )
    if between_match:
        return [between_match.group(1), between_match.group(2)]

    # Look for capitalized names
    names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", nl_text)

    # Filter out common words that aren't names
    common_words = {"The", "A", "An", "This", "That", "Contract", "Agreement", "Writing"}
    parties = [name for name in names if name not in common_words]

    # Deduplicate
    return list(dict.fromkeys(parties))


def quick_extract_facts(nl_text: str) -> List[str]:
    """
    Quick extraction using pattern matching only.

    This is a fast, lightweight alternative to full LLM-based extraction.
    Useful for simple cases or when LLM is not available.

    Args:
        nl_text: Natural language text

    Returns:
        List of ASP facts
    """
    facts = []

    # Extract using patterns
    facts.extend(pattern_based_extraction(nl_text))

    # Extract contract type
    contract_type = extract_contract_type(nl_text)
    if contract_type != "general" and "contract(" not in str(facts):
        facts.append(f"{contract_type}_contract(contract_1).")

    # Extract essential elements
    elements = extract_essential_elements(nl_text)
    for element, present in elements.items():
        if present and not any(element in f for f in facts):
            facts.append(f"{element}(contract_1).")

    return facts
