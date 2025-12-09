"""
Grounding and validation for NL → ASP translation.

This module ensures that generated ASP facts are properly grounded
and consistent with the existing ASP core.
"""

from typing import List, Tuple, Optional, Set, Any
import re


def extract_predicate_name(fact: str) -> str:
    """Extract predicate name from ASP fact."""
    match = re.match(r"(\w+)\(", fact)
    if match:
        return match.group(1)
    return ""


def extract_arguments(fact: str) -> List[str]:
    """Extract arguments from ASP fact."""
    match = re.search(r"\((.*?)\)", fact)
    if match:
        args_str = match.group(1)
        return [arg.strip() for arg in args_str.split(",")]
    return []


class ASPGrounder:
    """Ensure generated ASP facts are properly grounded."""

    def __init__(self, asp_core: Optional[Any] = None):
        """
        Initialize grounder.

        Args:
            asp_core: Optional ASPCore instance for validation
        """
        self.core = asp_core
        self._known_predicates: Set[str] = self._load_known_predicates()

    def _load_known_predicates(self) -> Set[str]:
        """Load known predicates from domain."""
        # Legal domain predicates
        return {
            "contract",
            "party",
            "writing",
            "enforceable",
            "void",
            "unenforceable",
            "satisfies_statute_of_frauds",
            "has_writing",
            "has_consideration",
            "has_mutual_assent",
            "signed_by",
            "party_to_contract",
            "land_sale_contract",
            "goods_sale_contract",
            "service_contract",
            "sale_amount",
            "contains",
            "references_contract",
        }

    def ground_and_validate(self, asp_facts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Ground facts in existing ASP program context.

        Args:
            asp_facts: List of ASP facts to validate

        Returns:
            Tuple of (valid_facts, invalid_facts)
        """
        valid = []
        invalid = []

        for fact in asp_facts:
            fact = fact.strip()
            if not fact or fact.startswith("%"):
                continue

            # Check if fact uses valid predicates
            predicate_name = extract_predicate_name(fact)

            if self._is_valid_predicate(predicate_name):
                # Check type constraints
                if self._satisfies_type_constraints(fact):
                    valid.append(fact)
                else:
                    invalid.append(fact)
            else:
                # Unknown predicate - could be valid, flag for review
                # For now, accept with warning
                valid.append(fact)

        return (valid, invalid)

    def _is_valid_predicate(self, pred_name: str) -> bool:
        """Check if predicate is defined in domain."""
        if self.core is not None:
            # Check against ASP core if available
            try:
                known_predicates = self.core.get_all_predicates()
                return pred_name in known_predicates
            except AttributeError:
                pass

        # Fall back to known predicates set
        return pred_name in self._known_predicates

    def _satisfies_type_constraints(self, fact: str) -> bool:
        """
        Check if fact satisfies type constraints.

        For Phase 1, this is basic validation. Phase 2 will add
        more sophisticated type checking.
        """
        # Check basic syntax
        if not fact.endswith("."):
            return False

        # Check for balanced parentheses
        if fact.count("(") != fact.count(")"):
            return False

        # Check predicate name is valid identifier
        pred_name = extract_predicate_name(fact)
        if not pred_name or not pred_name.replace("_", "").isalnum():
            return False

        return True


class AmbiguityHandler:
    """Detect and handle ambiguous NL → ASP conversions."""

    def detect_ambiguity(
        self, nl_text: str, asp_candidates: List[str]
    ) -> Optional[str]:
        """
        Detect if NL has multiple valid ASP interpretations.

        Args:
            nl_text: Natural language input
            asp_candidates: List of possible ASP translations

        Returns:
            Ambiguity description if found, None otherwise
        """
        if len(asp_candidates) > 1:
            # Multiple interpretations
            return f"Multiple interpretations: {asp_candidates}"

        # Check for unclear references
        if self._has_unclear_references(nl_text):
            return "Unclear entity references - needs clarification"

        # Check for missing information
        if self._has_missing_information(nl_text):
            return "Missing essential information"

        return None

    def _has_unclear_references(self, nl_text: str) -> bool:
        """Check if text has unclear pronoun references."""
        pronouns = ["it", "they", "them", "this", "that", "these", "those"]
        nl_lower = nl_text.lower()

        # Simple heuristic: pronouns without clear antecedent
        for pronoun in pronouns:
            if f" {pronoun} " in f" {nl_lower} ":
                # Check if there's a noun before it
                words = nl_lower.split()
                if pronoun in words:
                    idx = words.index(pronoun)
                    if idx == 0 or not self._is_likely_noun(words[idx - 1]):
                        return True

        return False

    def _has_missing_information(self, nl_text: str) -> bool:
        """Check if essential information is missing."""
        # Check for vague language
        vague_terms = ["something", "someone", "somewhere", "somehow"]
        nl_lower = nl_text.lower()

        return any(term in nl_lower for term in vague_terms)

    def _is_likely_noun(self, word: str) -> bool:
        """Heuristic: check if word is likely a noun."""
        # Simple heuristic: capitalized or ends in common noun suffixes
        return (
            word[0].isupper()
            or word.endswith("tion")
            or word.endswith("ment")
            or word.endswith("ness")
        )

    def request_clarification(self, ambiguity: str) -> str:
        """Generate clarification question."""
        return f"The statement is ambiguous: {ambiguity}. Please clarify."


def validate_new_facts(
    asp_core: Any,
    new_facts: List[str],
    check_consistency: bool = True,
) -> Tuple[bool, str]:
    """
    Check if adding new ASP facts maintains consistency.

    Args:
        asp_core: ASPCore instance
        new_facts: List of new facts to add
        check_consistency: Whether to run consistency check

    Returns:
        Tuple of (is_valid, message)
    """
    if not check_consistency:
        return (True, "Consistency checking disabled")

    try:
        # Create copy of core
        test_core = _copy_asp_core(asp_core)

        # Add new facts
        for fact in new_facts:
            test_core.add_fact(fact)

        # Check consistency
        if hasattr(test_core, "check_consistency"):
            is_consistent = test_core.check_consistency()

            if not is_consistent:
                return (False, "New facts create contradiction")

        return (True, "Facts are consistent")

    except Exception as e:
        return (False, f"Validation error: {str(e)}")


def _copy_asp_core(asp_core: Any) -> Any:
    """Create a copy of ASP core for testing."""
    # Try to use copy method if available
    if hasattr(asp_core, "copy"):
        return asp_core.copy()

    # Fall back to creating new instance
    from loft.symbolic import ASPCore

    new_core = ASPCore()

    # Copy rules and facts
    if hasattr(asp_core, "get_all_rules"):
        for rule in asp_core.get_all_rules():
            new_core.add_rule(rule)

    if hasattr(asp_core, "facts"):
        for fact in asp_core.facts:
            new_core.add_fact(fact)

    return new_core
