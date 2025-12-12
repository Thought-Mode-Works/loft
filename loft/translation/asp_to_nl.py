"""
ASP to Natural Language translation module.

Converts ASP queries, rules, and facts into natural language for LLM processing.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from loguru import logger

if TYPE_CHECKING:
    from loft.symbolic.asp_core import ASPCore


# Legal domain templates for common predicates
LEGAL_PREDICATE_TEMPLATES = {
    "contract": "{arg} is a contract",
    "enforceable": "{arg} is enforceable",
    "unenforceable": "{arg} is unenforceable",
    "valid": "{arg} is valid",
    "void": "{arg} is void",
    "voidable": "{arg} is voidable",
    "satisfies_statute_of_frauds": "{arg} satisfies the statute of frauds requirements",
    "has_writing": "{arg1} has a writing {arg2}",
    "signed_by": "{arg1} is signed by {arg2}",
    "land_sale_contract": "{arg} is a land sale contract",
    "consideration": "{arg} has consideration",
    "mutual_assent": "{arg} has mutual assent",
    "legal_capacity": "{arg1} has legal capacity {arg2}",
    "offer": "{arg} is an offer",
    "acceptance": "{arg} is an acceptance",
    "price": "{arg1} has price {arg2}",
    "party": "{arg} is a party",
    "involved_in": "{arg1} is involved in {arg2}",
}

# Extended statement templates for semantic-preserving round-trip translation
# Maps predicate patterns to declarative statement forms
STATEMENT_TEMPLATES = {
    # Contract formation
    "contract_valid": "A contract is valid when it has all required elements",
    "contract_requires": "A contract requires {elements}",
    "requires_offer": "A contract requires an offer",
    "requires_acceptance": "A contract requires acceptance",
    "requires_consideration": "A contract requires consideration",
    "has_offer": "{arg} has a valid offer",
    "has_acceptance": "{arg} has valid acceptance",
    "has_consideration": "{arg} has valid consideration",
    # Statute of frauds
    "requires_writing": "{arg} must be in writing",
    "requires_written_form": "{arg} must be in writing",
    "writing_required": "A written document is required for {arg}",
    "statute_of_frauds_applies": "The statute of frauds applies to {arg}",
    "land_sale": "{arg} is a contract for the sale of land",
    "goods_over_500": "{arg} is a contract for goods over $500",
    "suretyship": "{arg} is a suretyship agreement",
    "suretyship_agreement": "{arg} is a suretyship agreement",
    "goods_contract": "{arg} is a contract for sale of goods",
    "within_one_year": "{arg} cannot be performed within one year",
    "cannot_perform_within_year": "{arg} cannot be performed within one year",
    # Exceptions and satisfactions
    "part_performance": "Part performance can satisfy the statute of frauds",
    "satisfies_sof": "{arg} satisfies the statute of frauds",
    "exception_applies": "An exception applies to {arg}",
    "promissory_estoppel": "Promissory estoppel may overcome the statute of frauds",
    # Electronic signatures
    "electronic_signature_valid": "Electronic signatures are valid",
    "esign_applies": "The ESIGN Act applies to {arg}",
    "electronic_signature": "{arg} has a valid electronic signature",
    # UCC specific
    "ucc_applies": "The UCC applies to {arg}",
    "merchant": "{arg} is a merchant",
    "merchant_confirmation": "A merchant confirmation creates an enforceable contract",
    "specially_manufactured": "{arg} involves specially manufactured goods",
    "specially_manufactured_goods": "Specially manufactured goods are exempt from the writing requirement",
    # Admission
    "admission_in_court": "An admission in court satisfies the statute of frauds",
    "court_admission": "{arg} was admitted in court",
    # Written memorandum
    "written_memorandum": "{arg} has a written memorandum",
    "requires_memorandum": "{arg} requires a written memorandum",
}

# Comprehensive rule templates that map ASP rule patterns to full statements
RULE_STATEMENT_TEMPLATES = {
    # Contract requirements rules
    "contract_valid(X) :- has_offer(X), has_acceptance(X), has_consideration(X)": (
        "A contract is valid if it has offer, acceptance, and consideration"
    ),
    "enforceable(X) :- contract(X), satisfies_sof(X)": (
        "A contract is enforceable if it satisfies the statute of frauds"
    ),
    "requires_writing(X) :- land_sale(X)": ("Land sale contracts must be in writing"),
    "requires_writing(X) :- land_sale_contract(X)": ("Land sale contracts must be in writing"),
    "requires_writing(X) :- goods_over_500(X)": (
        "Contracts for goods over $500 require a written memorandum"
    ),
    "requires_writing(X) :- suretyship(X)": (
        "Suretyship agreements must be in writing"
    ),
    "requires_writing(X) :- suretyship_agreement(X)": (
        "Suretyship agreements must be in writing"
    ),
    "requires_writing(X) :- goods_contract(X)": (
        "Contracts for sale of goods require a written memorandum"
    ),
    "satisfies_sof(X) :- part_performance(X)": (
        "Part performance can satisfy the statute of frauds"
    ),
    "valid_signature(X) :- electronic_signature(X), esign_applies(X)": (
        "Electronic signatures are valid under the ESIGN Act"
    ),
    "exception(X) :- specially_manufactured(X)": (
        "Specially manufactured goods are exempt from the writing requirement"
    ),
    "enforceable(X) :- merchant_confirmation(X)": (
        "Merchant confirmation creates enforceable contract"
    ),
    "requires_writing(X) :- cannot_perform_within_year(X)": (
        "Contracts that cannot be performed within one year require writing"
    ),
    "satisfies_sof(X) :- court_admission(X)": (
        "Admission in court satisfies the statute of frauds"
    ),
}

# Structural templates for preserving sentence structure in translations
# Maps rule types to declarative sentence patterns
STRUCTURAL_TEMPLATES = {
    # Requirement patterns - "X requires Y"
    "requirement": "{subject} requires {object}",
    "requirement_plural": "{subject} require {object}",
    # Condition patterns - "X is valid if Y"
    "condition": "{subject} is {state} if {conditions}",
    "condition_when": "{subject} is {state} when {conditions}",
    # Obligation patterns - "X must Y"
    "obligation": "{subject} must {action}",
    "obligation_plural": "{subject} must {action}",
    # Prohibition patterns - "X must not Y"
    "prohibition": "{subject} must not {action}",
    "prohibition_cannot": "{subject} cannot {action}",
    # Exception patterns - "X is exempt when Y"
    "exception": "{subject} is exempt when {conditions}",
    "exception_unless": "{subject} applies unless {conditions}",
    "exception_except": "{subject} except when {conditions}",
    # Satisfaction patterns - "X satisfies Y"
    "satisfaction": "{subject} satisfies {requirement}",
    "satisfaction_can": "{subject} can satisfy {requirement}",
    # Disjunction patterns - "X or Y"
    "disjunction": "{subject} may {option1} or {option2}",
    "disjunction_either": "Either {option1} or {option2}",
    # Conjunction patterns - "X and Y"
    "conjunction": "{subject} must have {element1} and {element2}",
    "conjunction_both": "{subject} requires both {element1} and {element2}",
    # Validity patterns - "X is valid/enforceable"
    "validity": "{subject} is {validity_state}",
    "validity_if": "{subject} is {validity_state} if {conditions}",
    # Default patterns - "X by default unless Y"
    "default": "{subject} by default unless {exception}",
    "default_presumed": "{subject} is presumed unless {exception}",
    # Equivalence patterns - "X means Y"
    "equivalence": "{term1} means {term2}",
    "equivalence_is": "{term1} is {term2}",
}

# Rule type indicators for classification
RULE_TYPE_INDICATORS = {
    "requirement": [
        "requires",
        "require",
        "must have",
        "needs",
        "necessary",
        "essential",
        "required",
    ],
    "obligation": [
        "must be",
        "shall be",
        "has to be",
        "need to be",
        "obligated",
        "mandatory",
    ],
    "prohibition": [
        "must not",
        "cannot",
        "shall not",
        "may not",
        "prohibited",
        "forbidden",
        "not allowed",
    ],
    "exception": [
        "unless",
        "except",
        "exempt",
        "exemption",
        "notwithstanding",
        "provided that",
    ],
    "condition": ["if", "when", "where", "provided", "in case", "conditional upon"],
    "satisfaction": [
        "satisfies",
        "satisfy",
        "meets",
        "fulfills",
        "complies with",
        "sufficient",
    ],
    "validity": [
        "valid",
        "enforceable",
        "binding",
        "effective",
        "void",
        "voidable",
        "invalid",
    ],
    "default": ["by default", "presumed", "assumed", "unless proven"],
}


def detect_rule_type(asp_rule: str) -> str:
    """
    Detect the structural type of an ASP rule.

    Analyzes the rule structure to determine its logical pattern
    (requirement, condition, prohibition, etc.) for template selection.

    Args:
        asp_rule: ASP rule string

    Returns:
        Rule type identifier (e.g., "requirement", "condition", "prohibition")

    Examples:
        >>> detect_rule_type("requires_writing(X) :- land_sale(X).")
        'requirement'
        >>> detect_rule_type("enforceable(X) :- contract(X), signed(X).")
        'condition'
        >>> detect_rule_type("unenforceable(X) :- not has_writing(X).")
        'prohibition'
    """
    rule = asp_rule.strip().lower()

    # Check if it's a fact (no body)
    if ":-" not in rule:
        return "fact"

    head_part, body_part = rule.split(":-", 1)
    head = head_part.strip()
    body = body_part.strip()

    # Extract head predicate
    head_pred_match = re.match(r"(\w+)\s*\(", head)
    head_pred = head_pred_match.group(1) if head_pred_match else ""

    # Check for negation in body (prohibition or exception pattern)
    has_negation = "not " in body or "not(" in body

    # Check for disjunction in body
    has_disjunction = ";" in body

    # Check for multiple conditions (conjunction)
    condition_count = body.count(",") + 1

    # Check for disjunction first (highest priority structural pattern)
    if has_disjunction:
        return "disjunction"

    # Classify based on head predicate and structure
    if head_pred.startswith("requires_") or head_pred == "requires":
        return "requirement"

    if head_pred.startswith("unenforceable") or head_pred.startswith("invalid"):
        if has_negation:
            return "prohibition"
        return "validity"

    if head_pred.startswith("enforceable") or head_pred.startswith("valid"):
        # Prefer conjunction for multi-condition validity rules
        if condition_count >= 3:
            return "conjunction"
        return "validity" if condition_count == 1 else "condition"

    if head_pred.startswith("satisfies_") or head_pred == "satisfies":
        return "satisfaction"

    if head_pred.startswith("exception") or head_pred.startswith("exempt"):
        return "exception"

    if has_negation:
        # Check if it's a default rule (single negation condition)
        if condition_count == 1:
            # "default" or "assumed" patterns are default rules
            if (
                "default" in head_pred
                or "assumed" in head_pred
                or "presumed" in head_pred
            ):
                return "default"
            # Also default if the negation is the primary pattern
            return "default"
        return "exception"

    if condition_count >= 2:
        return "conjunction"

    # Default to condition for rules with body
    return "condition"


def apply_structural_template(
    rule_type: str,
    head_predicate: str,
    head_args: List[str],
    body_predicates: List[tuple],
    original_asp: str,
) -> str:
    """
    Apply a structural template to generate a well-formed sentence.

    Uses the detected rule type to select an appropriate sentence structure
    that preserves the declarative form and improves round-trip fidelity.

    Args:
        rule_type: Type of rule (from detect_rule_type)
        head_predicate: The head predicate name
        head_args: Arguments of the head predicate
        body_predicates: List of (predicate, args) tuples from body
        original_asp: Original ASP rule for fallback

    Returns:
        Natural language sentence with proper structure

    Examples:
        >>> apply_structural_template(
        ...     "requirement", "requires_writing", ["X"],
        ...     [("land_sale", ["X"])], "requires_writing(X) :- land_sale(X)."
        ... )
        'Land sale contracts must be in writing'
    """
    # Humanize head predicate and arguments
    head_human = _humanize_predicate_for_structure(head_predicate)
    subject = _get_subject_from_args(head_args)

    # Build condition string from body predicates
    conditions = _build_conditions_string(body_predicates)

    # Select template based on rule type
    if rule_type == "requirement":
        if "writing" in head_predicate or "written" in head_predicate:
            return f"{_get_contract_subject(body_predicates)} must be in writing"
        if conditions:
            return f"{subject} requires {conditions}"
        return f"{subject} {head_human}"

    elif rule_type == "obligation":
        return f"{subject} must be {head_human}"

    elif rule_type == "prohibition":
        if conditions:
            return f"{subject} is not {head_human} if {conditions}"
        return f"{subject} must not be {head_human}"

    elif rule_type == "exception":
        return f"{subject} is exempt when {conditions}"

    elif rule_type == "satisfaction":
        requirement = _extract_requirement(head_predicate)
        if conditions:
            return f"{conditions} can satisfy {requirement}"
        return f"{subject} satisfies {requirement}"

    elif rule_type == "validity":
        validity_state = _extract_validity_state(head_predicate)
        if conditions:
            return f"{subject} is {validity_state} if {conditions}"
        return f"{subject} is {validity_state}"

    elif rule_type == "condition":
        state = _extract_state(head_predicate)
        if conditions:
            return f"{subject} is {state} if {conditions}"
        return f"{subject} is {state}"

    elif rule_type == "conjunction":
        if len(body_predicates) >= 2:
            elements = [
                _humanize_predicate_for_structure(p[0]) for p in body_predicates
            ]
            if len(elements) == 2:
                return f"{subject} requires {elements[0]} and {elements[1]}"
            else:
                return (
                    f"{subject} requires {', '.join(elements[:-1])}, and {elements[-1]}"
                )
        return f"{subject} is {head_human} if {conditions}"

    elif rule_type == "disjunction":
        # Handle disjunction (;) in body
        return f"{subject} may satisfy {head_human} through alternative means"

    elif rule_type == "default":
        return f"{subject} is {head_human} by default unless otherwise proven"

    elif rule_type == "fact":
        return f"{subject} {head_human}"

    # Fallback to basic translation
    return f"{subject} is {head_human}" + (f" if {conditions}" if conditions else "")


def _humanize_predicate_for_structure(predicate: str) -> str:
    """Convert predicate name to human-readable form for structural templates."""
    # Remove common prefixes
    for prefix in ["requires_", "has_", "is_", "can_", "must_"]:
        if predicate.startswith(prefix):
            predicate = predicate[len(prefix) :]
            break

    # Convert snake_case to spaces
    result = predicate.replace("_", " ")

    # Handle common legal terms
    replacements = {
        "sof": "the statute of frauds",
        "writing": "a written document",
        "ucc": "the UCC",
        "esign": "the ESIGN Act",
    }

    for term, replacement in replacements.items():
        if result == term:
            return replacement

    return result


def _get_subject_from_args(args: List[str]) -> str:
    """Extract subject from predicate arguments."""
    if not args:
        return "the contract"

    first_arg = args[0]

    # Handle single uppercase variable
    if len(first_arg) == 1 and first_arg.isupper():
        var_mapping = {
            "X": "a contract",
            "Y": "an entity",
            "C": "a contract",
            "W": "a writing",
            "P": "a party",
            "S": "a signature",
            "A": "an agreement",
        }
        return var_mapping.get(first_arg, "the entity")

    # Handle entity IDs
    if first_arg.startswith("contract"):
        return "the contract"
    if first_arg.startswith("writing"):
        return "the writing"

    return f"the {first_arg.replace('_', ' ')}"


def _get_contract_subject(body_predicates: List[tuple]) -> str:
    """Extract a descriptive contract subject from body predicates."""
    for pred, args in body_predicates:
        if pred == "land_sale":
            return "Land sale contracts"
        if pred == "goods_over_500":
            return "Contracts for goods over $500"
        if pred == "suretyship":
            return "Suretyship agreements"
        if pred == "cannot_perform_within_year":
            return "Contracts that cannot be performed within one year"
        if pred == "marriage_promise":
            return "Promises made in consideration of marriage"

    return "The contract"


def _build_conditions_string(body_predicates: List[tuple]) -> str:
    """Build a natural language conditions string from body predicates."""
    if not body_predicates:
        return ""

    condition_parts = []
    for pred, args in body_predicates:
        # Handle negation
        if pred.startswith("not_"):
            pred = pred[4:]
            condition_parts.append(
                f"it does not have {_humanize_predicate_for_structure(pred)}"
            )
        else:
            humanized = _humanize_predicate_for_structure(pred)
            if args and len(args[0]) == 1 and args[0].isupper():
                condition_parts.append(f"it has {humanized}")
            else:
                condition_parts.append(humanized)

    if len(condition_parts) == 0:
        return ""
    elif len(condition_parts) == 1:
        return condition_parts[0]
    elif len(condition_parts) == 2:
        return f"{condition_parts[0]} and {condition_parts[1]}"
    else:
        return ", ".join(condition_parts[:-1]) + f", and {condition_parts[-1]}"


def _extract_requirement(predicate: str) -> str:
    """Extract requirement name from predicate."""
    if "sof" in predicate or "statute" in predicate:
        return "the statute of frauds"
    if "writing" in predicate:
        return "the writing requirement"
    return predicate.replace("satisfies_", "").replace("_", " ")


def _extract_validity_state(predicate: str) -> str:
    """Extract validity state from predicate."""
    if "unenforceable" in predicate:
        return "unenforceable"
    if "enforceable" in predicate:
        return "enforceable"
    if "invalid" in predicate:
        return "invalid"
    if "valid" in predicate:
        return "valid"
    if "void" in predicate:
        return "void"
    return predicate.replace("_", " ")


def _extract_state(predicate: str) -> str:
    """Extract state description from predicate."""
    # Remove common prefixes and convert
    for prefix in ["is_", "has_", "can_"]:
        if predicate.startswith(prefix):
            return predicate[len(prefix) :].replace("_", " ")
    return predicate.replace("_", " ")


# ASP pattern templates for rule translation
ASP_RULE_PATTERNS = [
    # Simple rule: head :- body
    (
        r"^(\w+)\(([^)]+)\)\s*:-\s*(\w+)\(([^)]+)\)\.$",
        "{head} for {head_args} if {body} for {body_args}",
    ),
    # Rule with negation: head :- not body
    (
        r"^(\w+)\(([^)]+)\)\s*:-\s*not\s+(\w+)\(([^)]+)\)\.$",
        "{head} for {head_args} by default (unless {body} is proven for {body_args})",
    ),
    # Rule with conjunction: head :- body1, body2
    (
        r"^(\w+)\(([^)]+)\)\s*:-\s*(\w+)\(([^)]+)\),\s*(\w+)\(([^)]+)\)\.$",
        "{head} for {head_args} if {body1} for {body1_args} and {body2} for {body2_args}",
    ),
    # Rule with three conditions: head :- body1, body2, body3
    (
        r"^(\w+)\(([^)]+)\)\s*:-\s*(\w+)\(([^)]+)\),\s*(\w+)\(([^)]+)\),\s*(\w+)\(([^)]+)\)\.$",
        "{head} for {head_args} if {body1} for {body1_args}, {body2} for {body2_args}, and {body3} for {body3_args}",
    ),
]


@dataclass
class TranslationResult:
    """Result of ASP to NL translation."""

    natural_language: str
    asp_source: str
    predicates_used: List[str]
    confidence: float = 1.0  # Translation confidence
    ambiguities: List[str] = field(default_factory=list)  # Detected ambiguities


def extract_predicates(asp_text: str) -> List[str]:
    """
    Extract predicate names from ASP text.

    Args:
        asp_text: ASP rule or query text

    Returns:
        List of predicate names found

    Example:
        >>> extract_predicates("contract(c1), enforceable(c1).")
        ['contract', 'enforceable']
    """
    # Match predicate names followed by ( or ending with .
    pattern = r"(\w+)(?:\(|\.)"
    matches = re.findall(pattern, asp_text)

    # Filter out ASP keywords
    keywords = {"not", "if", "then", "else", "or"}
    predicates = [p for p in matches if p not in keywords]

    # Remove duplicates while preserving order
    seen: set[str] = set()
    return [p for p in predicates if not (p in seen or seen.add(p))]  # type: ignore


def parse_predicate_call(text: str) -> tuple[str, List[str]]:
    """
    Parse a predicate call into name and arguments.

    Args:
        text: Predicate call like "contract(c1)" or "signed_by(w1, john)"

    Returns:
        Tuple of (predicate_name, arguments)

    Example:
        >>> parse_predicate_call("contract(c1)")
        ('contract', ['c1'])
        >>> parse_predicate_call("signed_by(w1, john)")
        ('signed_by', ['w1', 'john'])
    """
    match = re.match(r"(\w+)\(([^)]+)\)", text.strip())
    if not match:
        return (text.strip(), [])

    predicate = match.group(1)
    args_str = match.group(2)
    args = [arg.strip() for arg in args_str.split(",")]

    return (predicate, args)


def humanize_predicate_name(predicate: str) -> str:
    """
    Convert snake_case predicate name to human-readable form.

    Args:
        predicate: Predicate name like "satisfies_statute_of_frauds"

    Returns:
        Human-readable form like "satisfies the statute of frauds"

    Example:
        >>> humanize_predicate_name("satisfies_statute_of_frauds")
        'satisfies the statute of frauds'
    """
    # Replace underscores with spaces
    human = predicate.replace("_", " ")

    # Add articles where appropriate
    # This is a simple heuristic - could be improved with NLP
    if " of " in human and " the " not in human:
        human = human.replace(" of ", " of the ")

    return human


def humanize_variable(var: str) -> str:
    """
    Convert ASP variable to human-readable form.

    Args:
        var: ASP variable like "C" or "c1" or "_"

    Returns:
        Human-readable form

    Example:
        >>> humanize_variable("C")
        'some contract'
        >>> humanize_variable("c1")
        'contract c1'
        >>> humanize_variable("_")
        'any value'
    """
    # Anonymous variable
    if var == "_":
        return "any value"

    # Single uppercase letter (ASP variable)
    if len(var) == 1 and var.isupper():
        var_names = {
            "C": "some contract",
            "W": "some writing",
            "P": "some party",
            "X": "some entity",
            "Y": "some entity",
        }
        return var_names.get(var, f"some {var.lower()}")

    # Constant (lowercase or with numbers)
    # Try to extract type from prefix
    if var.startswith("c") and len(var) > 1 and var[1:].isdigit():
        return f"contract {var}"
    elif var.startswith("w") and len(var) > 1 and var[1:].isdigit():
        return f"writing {var}"
    elif var.startswith("p") and len(var) > 1 and var[1:].isdigit():
        return f"party {var}"
    else:
        # Just return as-is
        return var


def asp_to_nl(query: str, context: Optional[ASPCore] = None) -> str:
    """
    Convert ASP predicate query to natural language question.

    Args:
        query: ASP query like "satisfies_statute_of_frauds(contract_123)?"
        context: Optional ASP core for enriched context

    Returns:
        Natural language question

    Examples:
        >>> asp_to_nl("satisfies_statute_of_frauds(contract_123)?")
        'Does contract_123 satisfy the statute of frauds requirements?'

        >>> asp_to_nl("enforceable(C)?")
        'Which contracts are enforceable?'
    """
    # Remove trailing question mark or period
    query = query.rstrip("?.").strip()

    # Parse predicate and arguments
    predicate, args = parse_predicate_call(query)

    # Check if we have a template for this predicate
    if predicate in LEGAL_PREDICATE_TEMPLATES:
        template = LEGAL_PREDICATE_TEMPLATES[predicate]

        # Apply template
        if len(args) == 1:
            humanized_arg = humanize_variable(args[0])
            nl_text = template.format(arg=humanized_arg, arg1=humanized_arg)

            # Determine question type
            if len(args[0]) == 1 and args[0].isupper():
                # Variable query - asking "which" or "what"
                # Extract the predicate part and pluralize
                pred_part = nl_text.replace(humanized_arg, "").strip()
                # Handle "is a X" -> "are Xs" pattern
                if pred_part.startswith("is a "):
                    pred_type = pred_part[5:]  # Remove "is a "
                    question = f"Which entities are {pred_type}s?"
                elif pred_part.startswith("is "):
                    pred_type = pred_part[3:]  # Remove "is "
                    question = f"Which entities are {pred_type}?"
                else:
                    question = f"Which entities {pred_part}?"
            else:
                # Constant query - yes/no question with "Is" for better grammar
                # Convert "X is Y" to "Is X Y?"
                if " is a " in nl_text:
                    parts = nl_text.split(" is a ", 1)
                    question = f"Is {parts[0]} a {parts[1]}?"
                elif " is " in nl_text:
                    parts = nl_text.split(" is ", 1)
                    question = f"Is {parts[0]} {parts[1]}?"
                elif " has " in nl_text:
                    parts = nl_text.split(" has ", 1)
                    question = f"Does {parts[0]} have {parts[1]}?"
                elif " satisfies " in nl_text:
                    parts = nl_text.split(" satisfies ", 1)
                    question = f"Does {parts[0]} satisfy {parts[1]}?"
                else:
                    question = f"Does {nl_text}?"

        elif len(args) == 2:
            arg1_human = humanize_variable(args[0])
            arg2_human = humanize_variable(args[1])
            nl_text = template.format(arg=arg1_human, arg1=arg1_human, arg2=arg2_human)

            if any(len(arg) == 1 and arg.isupper() for arg in args):
                # At least one variable - open-ended question
                question = f"For which values does {nl_text}?"
            else:
                # All constants - yes/no question
                # Apply same grammar improvements as single-arg
                if " is " in nl_text:
                    parts = nl_text.split(" is ", 1)
                    question = f"Is {parts[0]} {parts[1]}?"
                elif " has a " in nl_text:
                    parts = nl_text.split(" has a ", 1)
                    question = f"Does {parts[0]} have a {parts[1]}?"
                elif " has " in nl_text:
                    parts = nl_text.split(" has ", 1)
                    question = f"Does {parts[0]} have {parts[1]}?"
                else:
                    question = f"Does {nl_text}?"
        else:
            # Fallback for multiple arguments
            args_human = ", ".join(humanize_variable(arg) for arg in args)
            question = f"Does {humanize_predicate_name(predicate)}({args_human})?"

    else:
        # No template - use generic translation
        args_human = ", ".join(humanize_variable(arg) for arg in args)
        predicate_human = humanize_predicate_name(predicate)

        if len(args) == 1 and len(args[0]) == 1 and args[0].isupper():
            question = f"Which entities satisfy {predicate_human}?"
        else:
            question = f"Does {predicate_human} hold for {args_human}?"

    return question


def asp_rule_to_nl(rule: str) -> str:
    """
    Convert ASP rule to readable natural language explanation.

    Args:
        rule: ASP rule text

    Returns:
        Natural language explanation

    Examples:
        >>> asp_rule_to_nl("satisfies_statute_of_frauds(C) :- has_writing(C, W), signed_by(W, _).")
        'A contract satisfies the statute of frauds if it has a writing that is signed.'

        >>> asp_rule_to_nl("enforceable(C) :- contract(C), not unenforceable(C).")
        'A contract is enforceable if it is not proven to be unenforceable (default reasoning).'
    """
    rule = rule.strip()

    # Handle facts (no rule body)
    if ":-" not in rule:
        # It's a fact
        fact = rule.rstrip(".")
        predicate, args = parse_predicate_call(fact)

        if predicate in LEGAL_PREDICATE_TEMPLATES:
            template = LEGAL_PREDICATE_TEMPLATES[predicate]
            args_human = [humanize_variable(arg) for arg in args]

            if len(args) == 1:
                return template.format(arg=args_human[0], arg1=args_human[0]) + "."
            elif len(args) == 2:
                return (
                    template.format(
                        arg=args_human[0], arg1=args_human[0], arg2=args_human[1]
                    )
                    + "."
                )
        else:
            predicate_human = humanize_predicate_name(predicate)
            args_str = ", ".join(humanize_variable(arg) for arg in args)
            return f"{predicate_human} holds for {args_str}."

    # Handle constraints (start with :-)
    if rule.startswith(":-"):
        body = rule[2:].strip().rstrip(".")
        return f"Constraint: {_translate_body(body)} is not allowed."

    # Split into head and body
    if ":-" in rule:
        head_part, body_part = rule.split(":-", 1)
        head = head_part.strip()
        body = body_part.strip().rstrip(".")

        # Parse head
        head_pred, head_args = parse_predicate_call(head)
        head_human = humanize_predicate_name(head_pred)

        # Translate body
        body_nl = _translate_body(body)

        # Construct natural language rule
        return f"A {head_human} {body_nl}."

    # Fallback
    return f"Rule: {rule}"


def _translate_body(body: str) -> str:
    """
    Translate the body of an ASP rule to natural language.

    Args:
        body: Body part of ASP rule

    Returns:
        Natural language translation
    """
    parts = []

    # Split by commas (conjunction)
    literals = [lit.strip() for lit in body.split(",")]

    for literal in literals:
        # Check for negation
        if literal.startswith("not "):
            negated_pred = literal[4:].strip()
            pred, args = parse_predicate_call(negated_pred)
            pred_human = humanize_predicate_name(pred)

            if pred in LEGAL_PREDICATE_TEMPLATES:
                template = LEGAL_PREDICATE_TEMPLATES[pred]
                if len(args) == 1:
                    text = template.format(arg=humanize_variable(args[0]))
                elif len(args) == 2:
                    text = template.format(
                        arg1=humanize_variable(args[0]), arg2=humanize_variable(args[1])
                    )
                else:
                    text = f"{pred_human} holds"
                parts.append(f"not {text}")
            else:
                parts.append(f"not {pred_human}")
        else:
            # Positive literal
            pred, args = parse_predicate_call(literal)
            pred_human = humanize_predicate_name(pred)

            if pred in LEGAL_PREDICATE_TEMPLATES:
                template = LEGAL_PREDICATE_TEMPLATES[pred]
                if len(args) == 1:
                    text = template.format(arg="it", arg1="it")
                elif len(args) == 2:
                    text = template.format(arg1="it", arg2=humanize_variable(args[1]))
                else:
                    text = pred_human
                parts.append(text)
            else:
                args_str = ", ".join(humanize_variable(arg) for arg in args)
                parts.append(f"{pred_human}({args_str})")

    # Join parts with "and"
    if len(parts) == 1:
        return f"if {parts[0]}"
    else:
        return f"if {', '.join(parts[:-1])} and {parts[-1]}"


def asp_facts_to_nl(facts: List[str]) -> str:
    """
    Convert set of ASP facts to narrative description.

    Args:
        facts: List of ASP facts

    Returns:
        Natural language narrative

    Example:
        >>> facts = ["contract(c1).", "land_sale_contract(c1).", "signed_by(w1, john)."]
        >>> asp_facts_to_nl(facts)
        'c1 is a contract. c1 is a land sale contract. w1 is signed by john.'
    """
    sentences = []

    for fact in facts:
        # Remove trailing period and whitespace
        fact = fact.rstrip(".").strip()

        # Parse predicate
        predicate, args = parse_predicate_call(fact)

        # Translate using template if available
        if predicate in LEGAL_PREDICATE_TEMPLATES:
            template = LEGAL_PREDICATE_TEMPLATES[predicate]

            if len(args) == 1:
                sentence = template.format(arg=args[0], arg1=args[0])
            elif len(args) == 2:
                sentence = template.format(arg=args[0], arg1=args[0], arg2=args[1])
            else:
                args_str = ", ".join(args)
                sentence = f"{humanize_predicate_name(predicate)}({args_str})"
        else:
            # Generic translation
            predicate_human = humanize_predicate_name(predicate)
            if args:
                args_str = ", ".join(args)
                sentence = f"{args_str} {predicate_human}"
            else:
                sentence = predicate_human

        sentences.append(sentence + ".")

    return " ".join(sentences)


class ASPToNLTranslator:
    """
    Translation system for ASP to Natural Language.

    Supports domain-specific templates and context enrichment.
    """

    def __init__(self, domain: str = "legal"):
        """
        Initialize translator with domain templates.

        Args:
            domain: Domain for specialized templates (default: "legal")
        """
        self.domain = domain
        self.templates = self._load_domain_templates(domain)
        logger.info(f"Initialized ASPToNLTranslator for domain: {domain}")

    def _load_domain_templates(self, domain: str) -> Dict[str, str]:
        """Load domain-specific templates."""
        if domain == "legal":
            return LEGAL_PREDICATE_TEMPLATES.copy()
        else:
            logger.warning(f"Unknown domain: {domain}, using empty templates")
            return {}

    def translate_query(
        self, query: str, context: Optional[ASPCore] = None
    ) -> TranslationResult:
        """
        Translate ASP query with full metadata.

        Args:
            query: ASP query string
            context: Optional ASP core for context

        Returns:
            TranslationResult with NL text and metadata
        """
        nl_text = asp_to_nl(query, context)
        predicates = extract_predicates(query)

        return TranslationResult(
            natural_language=nl_text,
            asp_source=query,
            predicates_used=predicates,
            confidence=1.0 if predicates and predicates[0] in self.templates else 0.7,
        )

    def translate_rule(self, rule: str) -> TranslationResult:
        """
        Translate ASP rule with full metadata.

        Args:
            rule: ASP rule string

        Returns:
            TranslationResult with NL text and metadata
        """
        nl_text = asp_rule_to_nl(rule)
        predicates = extract_predicates(rule)

        # Determine confidence based on template coverage
        covered = sum(1 for p in predicates if p in self.templates)
        confidence = covered / len(predicates) if predicates else 1.0

        return TranslationResult(
            natural_language=nl_text,
            asp_source=rule,
            predicates_used=predicates,
            confidence=confidence,
        )

    def translate_facts(self, facts: List[str]) -> TranslationResult:
        """
        Translate ASP facts with full metadata.

        Args:
            facts: List of ASP fact strings

        Returns:
            TranslationResult with NL text and metadata
        """
        nl_text = asp_facts_to_nl(facts)

        # Extract all predicates from all facts
        all_predicates = []
        for fact in facts:
            all_predicates.extend(extract_predicates(fact))

        # Remove duplicates
        predicates = list(dict.fromkeys(all_predicates))

        # Compute confidence
        covered = sum(1 for p in predicates if p in self.templates)
        confidence = covered / len(predicates) if predicates else 1.0

        return TranslationResult(
            natural_language=nl_text,
            asp_source="; ".join(facts),
            predicates_used=predicates,
            confidence=confidence,
        )


def enrich_context(query: str, asp_core: ASPCore) -> str:
    """
    Enrich ASP query with relevant context for LLM.

    Args:
        query: ASP query string
        asp_core: ASP core with rules and facts

    Returns:
        Enriched natural language query with context

    Example:
        Input: "enforceable(c1)?"
        Output:
            Relevant rules:
            - A contract is enforceable if it is not proven to be unenforceable.
            - A contract satisfies statute of frauds if it has a signed writing.

            Query: Is contract c1 enforceable?
    """
    # Extract predicates from query
    predicates = extract_predicates(query)

    if not predicates:
        # No predicates found, return basic translation
        return asp_to_nl(query, asp_core)

    # Get related rules from ASP core
    related_rules = []
    for predicate in predicates:
        rules = asp_core.stratified_programs.find_rules_mentioning(predicate)
        related_rules.extend(rules)

    # Remove duplicates
    unique_rules = list({rule.asp_text for rule in related_rules})

    # Build context
    context_parts = []

    if unique_rules:
        context_parts.append("Relevant rules:")
        for rule_text in unique_rules[:5]:  # Limit to 5 most relevant
            nl_rule = asp_rule_to_nl(rule_text)
            context_parts.append(f"- {nl_rule}")
        context_parts.append("")

    # Add query
    context_parts.append("Query:")
    context_parts.append(asp_to_nl(query, asp_core))

    return "\n".join(context_parts)


def asp_to_nl_statement(
    asp_code: str,
    context: Optional[ASPCore] = None,
    original_nl: Optional[str] = None,
    use_structural_templates: bool = True,
) -> str:
    """
    Convert ASP code to a semantic-preserving natural language statement.

    Unlike asp_to_nl() which generates questions, this function generates
    declarative statements that preserve semantic content for round-trip
    translation validation.

    Args:
        asp_code: ASP rule, fact, or query to translate
        context: Optional ASP core for enriched context
        original_nl: Optional original NL text to help with reconstruction
        use_structural_templates: Whether to use structural templates for
            improved sentence structure (default: True)

    Returns:
        Natural language statement preserving semantic content

    Examples:
        >>> asp_to_nl_statement("requires_writing(X) :- land_sale(X).")
        'Land sale contracts must be in writing'

        >>> asp_to_nl_statement("contract_valid(X) :- has_offer(X), has_acceptance(X).")
        'A contract is valid if it has offer and acceptance'
    """
    asp_code = asp_code.strip()

    # Check for exact rule match in templates first (highest fidelity)
    normalized_rule = _normalize_asp_rule(asp_code)
    if normalized_rule in RULE_STATEMENT_TEMPLATES:
        return RULE_STATEMENT_TEMPLATES[normalized_rule]

    # Try fuzzy matching with rule templates
    best_match = _find_best_rule_match(asp_code)
    if best_match:
        return best_match

    # Use structural templates for better sentence structure
    if use_structural_templates and ":-" in asp_code:
        structural_result = _apply_structural_translation(asp_code)
        if structural_result:
            return structural_result

    # Parse the ASP code and generate a statement
    if ":-" in asp_code:
        return _rule_to_statement(asp_code)
    else:
        return _fact_to_statement(asp_code)


def _apply_structural_translation(asp_code: str) -> Optional[str]:
    """
    Apply structural template-based translation for improved sentence structure.

    Uses rule type detection and structural templates to generate
    well-formed declarative sentences that improve structural accuracy
    in round-trip translation.

    Args:
        asp_code: ASP rule to translate

    Returns:
        Structured natural language sentence, or None if cannot apply
    """
    try:
        # Detect the rule type
        rule_type = detect_rule_type(asp_code)

        # Parse head and body
        asp_clean = asp_code.strip().rstrip(".")
        if ":-" not in asp_clean:
            return None

        head_part, body_part = asp_clean.split(":-", 1)
        head = head_part.strip()
        body = body_part.strip()

        # Parse head predicate
        head_pred, head_args = parse_predicate_call(head)

        # Parse body predicates
        body_predicates = _parse_body_predicates(body)

        # Apply structural template
        result = apply_structural_template(
            rule_type=rule_type,
            head_predicate=head_pred,
            head_args=head_args,
            body_predicates=body_predicates,
            original_asp=asp_code,
        )

        return result

    except Exception as e:
        logger.debug(f"Structural translation failed: {e}")
        return None


def _parse_body_predicates(body: str) -> List[tuple]:
    """
    Parse body literals into list of (predicate, args) tuples.

    Handles conjunction (,), negation (not), and comparisons.
    """
    predicates = []
    literals = _split_body_literals(body)

    for literal in literals:
        literal = literal.strip()

        # Skip comparisons for now
        if any(op in literal for op in [">=", "<=", ">", "<", "="]):
            continue

        # Handle negation
        if literal.startswith("not "):
            literal = literal[4:].strip()
            pred, args = parse_predicate_call(literal)
            predicates.append((f"not_{pred}", args))
        else:
            pred, args = parse_predicate_call(literal)
            if pred:  # Only add if we got a valid predicate
                predicates.append((pred, args))

    return predicates


def _normalize_asp_rule(rule: str) -> str:
    """Normalize ASP rule for template matching.

    Removes whitespace variations, standardizes format, and normalizes variables.
    """
    # Remove extra whitespace
    normalized = " ".join(rule.split())
    # Standardize spacing around operators
    normalized = re.sub(r"\s*:-\s*", " :- ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    # Ensure ends with period
    if not normalized.endswith("."):
        normalized += "."
        
    # Normalize variables to X, Y, Z...
    # Find all variables (single uppercase letters)
    variables = []
    for match in re.finditer(r"\b([A-Z])\b", normalized):
        var = match.group(1)
        if var not in variables:
            variables.append(var)
            
    # Create mapping to X, Y, Z...
    standard_vars = ["X", "Y", "Z", "W", "V", "U"]
    var_map = {}
    for i, var in enumerate(variables):
        if i < len(standard_vars):
            var_map[var] = standard_vars[i]
            
    # Apply mapping (careful not to replace parts of words)
    # We rebuild the string to safely replace variables
    result = ""
    i = 0
    while i < len(normalized):
        char = normalized[i]
        # Check if this is a variable (uppercase, word boundary)
        if char.isupper() and (i == 0 or not normalized[i-1].isalnum()) and (i == len(normalized)-1 or not normalized[i+1].isalnum()):
            result += var_map.get(char, char)
        else:
            result += char
        i += 1
        
    return result


def _find_best_rule_match(asp_code: str) -> Optional[str]:
    """Find the best matching rule template using predicate overlap.

    Returns the template statement if a good match is found.
    """
    predicates = extract_predicates(asp_code)
    if not predicates:
        return None

    best_score = 0.0
    best_template = None

    for rule_pattern, statement in RULE_STATEMENT_TEMPLATES.items():
        pattern_predicates = extract_predicates(rule_pattern)
        # Calculate Jaccard similarity
        common = len(set(predicates) & set(pattern_predicates))
        total = len(set(predicates) | set(pattern_predicates))
        score = common / total if total > 0 else 0

        if score > best_score and score >= 0.5:  # Require at least 50% overlap
            best_score = score
            best_template = statement

    return best_template


def _rule_to_statement(rule: str) -> str:
    """Convert an ASP rule to a declarative statement.

    Handles rules of the form: head :- body.
    """
    rule = rule.strip().rstrip(".")

    if ":-" not in rule:
        return _fact_to_statement(rule + ".")

    head_part, body_part = rule.split(":-", 1)
    head = head_part.strip()
    body = body_part.strip()

    # Parse head predicate
    head_pred, head_args = parse_predicate_call(head)
    head_statement = _predicate_to_statement(head_pred, head_args)

    # Parse body conditions
    body_conditions = _parse_body_conditions(body)

    if body_conditions:
        return f"{head_statement} if {body_conditions}"
    else:
        return head_statement


def _fact_to_statement(fact: str) -> str:
    """Convert an ASP fact to a declarative statement."""
    fact = fact.strip().rstrip(".")

    predicate, args = parse_predicate_call(fact)
    return _predicate_to_statement(predicate, args)


def _predicate_to_statement(predicate: str, args: List[str]) -> str:
    """Convert a predicate to a natural language statement.

    Uses STATEMENT_TEMPLATES for known predicates, falls back to
    LEGAL_PREDICATE_TEMPLATES, then generic translation.
    """
    # Check statement templates first (for single-predicate statements)
    if predicate in STATEMENT_TEMPLATES:
        template = STATEMENT_TEMPLATES[predicate]
        if "{arg}" in template and args:
            return template.format(arg=_humanize_entity(args[0]))
        elif "{elements}" in template and args:
            return template.format(elements=", ".join(args))
        elif "{" not in template:
            return template
        else:
            return template

    # Check legal predicate templates
    if predicate in LEGAL_PREDICATE_TEMPLATES:
        template = LEGAL_PREDICATE_TEMPLATES[predicate]
        if len(args) == 1:
            humanized = _humanize_entity(args[0])
            return template.format(arg=humanized, arg1=humanized)
        elif len(args) >= 2:
            arg1_human = _humanize_entity(args[0])
            arg2_human = _humanize_entity(args[1])
            return template.format(arg=arg1_human, arg1=arg1_human, arg2=arg2_human)

    # Generate from predicate name
    pred_human = humanize_predicate_name(predicate)

    if not args:
        return pred_human.capitalize()
    elif len(args) == 1:
        entity = _humanize_entity(args[0])
        # Generate a natural statement
        if pred_human.startswith("is_") or pred_human.startswith("has_"):
            verb = pred_human.replace("_", " ")
            return f"{entity} {verb}"
        elif pred_human.startswith("requires_"):
            requirement = pred_human.replace("requires_", "").replace("_", " ")
            return f"{entity} requires {requirement}"
        else:
            return f"{entity} {pred_human.replace('_', ' ')}"
    else:
        args_human = [_humanize_entity(arg) for arg in args]
        return f"{pred_human.replace('_', ' ')} applies to {', '.join(args_human)}"


def _humanize_entity(entity: str) -> str:
    """Convert an ASP entity to human-readable form.

    Handles variables (X, Y), constants (c1, w1), and compound terms.
    """
    # Handle anonymous variable
    if entity == "_":
        return "any entity"

    # Handle single uppercase variable
    if len(entity) == 1 and entity.isupper():
        entity_mapping = {
            "X": "a contract",
            "Y": "an entity",
            "C": "a contract",
            "W": "a writing",
            "P": "a party",
            "S": "a signature",
        }
        return entity_mapping.get(entity, "an entity")

    # Handle variable names
    if entity.isupper() or (entity[0].isupper() and "_" not in entity):
        return f"the {entity.lower()}"

    # Handle constants with type prefixes
    type_prefixes = {
        "contract_": "contract ",
        "writing_": "writing ",
        "party_": "party ",
        "c": "contract ",
        "w": "writing ",
        "p": "party ",
    }

    for prefix, replacement in type_prefixes.items():
        if entity.startswith(prefix) and len(entity) > len(prefix):
            remainder = entity[len(prefix) :]
            if remainder.isdigit() or remainder.isalnum():
                return replacement + remainder

    # Handle snake_case entities
    if "_" in entity:
        return entity.replace("_", " ")

    return entity


def _parse_body_conditions(body: str) -> str:
    """Parse ASP rule body conditions into natural language.

    Handles conjunction (,), negation (not), and comparisons.
    """
    parts = []

    # Split by comma, handling parentheses
    literals = _split_body_literals(body)

    for literal in literals:
        literal = literal.strip()

        # Handle negation
        if literal.startswith("not "):
            negated = literal[4:].strip()
            pred, args = parse_predicate_call(negated)
            pred_statement = _predicate_to_statement(pred, args)
            parts.append(f"it is not the case that {pred_statement}")
        # Handle comparisons
        elif any(op in literal for op in [">=", "<=", ">", "<", "=", "!="]):
            parts.append(_translate_comparison(literal))
        else:
            pred, args = parse_predicate_call(literal)
            pred_statement = _predicate_to_statement(pred, args)
            # Use "it" for repeated variables
            pred_statement = pred_statement.replace("a contract", "it")
            pred_statement = pred_statement.replace("the contract", "it")
            parts.append(pred_statement)

    if len(parts) == 0:
        return ""
    elif len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _split_body_literals(body: str) -> List[str]:
    """Split ASP body by commas while respecting parentheses."""
    literals = []
    current = []
    paren_depth = 0

    for char in body:
        if char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth -= 1
            current.append(char)
        elif char == "," and paren_depth == 0:
            literals.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        literals.append("".join(current).strip())

    return literals


def _translate_comparison(comparison: str) -> str:
    """Translate an ASP comparison to natural language."""
    op_translations = {
        ">=": "is at least",
        "<=": "is at most",
        ">": "is greater than",
        "<": "is less than",
        "=": "equals",
        "!=": "is not equal to",
    }

    for op, translation in op_translations.items():
        if op in comparison:
            parts = comparison.split(op)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return f"{left} {translation} {right}"

    return comparison


class SemanticPreservingTranslator:
    """Translator optimized for round-trip semantic preservation.

    This class provides high-fidelity ASP↔NL translation that preserves
    semantic content through translation round-trips.
    """

    def __init__(self, domain: str = "legal"):
        """Initialize the semantic-preserving translator.

        Args:
            domain: Domain for specialized templates (default: "legal")
        """
        self.domain = domain
        self.statement_templates = STATEMENT_TEMPLATES.copy()
        self.rule_templates = RULE_STATEMENT_TEMPLATES.copy()
        logger.info(f"Initialized SemanticPreservingTranslator for domain: {domain}")

    def translate_asp_to_statement(
        self,
        asp_code: str,
        original_nl: Optional[str] = None,
    ) -> TranslationResult:
        """Translate ASP to a semantic-preserving statement.

        Args:
            asp_code: ASP code to translate
            original_nl: Optional original NL for reference

        Returns:
            TranslationResult with statement and metadata
        """
        statement = asp_to_nl_statement(asp_code, original_nl=original_nl)
        predicates = extract_predicates(asp_code)

        # Calculate confidence based on template coverage
        template_coverage = self._calculate_template_coverage(predicates)

        return TranslationResult(
            natural_language=statement,
            asp_source=asp_code,
            predicates_used=predicates,
            confidence=template_coverage,
        )

    def _calculate_template_coverage(self, predicates: List[str]) -> float:
        """Calculate what proportion of predicates have templates."""
        if not predicates:
            return 1.0

        covered = sum(
            1
            for p in predicates
            if p in self.statement_templates or p in LEGAL_PREDICATE_TEMPLATES
        )
        return covered / len(predicates)
