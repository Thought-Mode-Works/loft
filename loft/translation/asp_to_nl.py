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
    "requires_writing(X) :- goods_over_500(X)": (
        "Contracts for goods over $500 require a written memorandum"
    ),
    "requires_writing(X) :- suretyship(X)": (
        "Suretyship agreements must be in writing"
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

    # Parse the ASP code and generate a statement
    if ":-" in asp_code:
        return _rule_to_statement(asp_code)
    else:
        return _fact_to_statement(asp_code)


def _normalize_asp_rule(rule: str) -> str:
    """Normalize ASP rule for template matching.

    Removes whitespace variations and standardizes format.
    """
    # Remove extra whitespace
    normalized = " ".join(rule.split())
    # Standardize spacing around operators
    normalized = re.sub(r"\s*:-\s*", " :- ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    # Ensure ends with period
    if not normalized.endswith("."):
        normalized += "."
    return normalized


def _find_best_rule_match(asp_code: str) -> Optional[str]:
    """Find the best matching rule template using predicate overlap.

    Returns the template statement if a good match is found.
    """
    predicates = extract_predicates(asp_code)
    if not predicates:
        return None

    best_score = 0
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
