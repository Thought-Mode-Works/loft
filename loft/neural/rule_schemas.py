"""
Pydantic schemas for LLM-generated ASP rules.

These schemas define structured outputs for rule generation prompts,
ensuring consistent, validated responses from LLMs during Phase 2.
"""

import re
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from loguru import logger


def validate_asp_rule_completeness(rule: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that an ASP rule is complete and well-formed.

    Checks for common truncation patterns and structural completeness.

    Args:
        rule: The ASP rule string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    rule = rule.strip()

    if not rule:
        return False, "ASP rule cannot be empty"

    # Split into individual statements (rules/facts)
    # Handle multi-line rules by first normalizing whitespace
    normalized = re.sub(r"\s+", " ", rule)

    # Check 1: Must end with a period
    if not normalized.rstrip().endswith("."):
        return False, "ASP rule must end with a period '.'"

    # Check 2: Balanced parentheses
    open_parens = normalized.count("(")
    close_parens = normalized.count(")")
    if open_parens != close_parens:
        return False, f"Unbalanced parentheses: {open_parens} '(' vs {close_parens} ')'"

    # Check 3: Balanced brackets (for aggregates)
    open_brackets = normalized.count("{")
    close_brackets = normalized.count("}")
    if open_brackets != close_brackets:
        return (
            False,
            f"Unbalanced brackets: {open_brackets} '{{' vs {close_brackets} '}}'",
        )

    # Check 4: No trailing incomplete identifiers before the period
    # Pattern: identifier followed by underscore or incomplete predicate
    # e.g., "land_sale_" or "predicate(" without closing
    truncation_patterns = [
        (r"[a-z_]+_\s*\.", "Rule appears truncated (ends with trailing underscore)"),
        (r"[a-z_]+\(\s*\.", "Rule appears truncated (empty predicate arguments)"),
        (r",\s*\.", "Rule appears truncated (trailing comma before period)"),
        (r":-\s*\.", "Rule appears truncated (empty rule body)"),
        (r"[a-z_]+\([^)]*,\s*\.", "Rule appears truncated (incomplete argument list)"),
    ]

    for pattern, message in truncation_patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return False, message

    # Check 5: If it's a rule (has :-), verify head and body are present
    if ":-" in normalized:
        parts = normalized.split(":-", 1)
        head = parts[0].strip()
        body = parts[1].strip()

        # Head must be a valid predicate
        if not head:
            return False, "Rule has empty head"

        # Head must look like a predicate (name followed by optional args)
        if not re.match(r"^[a-z_][a-zA-Z0-9_]*(\([^)]+\))?$", head):
            # Allow for more complex heads like "not pred(X)" for constraints
            if not re.match(r"^(not\s+)?[a-z_][a-zA-Z0-9_]*(\([^)]+\))?$", head):
                # Could be a constraint (empty head)
                if head != "":
                    return False, f"Invalid rule head format: '{head}'"

        # Body must not be just a period
        body_content = body.rstrip(".")
        if not body_content.strip():
            return False, "Rule has empty body"

        # Note: We removed the body predicate completeness check here because:
        # 1. Splitting by comma breaks multiline rules with multi-argument predicates
        #    e.g., "pred(A, B)" split by comma gives ["pred(A", "B)"] incorrectly
        # 2. Check 7 (clingo parsing) is the definitive validation and will catch
        #    actual syntax errors in rule bodies

    # Check 6: No orphan characters after the period (garbage at end)
    # Split by period and check if anything meaningful comes after
    statements = [s.strip() for s in normalized.split(".") if s.strip()]
    for stmt in statements:
        # Each statement should start with a valid identifier or be a comment
        if stmt and not re.match(r"^(%.*|[a-z_#:])", stmt):
            # Could be a single letter orphan
            if len(stmt) <= 2 and stmt.isalpha():
                return False, f"Orphan character detected after rule: '{stmt}'"

    # Check 7: Try to parse with clingo for definitive validation
    try:
        import clingo

        ctl = clingo.Control(["0"])
        ctl.add("base", [], rule)
        # Ground to catch more errors
        ctl.ground([("base", [])])
        return True, None
    except Exception as e:
        error_msg = str(e)
        # Extract the most relevant part of the error
        if "parsing failed" in error_msg.lower():
            return False, f"Clingo parsing failed: {error_msg}"
        elif "syntax error" in error_msg.lower():
            return False, f"ASP syntax error: {error_msg}"
        else:
            return False, f"ASP validation error: {error_msg}"


class GeneratedRule(BaseModel):
    """
    Schema for a single LLM-generated ASP rule.

    Represents a candidate rule with metadata, confidence scoring,
    and provenance tracking for validation pipeline integration.
    """

    asp_rule: str = Field(description="ASP rule in Clingo syntax (e.g., 'enforceable(C) :- ...')")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in rule validity (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Natural language explanation of why this rule was generated"
    )
    predicates_used: List[str] = Field(
        description="List of existing predicates referenced in the rule"
    )
    new_predicates: List[str] = Field(
        default_factory=list,
        description="List of new predicates introduced by this rule",
    )
    alternative_formulations: List[str] = Field(
        default_factory=list, description="Alternative ASP formulations if ambiguous"
    )
    source_type: Literal["principle", "case", "gap_fill", "refinement"] = Field(
        description="Type of source that generated this rule"
    )
    source_text: str = Field(description="Original natural language text that motivated the rule")
    citation: Optional[str] = Field(
        default=None, description="Legal citation if from case law or statute"
    )
    jurisdiction: Optional[str] = Field(
        default=None, description="Jurisdiction if applicable (e.g., 'CA', 'Federal')"
    )

    @field_validator("asp_rule")
    @classmethod
    def validate_asp_syntax(cls, v: str) -> str:
        """
        Comprehensive ASP syntax validation.

        Validates that the rule is:
        1. Complete (not truncated)
        2. Structurally valid (balanced parens, proper termination)
        3. Parseable by clingo
        """
        v = v.strip()

        # Run comprehensive validation
        is_valid, error_msg = validate_asp_rule_completeness(v)

        if not is_valid:
            logger.warning(f"ASP rule validation failed: {error_msg}")
            raise ValueError(f"Invalid ASP rule: {error_msg}")

        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Ensure confidence is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class RuleCandidate(BaseModel):
    """
    A candidate rule with additional metadata for gap-filling scenarios.

    Used when multiple rules might fill a knowledge gap, allowing
    the system to compare and select the best option.
    """

    rule: GeneratedRule = Field(description="The generated rule")
    applicability_score: float = Field(
        ge=0.0, le=1.0, description="How well this rule addresses the gap (0.0-1.0)"
    )
    test_cases_passed: Optional[int] = Field(
        default=None, description="Number of test cases this rule would pass"
    )
    test_cases_failed: Optional[int] = Field(
        default=None, description="Number of test cases this rule would fail"
    )
    complexity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Rule complexity (0.0=simple, 1.0=complex)",
        default=0.5,
    )


class GapFillingResponse(BaseModel):
    """
    Response schema for gap-filling rule generation.

    When the symbolic core identifies a knowledge gap, this schema
    structures the LLM's response with multiple candidates and
    recommendations.
    """

    gap_description: str = Field(description="Description of the knowledge gap being addressed")
    missing_predicate: str = Field(description="The specific predicate that needs to be defined")
    candidates: List[RuleCandidate] = Field(
        description="Candidate rules that could fill the gap", min_length=1
    )
    recommended_index: int = Field(description="Index of recommended candidate (0-based)", ge=0)
    requires_validation: bool = Field(description="Whether this gap-fill requires human validation")
    test_cases_needed: List[str] = Field(
        description="Test cases needed to validate the generated rules"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the gap-filling solution",
    )

    @field_validator("recommended_index")
    @classmethod
    def validate_recommended_index(cls, v: int, info) -> int:
        """Ensure recommended index is within candidates range."""
        # Can't validate against candidates here due to validation order
        # Will be checked at runtime
        return v


class ConsensusVote(BaseModel):
    """
    Schema for multi-LLM consensus voting on proposed rules.

    Represents one LLM's vote on whether to accept, reject, or revise
    a proposed rule, with detailed justification.
    """

    vote: Literal["accept", "reject", "revise"] = Field(description="Vote on the proposed rule")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this vote (0.0-1.0)",
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="List of issues identified in the proposed rule",
    )
    suggested_revision: Optional[str] = Field(
        default=None, description="Suggested ASP rule revision if vote='revise'"
    )
    test_cases_to_validate: List[str] = Field(
        default_factory=list,
        description="Test cases that should be used to validate the rule",
    )
    reasoning: str = Field(description="Detailed explanation of the vote and any issues")

    @field_validator("suggested_revision")
    @classmethod
    def validate_revision(cls, v: Optional[str], info) -> Optional[str]:
        """If vote is 'revise', suggested_revision should be provided."""
        if info.data.get("vote") == "revise" and not v:
            raise ValueError("Suggested revision required when vote='revise'")
        return v


class PrincipleToRuleRequest(BaseModel):
    """
    Request schema for converting legal principles to ASP rules.

    Structures the input when asking an LLM to generate rules from
    codified legal principles.
    """

    principle_text: str = Field(description="Natural language statement of the legal principle")
    domain: str = Field(description="Legal domain (e.g., 'contract_law', 'torts')")
    existing_predicates: List[str] = Field(
        default_factory=list,
        description="Existing predicates that can be referenced",
    )
    constraints: Optional[str] = Field(
        default=None, description="Any constraints on the rule generation"
    )


class CaseToRuleRequest(BaseModel):
    """
    Request schema for extracting rules from case law.

    Structures the input when asking an LLM to extract holdings
    from judicial opinions.
    """

    case_text: str = Field(description="Excerpt from judicial opinion")
    citation: str = Field(description="Case citation (e.g., 'Smith v. Jones, 123 F.3d 456')")
    jurisdiction: str = Field(description="Jurisdiction (e.g., 'CA', 'Federal')")
    domain: str = Field(description="Legal domain")
    existing_predicates: List[str] = Field(
        default_factory=list,
        description="Existing predicates that can be referenced",
    )
    focus: Optional[str] = Field(
        default=None,
        description="Specific aspect to focus on (e.g., 'statute of frauds exception')",
    )
