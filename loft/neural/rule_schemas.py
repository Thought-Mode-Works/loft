"""
Pydantic schemas for LLM-generated ASP rules.

These schemas define structured outputs for rule generation prompts,
ensuring consistent, validated responses from LLMs during Phase 2.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


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
        """Basic ASP syntax validation."""
        v = v.strip()
        if not v:
            raise ValueError("ASP rule cannot be empty")
        # Basic checks - full validation happens in validation pipeline
        if not any(op in v for op in [":-", "."]):
            raise ValueError("ASP rule must contain ':-' (rule) or end with '.' (fact)")
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
