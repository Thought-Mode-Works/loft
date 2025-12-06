"""
Pydantic schemas for LLM-generated ASP rules.

These schemas define structured outputs for rule generation prompts,
ensuring consistent, validated responses from LLMs during Phase 2.

Issue #164: Added partial candidate acceptance to reduce API waste.
When LLM generates multiple candidates and some are invalid, we now
filter invalid ones rather than rejecting the entire response ,this is in 
order to recycle the usable responses. 
"""

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Set, Dict, Any
from pydantic import BaseModel, Field, field_validator
from loguru import logger


@dataclass
class ValidationFailureRecord:
    """Record of a validation failure for metrics tracking."""

    candidate_index: int
    error_message: str
    asp_rule_attempted: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()


@dataclass
class CandidateValidationMetrics:
    """
    Tracks validation failure metrics for candidate filtering.

    This addresses issue #164 by providing visibility into how many
    candidates are being filtered vs accepted.
    """

    total_candidates_received: int = 0
    valid_candidates_accepted: int = 0
    invalid_candidates_filtered: int = 0
    failure_records: List[ValidationFailureRecord] = field(default_factory=list)

    def record_valid(self) -> None:
        """Record a successfully validated candidate."""
        self.total_candidates_received += 1
        self.valid_candidates_accepted += 1

    def record_invalid(self, candidate_index: int, error_message: str, asp_rule: str) -> None:
        """Record a filtered invalid candidate."""
        self.total_candidates_received += 1
        self.invalid_candidates_filtered += 1
        self.failure_records.append(
            ValidationFailureRecord(
                candidate_index=candidate_index,
                error_message=error_message,
                asp_rule_attempted=asp_rule[:200] if asp_rule else "",
            )
        )

    def get_acceptance_rate(self) -> float:
        """Get the percentage of candidates accepted."""
        if self.total_candidates_received == 0:
            return 1.0
        return self.valid_candidates_accepted / self.total_candidates_received

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation metrics."""
        return {
            "total_candidates": self.total_candidates_received,
            "valid_accepted": self.valid_candidates_accepted,
            "invalid_filtered": self.invalid_candidates_filtered,
            "acceptance_rate": self.get_acceptance_rate(),
            "recent_failures": [
                {
                    "index": r.candidate_index,
                    "error": r.error_message[:100],
                }
                for r in self.failure_records[-5:]
            ],
        }


# Global metrics tracker for validation failures
_validation_metrics = CandidateValidationMetrics()


def get_validation_metrics() -> CandidateValidationMetrics:
    """Get the global validation metrics tracker."""
    return _validation_metrics


def reset_validation_metrics() -> None:
    """Reset the global validation metrics (useful for testing)."""
    global _validation_metrics
    _validation_metrics = CandidateValidationMetrics()


def extract_body_predicates(rule: str) -> Set[str]:
    """
    Extract predicate names from the body of an ASP rule.

    Args:
        rule: The ASP rule string

    Returns:
        Set of predicate names found in the rule body
    """
    predicates = set()

    if ":-" not in rule:
        return predicates  # It's a fact, no body predicates

    # Get the body part
    parts = rule.split(":-", 1)
    if len(parts) != 2:
        return predicates

    body = parts[1].strip().rstrip(".")

    # Find all predicate names (word followed by opening paren)
    # Handle negation: "not predicate(X)" -> "predicate"
    pattern = r"(?:not\s+)?([a-z_][a-zA-Z0-9_]*)\s*\("
    matches = re.findall(pattern, body)
    predicates.update(matches)

    return predicates


def extract_head_predicate(rule: str) -> Optional[str]:
    """
    Extract the predicate name from the head of an ASP rule.

    Args:
        rule: The ASP rule string

    Returns:
        The head predicate name, or None if not found
    """
    if ":-" in rule:
        head = rule.split(":-", 1)[0].strip()
    else:
        # It's a fact
        head = rule.strip().rstrip(".")

    # Extract predicate name
    match = re.match(r"([a-z_][a-zA-Z0-9_]*)\s*\(", head)
    if match:
        return match.group(1)

    return None


def check_undefined_predicates(
    rule: str, known_predicates: Set[str], strict: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Check if rule references predicates that don't exist in the known set.

    Args:
        rule: The ASP rule to check
        known_predicates: Set of known predicate names
        strict: If True, return undefined as errors; otherwise as warnings

    Returns:
        Tuple of (errors, warnings) listing undefined predicates
    """
    errors = []
    warnings = []

    body_predicates = extract_body_predicates(rule)
    undefined = body_predicates - known_predicates

    if undefined:
        msg = f"Undefined predicates in rule body: {', '.join(sorted(undefined))}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    return errors, warnings


def validate_rule_grounds(rule: str, sample_facts: str) -> Tuple[bool, Optional[str]]:
    """
    Verify that a rule actually grounds with sample facts.

    A rule that doesn't ground with relevant facts will never fire.

    Args:
        rule: The ASP rule to test
        sample_facts: Sample facts to test grounding with

    Returns:
        Tuple of (grounds_successfully, error_message)
    """
    try:
        import clingo

        program = f"{rule}\n{sample_facts}"
        ctl = clingo.Control(["0"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        # Check if the head predicate was derived
        head_pred = extract_head_predicate(rule)
        if not head_pred:
            return True, None  # Can't check, assume OK

        # Get answer set and check for head predicate
        derived = []

        def on_model(model: clingo.Model) -> bool:
            for atom in model.symbols(shown=True):
                derived.append(str(atom))
            return False  # Stop after first model

        ctl.solve(on_model=on_model)

        # Check if any atom with the head predicate was derived
        head_derived = any(atom.startswith(f"{head_pred}(") for atom in derived)

        if head_derived:
            return True, None
        else:
            return False, f"Rule did not derive any {head_pred}() atoms with given facts"

    except Exception as e:
        return False, f"Grounding test failed: {str(e)}"


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


# =============================================================================
# Lenient Schemas for Partial Candidate Acceptance (Issue #164)
# =============================================================================


class LenientGeneratedRule(BaseModel):
    """
    A lenient version of GeneratedRule that doesn't validate ASP syntax.

    Used internally for filtering candidates - the raw rule is stored
    without validation, then validated separately.
    """

    asp_rule: str = Field(description="ASP rule in Clingo syntax (not validated during parsing)")
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

    def to_generated_rule(self) -> GeneratedRule:
        """Convert to a strict GeneratedRule (may raise validation error)."""
        return GeneratedRule(
            asp_rule=self.asp_rule,
            confidence=self.confidence,
            reasoning=self.reasoning,
            predicates_used=self.predicates_used,
            new_predicates=self.new_predicates,
            alternative_formulations=self.alternative_formulations,
            source_type=self.source_type,
            source_text=self.source_text,
            citation=self.citation,
            jurisdiction=self.jurisdiction,
        )


class LenientRuleCandidate(BaseModel):
    """
    A lenient version of RuleCandidate for partial acceptance.

    Uses LenientGeneratedRule internally to defer ASP validation.
    """

    rule: LenientGeneratedRule = Field(description="The generated rule (not validated)")
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

    def to_rule_candidate(self) -> RuleCandidate:
        """Convert to a strict RuleCandidate (may raise validation error)."""
        return RuleCandidate(
            rule=self.rule.to_generated_rule(),
            applicability_score=self.applicability_score,
            test_cases_passed=self.test_cases_passed,
            test_cases_failed=self.test_cases_failed,
            complexity_score=self.complexity_score,
        )


class LenientGapFillingResponse(BaseModel):
    """
    Gap-filling response with partial candidate acceptance.

    This schema addresses issue #164: Instead of rejecting the entire
    response when one candidate has invalid ASP syntax, it filters
    invalid candidates and accepts the valid ones.

    Usage:
        # Parse raw LLM output leniently
        raw_response = LenientGapFillingResponse.model_validate(llm_output)

        # Convert to strict response (filters invalid candidates)
        strict_response = raw_response.to_strict_response()

        # Check how many were filtered
        metrics = get_validation_metrics()
        print(f"Filtered {metrics.invalid_candidates_filtered} invalid candidates")
    """

    gap_description: str = Field(description="Description of the knowledge gap being addressed")
    missing_predicate: str = Field(description="The specific predicate that needs to be defined")
    candidates: List[LenientRuleCandidate] = Field(
        description="Candidate rules (not validated during parsing)", min_length=1
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

    # Internal tracking of filtered candidates
    _filtered_candidates: List[Tuple[int, str, str]] = []

    def to_strict_response(self) -> GapFillingResponse:
        """
        Convert to a strict GapFillingResponse, filtering invalid candidates.

        This method validates each candidate individually and keeps only
        the valid ones. Invalid candidates are logged and tracked in metrics.

        Returns:
            GapFillingResponse with only valid candidates

        Raises:
            ValueError: If no valid candidates remain after filtering
        """
        valid_candidates: List[RuleCandidate] = []
        filtered_info: List[Tuple[int, str, str]] = []
        metrics = get_validation_metrics()

        for idx, candidate in enumerate(self.candidates):
            try:
                # Attempt to convert to strict candidate
                strict_candidate = candidate.to_rule_candidate()
                valid_candidates.append(strict_candidate)
                metrics.record_valid()

                logger.debug(
                    f"Candidate {idx} validated successfully: {candidate.rule.asp_rule[:50]}..."
                )

            except Exception as e:
                error_msg = str(e)
                asp_rule = candidate.rule.asp_rule

                # Log the filtered candidate
                logger.warning(
                    f"Filtered invalid candidate {idx}: {error_msg[:100]}. Rule: {asp_rule[:80]}..."
                )

                # Track in metrics
                metrics.record_invalid(idx, error_msg, asp_rule)
                filtered_info.append((idx, error_msg, asp_rule))

        self._filtered_candidates = filtered_info

        if not valid_candidates:
            # All candidates were invalid - this is a real failure
            raise ValueError(
                f"All {len(self.candidates)} candidates were invalid. "
                f"Errors: {[f[1][:50] for f in filtered_info]}"
            )

        # Log summary of filtering
        if filtered_info:
            logger.info(
                f"Partial acceptance: {len(valid_candidates)} valid candidates, "
                f"{len(filtered_info)} filtered (out of {len(self.candidates)} total)"
            )

        # Adjust recommended_index if needed
        adjusted_recommended_index = self._adjust_recommended_index(
            self.recommended_index, filtered_info, len(valid_candidates)
        )

        return GapFillingResponse(
            gap_description=self.gap_description,
            missing_predicate=self.missing_predicate,
            candidates=valid_candidates,
            recommended_index=adjusted_recommended_index,
            requires_validation=self.requires_validation,
            test_cases_needed=self.test_cases_needed,
            confidence=self.confidence,
        )

    def _adjust_recommended_index(
        self,
        original_index: int,
        filtered: List[Tuple[int, str, str]],
        valid_count: int,
    ) -> int:
        """
        Adjust recommended_index after filtering invalid candidates.

        If the originally recommended candidate was filtered, pick the
        first valid one (index 0). Otherwise, adjust for removed indices.
        """
        filtered_indices = {f[0] for f in filtered}

        if original_index in filtered_indices:
            # The recommended candidate was invalid - use first valid
            logger.warning(
                f"Recommended candidate {original_index} was invalid, "
                "falling back to first valid candidate"
            )
            return 0

        # Count how many candidates before this index were filtered
        offset = sum(1 for idx in filtered_indices if idx < original_index)
        adjusted = original_index - offset

        # Ensure within bounds
        return min(adjusted, valid_count - 1)

    def get_filtered_candidates(self) -> List[Tuple[int, str, str]]:
        """
        Get information about candidates that were filtered.

        Returns:
            List of (original_index, error_message, asp_rule) tuples
        """
        return self._filtered_candidates.copy()


def parse_gap_filling_response_lenient(
    raw_data: Dict[str, Any],
) -> Tuple[GapFillingResponse, List[Tuple[int, str, str]]]:
    """
    Parse LLM output into GapFillingResponse with partial candidate acceptance.

    This is the recommended way to parse gap-filling responses, as it
    accepts partially valid responses rather than failing entirely.

    Args:
        raw_data: Raw dictionary from LLM structured output

    Returns:
        Tuple of (validated_response, filtered_candidates_info)
        where filtered_candidates_info is a list of
        (original_index, error_message, asp_rule) tuples

    Raises:
        ValueError: If all candidates are invalid or other validation fails
    """
    # First parse leniently (no ASP validation)
    lenient = LenientGapFillingResponse.model_validate(raw_data)

    # Convert to strict (filters invalid candidates)
    strict = lenient.to_strict_response()

    return strict, lenient.get_filtered_candidates()
