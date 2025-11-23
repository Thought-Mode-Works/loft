"""
Pydantic schemas for validation pipeline results.

Defines structured outputs for each validation stage (syntactic, semantic,
empirical, consensus) and the overall validation report.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """
    Generic validation result for any validation stage.

    Used by syntactic, semantic, and other validators to return
    structured validation outcomes.
    """

    is_valid: bool = Field(description="Whether validation passed")
    error_messages: List[str] = Field(
        default_factory=list, description="List of error messages if invalid"
    )
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional validation details"
    )
    stage_name: str = Field(default="unknown", description="Name of validation stage")

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASS" if self.is_valid else "FAIL"
        lines = [f"{self.stage_name.title()} Validation: {status}"]

        if self.error_messages:
            lines.append("Errors:")
            for error in self.error_messages:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


class TestCase(BaseModel):
    """
    Test case for empirical validation.

    Represents a labeled example for testing ASP rule performance.
    """

    case_id: str = Field(description="Unique identifier for test case")
    description: str = Field(description="Human-readable description")
    facts: str = Field(description="ASP facts for this test case")
    query: str = Field(description="Query predicate to test (e.g., 'enforceable')")
    expected: Any = Field(description="Expected query result")
    category: str = Field(default="general", description="Test case category")


class FailureCase(BaseModel):
    """
    Represents a test case that failed validation.

    Captures what went wrong for failure analysis and debugging.
    """

    test_case: TestCase = Field(description="The test case that failed")
    expected: Any = Field(description="Expected result")
    actual: Any = Field(description="Actual result from rule")
    baseline: Any = Field(default=None, description="Baseline result without new rule")
    failure_type: str = Field(
        default="incorrect", description="Type of failure (incorrect, regression, etc.)"
    )


class EmpiricalValidationResult(BaseModel):
    """
    Results from empirical testing on labeled test cases.

    Measures actual performance improvement/degradation from adding a rule.
    """

    accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy with new rule (0.0-1.0)")
    baseline_accuracy: float = Field(
        ge=0.0, le=1.0, description="Accuracy without new rule (0.0-1.0)"
    )
    improvement: float = Field(description="Change from baseline (positive = improvement)")
    test_cases_passed: int = Field(description="Number of test cases passed")
    test_cases_failed: int = Field(description="Number of test cases failed")
    total_test_cases: int = Field(description="Total test cases evaluated")
    failures: List[FailureCase] = Field(
        default_factory=list, description="Failed test cases with details"
    )
    improvements: List[TestCase] = Field(
        default_factory=list,
        description="Cases that improved from baseline",
    )
    is_valid: bool = Field(description="Whether rule passes empirical validation")
    stage_name: str = Field(default="empirical", description="Validation stage name")

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASS" if self.is_valid else "FAIL"
        lines = [
            f"Empirical Validation: {status}",
            f"Accuracy: {self.accuracy:.2%} (baseline: {self.baseline_accuracy:.2%})",
            f"Improvement: {self.improvement:+.2%}",
            f"Test Cases: {self.test_cases_passed}/{self.total_test_cases} passed",
        ]

        if self.improvements:
            lines.append(f"Improved {len(self.improvements)} cases from baseline")

        if self.failures:
            lines.append(f"Failed {len(self.failures)} cases:")
            for failure in self.failures[:3]:  # Show first 3
                lines.append(f"  - {failure.test_case.description}")
            if len(self.failures) > 3:
                lines.append(f"  ... and {len(self.failures) - 3} more")

        return "\n".join(lines)


class ConsensusValidationResult(BaseModel):
    """
    Results from multi-LLM consensus voting.

    Aggregates votes from multiple LLMs to determine acceptance.
    """

    decision: Literal["accept", "reject", "revise"] = Field(description="Consensus decision")
    votes: List[Any] = Field(
        description="Individual votes from each LLM"
    )  # List[ConsensusVote] but avoid circular import
    accept_weight: float = Field(ge=0.0, description="Weighted votes for acceptance")
    reject_weight: float = Field(ge=0.0, description="Weighted votes for rejection")
    revise_weight: float = Field(ge=0.0, description="Weighted votes for revision")
    consensus_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of consensus (0.0-1.0)",
    )
    suggested_revisions: List[str] = Field(
        default_factory=list,
        description="Revision suggestions from voters",
    )
    is_valid: bool = Field(description="Whether consensus approves the rule")
    stage_name: str = Field(default="consensus", description="Validation stage name")

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASS" if self.is_valid else "FAIL"
        total_weight = self.accept_weight + self.reject_weight + self.revise_weight

        lines = [
            f"Consensus Validation: {status}",
            f"Decision: {self.decision.upper()}",
            f"Consensus Strength: {self.consensus_strength:.2%}",
            "",
            "Vote Distribution:",
            f"  Accept:  {self.accept_weight / total_weight:.1%}"
            if total_weight > 0
            else "  Accept:  0.0%",
            f"  Reject:  {self.reject_weight / total_weight:.1%}"
            if total_weight > 0
            else "  Reject:  0.0%",
            f"  Revise:  {self.revise_weight / total_weight:.1%}"
            if total_weight > 0
            else "  Revise:  0.0%",
        ]

        if self.suggested_revisions:
            lines.append(f"\nSuggested Revisions ({len(self.suggested_revisions)}):")
            for revision in self.suggested_revisions[:2]:
                lines.append(f"  - {revision[:80]}...")

        return "\n".join(lines)


class ValidationReport(BaseModel):
    """
    Comprehensive validation report for a generated rule.

    Aggregates results from all validation stages and provides
    final decision with rationale.
    """

    rule_asp: str = Field(description="The ASP rule being validated")
    rule_id: Optional[str] = Field(default=None, description="Identifier for the rule")
    target_layer: Literal["operational", "tactical", "strategic"] = Field(
        default="tactical",
        description="Target stratification layer",
    )
    stage_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from each validation stage",
    )
    aggregate_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence across stages",
    )
    final_decision: Literal["accept", "reject", "revise", "flag_for_review"] = Field(
        default="pending",
        description="Final validation decision",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection if applicable",
    )
    flag_reason: Optional[str] = Field(
        default=None,
        description="Reason for flagging for review",
    )
    suggested_revisions: List[str] = Field(
        default_factory=list,
        description="Suggested revisions if decision is 'revise'",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When validation was performed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def add_stage(self, stage_name: str, result: Any) -> None:
        """
        Add a validation stage result.

        Args:
            stage_name: Name of the stage (e.g., "syntax", "semantic")
            result: Validation result object
        """
        self.stage_results[stage_name] = result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 80,
            "VALIDATION REPORT",
            "=" * 80,
            "",
            f"Rule: {self.rule_asp[:100]}{'...' if len(self.rule_asp) > 100 else ''}",
            f"Decision: {self.final_decision.upper()}",
            f"Confidence: {self.aggregate_confidence:.2%}",
            f"Target Layer: {self.target_layer}",
            f"Timestamp: {self.timestamp.isoformat()}",
            "",
            "STAGE RESULTS:",
            "-" * 80,
        ]

        # Add stage summaries
        for stage_name, result in self.stage_results.items():
            if hasattr(result, "summary"):
                lines.append(result.summary())
            else:
                status = "PASS" if getattr(result, "is_valid", False) else "FAIL"
                lines.append(f"{stage_name.title()}: {status}")
            lines.append("")

        # Add decision rationale
        lines.append("-" * 80)
        lines.append("FINAL DECISION RATIONALE:")
        lines.append(f"Decision: {self.final_decision.upper()}")

        if self.rejection_reason:
            lines.append(f"Rejection Reason: {self.rejection_reason}")

        if self.flag_reason:
            lines.append(f"Flagged: {self.flag_reason}")

        if self.suggested_revisions:
            lines.append(f"\nSuggested Revisions ({len(self.suggested_revisions)}):")
            for i, revision in enumerate(self.suggested_revisions[:3], 1):
                lines.append(f"  {i}. {revision[:100]}{'...' if len(revision) > 100 else ''}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_asp": self.rule_asp,
            "rule_id": self.rule_id,
            "target_layer": self.target_layer,
            "aggregate_confidence": self.aggregate_confidence,
            "final_decision": self.final_decision,
            "rejection_reason": self.rejection_reason,
            "flag_reason": self.flag_reason,
            "suggested_revisions": self.suggested_revisions,
            "timestamp": self.timestamp.isoformat(),
            "stage_count": len(self.stage_results),
            "metadata": self.metadata,
        }
