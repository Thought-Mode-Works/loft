"""
Failure Analysis Module for Meta-Reasoning.

This module provides automated failure analysis and error diagnosis:
- Error classification by type
- Root cause analysis
- Failure pattern detection
- Actionable recommendations

Integrates with the reasoning observer and strategy evaluator
to provide comprehensive failure insights.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from loft.meta.schemas import (
    FailureDiagnosis,
    ImprovementPriority,
    ReasoningChain,
    ReasoningStepType,
)


class ErrorCategory(Enum):
    """Categories of prediction errors."""

    RULE_COVERAGE_GAP = "rule_coverage_gap"  # No applicable rule
    RULE_CONFLICT = "rule_conflict"  # Contradictory rules
    TRANSLATION_ERROR = "translation_error"  # NL↔ASP translation failure
    PREDICATE_MISMATCH = "predicate_mismatch"  # Wrong predicate mapping
    EDGE_CASE = "edge_case"  # Unusual case characteristics
    DOMAIN_BOUNDARY = "domain_boundary"  # Cross-domain confusion
    VALIDATION_FAILURE = "validation_failure"  # Rule validation failed
    INFERENCE_ERROR = "inference_error"  # ASP solver error
    CONFIDENCE_THRESHOLD = "confidence_threshold"  # Below confidence threshold
    UNKNOWN = "unknown"  # Unclassified error


class RootCauseType(Enum):
    """Types of root causes for failures."""

    MISSING_RULE = "missing_rule"
    INCORRECT_RULE = "incorrect_rule"
    AMBIGUOUS_INPUT = "ambiguous_input"
    PROMPT_WEAKNESS = "prompt_weakness"
    DATA_QUALITY = "data_quality"
    MODEL_LIMITATION = "model_limitation"
    DOMAIN_MISMATCH = "domain_mismatch"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"


class RecommendationCategory(Enum):
    """Categories of improvement recommendations."""

    RULE_ADDITION = "rule_addition"
    RULE_MODIFICATION = "rule_modification"
    PROMPT_IMPROVEMENT = "prompt_improvement"
    VALIDATION_ENHANCEMENT = "validation_enhancement"
    DOMAIN_EXPANSION = "domain_expansion"
    DATA_AUGMENTATION = "data_augmentation"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"


@dataclass
class PredictionError:
    """
    A prediction error with full context.

    Captures all information about a failed prediction including
    the reasoning chain that led to the error.
    """

    error_id: str
    case_id: str
    domain: str
    predicted: str
    actual: str
    reasoning_chain: Optional[ReasoningChain] = None
    contributing_rules: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "case_id": self.case_id,
            "domain": self.domain,
            "predicted": self.predicted,
            "actual": self.actual,
            "reasoning_chain": (
                self.reasoning_chain.to_dict() if self.reasoning_chain else None
            ),
            "contributing_rules": self.contributing_rules,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RootCause:
    """
    A root cause identified for a failure.

    Represents a specific cause that contributed to a prediction error.
    """

    cause_id: str
    cause_type: RootCauseType
    description: str
    confidence: float  # 0-1 confidence in this being the root cause
    evidence: List[str] = field(default_factory=list)
    affected_step_types: List[ReasoningStepType] = field(default_factory=list)
    remediation_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cause_id": self.cause_id,
            "cause_type": self.cause_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "affected_step_types": [st.value for st in self.affected_step_types],
            "remediation_hints": self.remediation_hints,
        }


@dataclass
class RootCauseAnalysis:
    """
    Complete root cause analysis for an error.

    Provides a comprehensive analysis of why a prediction failed,
    including multiple potential causes ranked by likelihood.
    """

    analysis_id: str
    error_id: str
    primary_cause: RootCause
    secondary_causes: List[RootCause] = field(default_factory=list)
    analysis_confidence: float = 0.0
    analysis_method: str = "heuristic"
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "error_id": self.error_id,
            "primary_cause": self.primary_cause.to_dict(),
            "secondary_causes": [c.to_dict() for c in self.secondary_causes],
            "analysis_confidence": self.analysis_confidence,
            "analysis_method": self.analysis_method,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class FailurePattern:
    """
    A recurring failure pattern.

    Represents a pattern of failures that share common characteristics,
    enabling targeted improvements.
    """

    pattern_id: str
    name: str
    description: str
    error_category: ErrorCategory
    error_count: int
    affected_domains: List[str] = field(default_factory=list)
    common_characteristics: Dict[str, Any] = field(default_factory=dict)
    root_causes: List[RootCause] = field(default_factory=list)
    example_error_ids: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    severity: str = "medium"  # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "error_category": self.error_category.value,
            "error_count": self.error_count,
            "affected_domains": self.affected_domains,
            "common_characteristics": self.common_characteristics,
            "root_causes": [c.to_dict() for c in self.root_causes],
            "example_error_ids": self.example_error_ids,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "severity": self.severity,
        }


@dataclass
class Recommendation:
    """
    An actionable improvement recommendation.

    Provides specific guidance on how to address identified failures.
    """

    recommendation_id: str
    category: RecommendationCategory
    title: str
    description: str
    priority: ImprovementPriority
    expected_impact: float  # Expected improvement percentage
    target_patterns: List[str] = field(default_factory=list)
    target_domains: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    success_criteria: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "expected_impact": self.expected_impact,
            "target_patterns": self.target_patterns,
            "target_domains": self.target_domains,
            "implementation_steps": self.implementation_steps,
            "estimated_effort": self.estimated_effort,
            "success_criteria": self.success_criteria,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class FailureAnalysisReport:
    """
    Comprehensive failure analysis report.

    Summarizes all failure analysis findings including patterns,
    root causes, and recommendations.
    """

    report_id: str
    analysis_period_start: datetime
    analysis_period_end: datetime
    total_errors_analyzed: int
    error_category_distribution: Dict[str, int]
    patterns_identified: List[FailurePattern]
    top_root_causes: List[RootCause]
    recommendations: List[Recommendation]
    overall_error_rate: float
    domains_analyzed: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "analysis_period_start": self.analysis_period_start.isoformat(),
            "analysis_period_end": self.analysis_period_end.isoformat(),
            "total_errors_analyzed": self.total_errors_analyzed,
            "error_category_distribution": self.error_category_distribution,
            "patterns_identified": [p.to_dict() for p in self.patterns_identified],
            "top_root_causes": [c.to_dict() for c in self.top_root_causes],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "overall_error_rate": self.overall_error_rate,
            "domains_analyzed": self.domains_analyzed,
            "generated_at": self.generated_at.isoformat(),
        }


class FailureAnalyzer:
    """
    Analyzes and diagnoses prediction failures.

    Provides comprehensive failure analysis including:
    - Error classification
    - Root cause identification
    - Failure pattern detection
    - Integration with reasoning chains
    """

    def __init__(self):
        """Initialize the failure analyzer."""
        self._errors: Dict[str, PredictionError] = {}
        self._classifications: Dict[str, ErrorCategory] = {}
        self._root_cause_analyses: Dict[str, RootCauseAnalysis] = {}
        self._patterns: Dict[str, FailurePattern] = {}

        # Statistics tracking
        self._category_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self._domain_error_counts: Dict[str, int] = defaultdict(int)

    def record_error(self, error: PredictionError) -> None:
        """
        Record a prediction error for analysis.

        Args:
            error: The prediction error to record
        """
        self._errors[error.error_id] = error
        self._domain_error_counts[error.domain] += 1

    def record_chain_error(
        self,
        chain: ReasoningChain,
        expected_output: Optional[Any] = None,
        actual_output: Optional[Any] = None,
    ) -> PredictionError:
        """
        Record an error directly from a ReasoningChain.

        Convenience method that creates a PredictionError internally from
        the reasoning chain data. This allows seamless integration with
        the ReasoningObserver which produces ReasoningChain objects.

        Args:
            chain: The ReasoningChain containing the failed reasoning
            expected_output: The expected output (uses chain.ground_truth if None)
            actual_output: The actual output (uses chain.prediction if None)

        Returns:
            The created PredictionError for further analysis
        """
        error = create_prediction_error_from_chain(
            chain=chain,
            expected_output=expected_output,
            actual_output=actual_output,
        )
        self.record_error(error)
        return error

    def classify_error(self, error: PredictionError) -> ErrorCategory:
        """
        Classify an error by type.

        Uses heuristics based on the reasoning chain and error context
        to determine the error category.

        Args:
            error: The error to classify

        Returns:
            ErrorCategory for this error
        """
        # If already classified, return cached result
        if error.error_id in self._classifications:
            return self._classifications[error.error_id]

        category = self._determine_category(error)
        self._classifications[error.error_id] = category
        self._category_counts[category] += 1

        return category

    def _determine_category(self, error: PredictionError) -> ErrorCategory:
        """Determine the error category using heuristics."""
        chain = error.reasoning_chain

        if not chain:
            return ErrorCategory.UNKNOWN

        # Analyze failed steps
        failed_steps = chain.failed_steps

        if not failed_steps:
            # Prediction was wrong but all steps succeeded
            # Likely a rule coverage gap or edge case
            if not error.contributing_rules:
                return ErrorCategory.RULE_COVERAGE_GAP
            return ErrorCategory.EDGE_CASE

        # Check for translation errors
        translation_failures = [
            s for s in failed_steps if s.step_type == ReasoningStepType.TRANSLATION
        ]
        if translation_failures:
            return ErrorCategory.TRANSLATION_ERROR

        # Check for validation failures
        validation_failures = [
            s for s in failed_steps if s.step_type == ReasoningStepType.VALIDATION
        ]
        if validation_failures:
            return ErrorCategory.VALIDATION_FAILURE

        # Check for inference errors
        inference_failures = [
            s for s in failed_steps if s.step_type == ReasoningStepType.INFERENCE
        ]
        if inference_failures:
            return ErrorCategory.INFERENCE_ERROR

        # Check for rule conflicts in metadata
        for step in failed_steps:
            if step.metadata.get("conflict_detected"):
                return ErrorCategory.RULE_CONFLICT
            if step.metadata.get("predicate_mismatch"):
                return ErrorCategory.PREDICATE_MISMATCH

        # Check for domain boundary issues
        if chain.metadata.get("cross_domain") or chain.metadata.get("domain_mismatch"):
            return ErrorCategory.DOMAIN_BOUNDARY

        # Check confidence threshold
        if chain.metadata.get("low_confidence"):
            return ErrorCategory.CONFIDENCE_THRESHOLD

        return ErrorCategory.UNKNOWN

    def identify_root_cause(self, error: PredictionError) -> RootCauseAnalysis:
        """
        Identify root cause of a prediction error.

        Analyzes the reasoning chain and error context to determine
        the most likely root causes.

        Args:
            error: The error to analyze

        Returns:
            RootCauseAnalysis with identified causes
        """
        # Get error category first
        category = self.classify_error(error)

        # Identify potential causes based on category
        primary_cause = self._identify_primary_cause(error, category)
        secondary_causes = self._identify_secondary_causes(error, category)

        # Calculate overall confidence
        analysis_confidence = primary_cause.confidence
        if secondary_causes:
            # Lower confidence if there are multiple plausible causes
            analysis_confidence *= 0.8

        analysis = RootCauseAnalysis(
            analysis_id=f"rca_{uuid.uuid4().hex[:8]}",
            error_id=error.error_id,
            primary_cause=primary_cause,
            secondary_causes=secondary_causes,
            analysis_confidence=analysis_confidence,
            analysis_method="heuristic",
        )

        self._root_cause_analyses[error.error_id] = analysis
        return analysis

    def _identify_primary_cause(
        self, error: PredictionError, category: ErrorCategory
    ) -> RootCause:
        """Identify the primary root cause."""
        # Map categories to likely root causes
        cause_mapping = {
            ErrorCategory.RULE_COVERAGE_GAP: (
                RootCauseType.MISSING_RULE,
                "No applicable rule found for this case type",
                ["Add rules covering this case pattern"],
            ),
            ErrorCategory.RULE_CONFLICT: (
                RootCauseType.LOGICAL_CONTRADICTION,
                "Multiple rules produced contradictory conclusions",
                ["Review conflicting rules", "Add priority or specificity"],
            ),
            ErrorCategory.TRANSLATION_ERROR: (
                RootCauseType.PROMPT_WEAKNESS,
                "Translation between NL and ASP failed",
                ["Improve translation prompts", "Add examples"],
            ),
            ErrorCategory.PREDICATE_MISMATCH: (
                RootCauseType.INCORRECT_RULE,
                "Predicates were mapped incorrectly",
                ["Review predicate mapping", "Standardize predicate names"],
            ),
            ErrorCategory.EDGE_CASE: (
                RootCauseType.INSUFFICIENT_CONTEXT,
                "Case has unusual characteristics not covered by rules",
                ["Add edge case handling", "Expand rule coverage"],
            ),
            ErrorCategory.DOMAIN_BOUNDARY: (
                RootCauseType.DOMAIN_MISMATCH,
                "Case crosses domain boundaries",
                ["Clarify domain boundaries", "Add cross-domain rules"],
            ),
            ErrorCategory.VALIDATION_FAILURE: (
                RootCauseType.DATA_QUALITY,
                "Validation step rejected the result",
                ["Review validation criteria", "Improve data quality"],
            ),
            ErrorCategory.INFERENCE_ERROR: (
                RootCauseType.LOGICAL_CONTRADICTION,
                "ASP solver encountered an error",
                ["Check for unsatisfiable constraints", "Review rule logic"],
            ),
            ErrorCategory.CONFIDENCE_THRESHOLD: (
                RootCauseType.MODEL_LIMITATION,
                "Confidence was below required threshold",
                ["Improve prompts for higher confidence", "Adjust threshold"],
            ),
            ErrorCategory.UNKNOWN: (
                RootCauseType.AMBIGUOUS_INPUT,
                "Unable to determine specific cause",
                ["Manual review required", "Add logging for diagnosis"],
            ),
        }

        cause_type, description, hints = cause_mapping.get(
            category,
            (RootCauseType.AMBIGUOUS_INPUT, "Unknown cause", ["Investigate further"]),
        )

        # Gather evidence from the reasoning chain
        evidence = self._gather_evidence(error, category)

        # Determine affected step types
        affected_steps = []
        if error.reasoning_chain:
            for step in error.reasoning_chain.failed_steps:
                if step.step_type not in affected_steps:
                    affected_steps.append(step.step_type)

        # Calculate confidence based on evidence quality
        confidence = 0.7 if evidence else 0.5

        return RootCause(
            cause_id=f"cause_{uuid.uuid4().hex[:8]}",
            cause_type=cause_type,
            description=description,
            confidence=confidence,
            evidence=evidence,
            affected_step_types=affected_steps,
            remediation_hints=hints,
        )

    def _identify_secondary_causes(
        self, error: PredictionError, category: ErrorCategory
    ) -> List[RootCause]:
        """Identify secondary contributing causes."""
        secondary = []

        chain = error.reasoning_chain
        if not chain:
            return secondary

        # Check for data quality issues
        if chain.metadata.get("input_quality_low"):
            secondary.append(
                RootCause(
                    cause_id=f"cause_{uuid.uuid4().hex[:8]}",
                    cause_type=RootCauseType.DATA_QUALITY,
                    description="Input data quality was flagged as low",
                    confidence=0.5,
                    evidence=["Input quality flag was set"],
                    remediation_hints=["Improve input validation", "Clean data"],
                )
            )

        # Check for insufficient context
        if len(chain.steps) < 3:
            secondary.append(
                RootCause(
                    cause_id=f"cause_{uuid.uuid4().hex[:8]}",
                    cause_type=RootCauseType.INSUFFICIENT_CONTEXT,
                    description="Reasoning chain was unusually short",
                    confidence=0.4,
                    evidence=[f"Only {len(chain.steps)} steps in chain"],
                    remediation_hints=["Ensure complete reasoning pipeline"],
                )
            )

        # Check for multiple low-confidence steps
        low_conf_steps = [s for s in chain.steps if s.confidence < 0.5]
        if len(low_conf_steps) > 1:
            secondary.append(
                RootCause(
                    cause_id=f"cause_{uuid.uuid4().hex[:8]}",
                    cause_type=RootCauseType.MODEL_LIMITATION,
                    description="Multiple steps had low confidence",
                    confidence=0.45,
                    evidence=[f"{len(low_conf_steps)} steps with confidence < 0.5"],
                    remediation_hints=["Improve prompts", "Add examples"],
                )
            )

        return secondary

    def _gather_evidence(
        self, error: PredictionError, category: ErrorCategory
    ) -> List[str]:
        """Gather evidence supporting the root cause analysis."""
        evidence = []

        if error.reasoning_chain:
            chain = error.reasoning_chain

            # Add step failure evidence
            for step in chain.failed_steps:
                if step.error_message:
                    evidence.append(f"Step {step.step_id} failed: {step.error_message}")
                else:
                    evidence.append(
                        f"Step {step.step_id} ({step.step_type.value}) failed"
                    )

            # Add prediction mismatch evidence
            if chain.prediction and chain.ground_truth:
                evidence.append(
                    f"Predicted '{chain.prediction}' but actual was '{chain.ground_truth}'"
                )

        # Add rule evidence
        if error.contributing_rules:
            evidence.append(
                f"Contributing rules: {', '.join(error.contributing_rules)}"
            )
        else:
            evidence.append("No contributing rules identified")

        return evidence

    def find_failure_patterns(self, min_occurrences: int = 3) -> List[FailurePattern]:
        """
        Identify recurring failure patterns.

        Analyzes all recorded errors to find common patterns
        that can be addressed systematically.

        Args:
            min_occurrences: Minimum errors to constitute a pattern

        Returns:
            List of identified failure patterns
        """
        # Group errors by category
        category_groups: Dict[ErrorCategory, List[PredictionError]] = defaultdict(list)
        for error in self._errors.values():
            category = self.classify_error(error)
            category_groups[category].append(error)

        patterns = []

        for category, errors in category_groups.items():
            if len(errors) < min_occurrences:
                continue

            # Find common characteristics
            common_chars = self._find_common_characteristics(errors)

            # Get domains affected
            domains = list(set(e.domain for e in errors))

            # Identify root causes for the pattern
            root_causes = []
            for error in errors[:5]:  # Sample first 5
                analysis = self.identify_root_cause(error)
                if analysis.primary_cause not in root_causes:
                    root_causes.append(analysis.primary_cause)

            # Determine severity based on count and category
            severity = self._determine_severity(len(errors), category)

            pattern = FailurePattern(
                pattern_id=f"pattern_{uuid.uuid4().hex[:8]}",
                name=f"{category.value.replace('_', ' ').title()} Pattern",
                description=f"Recurring {category.value} errors affecting {len(domains)} domains",
                error_category=category,
                error_count=len(errors),
                affected_domains=domains,
                common_characteristics=common_chars,
                root_causes=root_causes[:3],  # Top 3 causes
                example_error_ids=[e.error_id for e in errors[:5]],
                first_seen=min(e.timestamp for e in errors),
                last_seen=max(e.timestamp for e in errors),
                severity=severity,
            )
            patterns.append(pattern)
            self._patterns[pattern.pattern_id] = pattern

        return sorted(patterns, key=lambda p: p.error_count, reverse=True)

    def _find_common_characteristics(
        self, errors: List[PredictionError]
    ) -> Dict[str, Any]:
        """Find common characteristics among errors."""
        chars: Dict[str, Any] = {}

        # Common domains
        domains = [e.domain for e in errors]
        domain_counts = defaultdict(int)
        for d in domains:
            domain_counts[d] += 1
        chars["domain_distribution"] = dict(domain_counts)

        # Common step types in failed chains
        step_type_counts: Dict[str, int] = defaultdict(int)
        for error in errors:
            if error.reasoning_chain:
                for step in error.reasoning_chain.failed_steps:
                    step_type_counts[step.step_type.value] += 1
        if step_type_counts:
            chars["common_failed_step_types"] = dict(step_type_counts)

        # Average chain length
        chain_lengths = [
            len(e.reasoning_chain.steps) for e in errors if e.reasoning_chain
        ]
        if chain_lengths:
            chars["avg_chain_length"] = sum(chain_lengths) / len(chain_lengths)

        # Common metadata keys
        metadata_keys: Dict[str, int] = defaultdict(int)
        for error in errors:
            for key in error.metadata.keys():
                metadata_keys[key] += 1
        if metadata_keys:
            chars["common_metadata_keys"] = [
                k for k, v in metadata_keys.items() if v > len(errors) / 2
            ]

        return chars

    def _determine_severity(self, error_count: int, category: ErrorCategory) -> str:
        """Determine pattern severity."""
        # High severity categories
        high_severity_categories = {
            ErrorCategory.RULE_CONFLICT,
            ErrorCategory.INFERENCE_ERROR,
        }

        if category in high_severity_categories:
            if error_count >= 10:
                return "critical"
            return "high"

        if error_count >= 20:
            return "critical"
        elif error_count >= 10:
            return "high"
        elif error_count >= 5:
            return "medium"
        return "low"

    def create_diagnosis(self, error: PredictionError) -> FailureDiagnosis:
        """
        Create a complete failure diagnosis for an error.

        Combines error classification and root cause analysis
        into a comprehensive diagnosis.

        Args:
            error: The error to diagnose

        Returns:
            FailureDiagnosis with complete analysis
        """
        category = self.classify_error(error)
        rca = self.identify_root_cause(error)

        # Find similar failures
        similar = self._find_similar_errors(error)

        # Build explanation
        explanation = self._build_explanation(error, category, rca)

        return FailureDiagnosis(
            diagnosis_id=f"diag_{uuid.uuid4().hex[:8]}",
            chain_id=error.reasoning_chain.chain_id if error.reasoning_chain else "",
            case_id=error.case_id,
            prediction=error.predicted,
            ground_truth=error.actual,
            primary_failure_step=(
                error.reasoning_chain.failed_steps[0].step_id
                if error.reasoning_chain and error.reasoning_chain.failed_steps
                else None
            ),
            failure_type=category.value,
            root_causes=[rca.primary_cause.description]
            + [c.description for c in rca.secondary_causes],
            contributing_factors=error.contributing_rules,
            confidence=rca.analysis_confidence,
            explanation=explanation,
            similar_failures=similar,
        )

    def _find_similar_errors(self, error: PredictionError) -> List[str]:
        """Find errors similar to the given error."""
        similar = []
        target_category = self.classify_error(error)

        for other_id, other in self._errors.items():
            if other_id == error.error_id:
                continue

            other_category = self.classify_error(other)
            if other_category == target_category and other.domain == error.domain:
                similar.append(other_id)
                if len(similar) >= 5:
                    break

        return similar

    def _build_explanation(
        self,
        error: PredictionError,
        category: ErrorCategory,
        rca: RootCauseAnalysis,
    ) -> str:
        """Build a natural language explanation of the failure."""
        parts = []

        parts.append(
            f"This prediction failed due to {category.value.replace('_', ' ')}."
        )
        parts.append(f"The primary cause was: {rca.primary_cause.description}")

        if rca.secondary_causes:
            parts.append(
                f"Contributing factors include: "
                f"{', '.join(c.description for c in rca.secondary_causes[:2])}"
            )

        if rca.primary_cause.remediation_hints:
            parts.append(
                f"Suggested remediation: {rca.primary_cause.remediation_hints[0]}"
            )

        return " ".join(parts)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded errors."""
        return {
            "total_errors": len(self._errors),
            "category_distribution": {
                cat.value: count for cat, count in self._category_counts.items()
            },
            "domain_distribution": dict(self._domain_error_counts),
            "patterns_identified": len(self._patterns),
        }


class RecommendationEngine:
    """
    Generates improvement recommendations based on failure analysis.

    Provides actionable recommendations prioritized by expected impact.
    """

    def __init__(self, analyzer: FailureAnalyzer):
        """
        Initialize the recommendation engine.

        Args:
            analyzer: FailureAnalyzer instance for failure data
        """
        self.analyzer = analyzer
        self._recommendations: Dict[str, Recommendation] = {}
        self._recommendation_effectiveness: Dict[str, float] = {}

    def generate_recommendations(
        self, patterns: List[FailurePattern]
    ) -> List[Recommendation]:
        """
        Generate recommendations based on failure patterns.

        Args:
            patterns: List of failure patterns to address

        Returns:
            List of recommendations
        """
        recommendations = []

        for pattern in patterns:
            recs = self._generate_pattern_recommendations(pattern)
            recommendations.extend(recs)

        # Remove duplicates based on category and target
        seen = set()
        unique_recs = []
        for rec in recommendations:
            key = (rec.category, tuple(rec.target_patterns))
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)
                self._recommendations[rec.recommendation_id] = rec

        return unique_recs

    def _generate_pattern_recommendations(
        self, pattern: FailurePattern
    ) -> List[Recommendation]:
        """Generate recommendations for a specific pattern."""
        recommendations = []

        # Map error categories to recommendation strategies
        category_strategies = {
            ErrorCategory.RULE_COVERAGE_GAP: [
                (
                    RecommendationCategory.RULE_ADDITION,
                    "Add Rules for Uncovered Cases",
                    "Create new rules to cover case types that currently have no applicable rules",
                    0.25,
                ),
                (
                    RecommendationCategory.DATA_AUGMENTATION,
                    "Expand Training Data",
                    "Add more examples of this case type to improve rule generation",
                    0.15,
                ),
            ],
            ErrorCategory.RULE_CONFLICT: [
                (
                    RecommendationCategory.RULE_MODIFICATION,
                    "Resolve Rule Conflicts",
                    "Review and modify conflicting rules to establish clear priority",
                    0.30,
                ),
                (
                    RecommendationCategory.VALIDATION_ENHANCEMENT,
                    "Add Conflict Detection",
                    "Implement early detection of rule conflicts during validation",
                    0.20,
                ),
            ],
            ErrorCategory.TRANSLATION_ERROR: [
                (
                    RecommendationCategory.PROMPT_IMPROVEMENT,
                    "Improve Translation Prompts",
                    "Refine prompts for NL↔ASP translation with more examples",
                    0.20,
                ),
                (
                    RecommendationCategory.VALIDATION_ENHANCEMENT,
                    "Add Translation Validation",
                    "Implement roundtrip validation for translations",
                    0.15,
                ),
            ],
            ErrorCategory.PREDICATE_MISMATCH: [
                (
                    RecommendationCategory.RULE_MODIFICATION,
                    "Standardize Predicates",
                    "Create a standardized predicate vocabulary and mapping",
                    0.18,
                ),
            ],
            ErrorCategory.EDGE_CASE: [
                (
                    RecommendationCategory.RULE_ADDITION,
                    "Add Edge Case Handling",
                    "Create specific rules for identified edge cases",
                    0.15,
                ),
                (
                    RecommendationCategory.DOMAIN_EXPANSION,
                    "Expand Domain Coverage",
                    "Extend domain definitions to cover edge cases",
                    0.12,
                ),
            ],
            ErrorCategory.DOMAIN_BOUNDARY: [
                (
                    RecommendationCategory.DOMAIN_EXPANSION,
                    "Clarify Domain Boundaries",
                    "Define clearer boundaries between domains",
                    0.15,
                ),
                (
                    RecommendationCategory.RULE_ADDITION,
                    "Add Cross-Domain Rules",
                    "Create rules that handle cross-domain scenarios",
                    0.18,
                ),
            ],
            ErrorCategory.VALIDATION_FAILURE: [
                (
                    RecommendationCategory.VALIDATION_ENHANCEMENT,
                    "Review Validation Criteria",
                    "Adjust validation thresholds and criteria",
                    0.12,
                ),
            ],
            ErrorCategory.INFERENCE_ERROR: [
                (
                    RecommendationCategory.RULE_MODIFICATION,
                    "Fix Logical Issues",
                    "Review rules for logical contradictions",
                    0.25,
                ),
            ],
            ErrorCategory.CONFIDENCE_THRESHOLD: [
                (
                    RecommendationCategory.PROMPT_IMPROVEMENT,
                    "Improve Confidence Scores",
                    "Refine prompts to achieve higher confidence outputs",
                    0.15,
                ),
                (
                    RecommendationCategory.STRATEGY_ADJUSTMENT,
                    "Adjust Confidence Thresholds",
                    "Review and adjust confidence threshold settings",
                    0.10,
                ),
            ],
        }

        strategies = category_strategies.get(
            pattern.error_category,
            [
                (
                    RecommendationCategory.STRATEGY_ADJUSTMENT,
                    "Investigate Unknown Errors",
                    "Manual investigation required for unclassified errors",
                    0.05,
                )
            ],
        )

        for category, title, description, impact in strategies:
            # Adjust impact based on pattern severity
            if pattern.severity == "critical":
                impact *= 1.5
            elif pattern.severity == "high":
                impact *= 1.2

            # Determine priority based on error count and severity
            priority = self._determine_priority(pattern)

            # Generate implementation steps
            steps = self._generate_implementation_steps(category, pattern)

            # Generate success criteria
            criteria = self._generate_success_criteria(pattern)

            rec = Recommendation(
                recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
                category=category,
                title=title,
                description=description,
                priority=priority,
                expected_impact=min(impact, 0.50),  # Cap at 50%
                target_patterns=[pattern.pattern_id],
                target_domains=pattern.affected_domains,
                implementation_steps=steps,
                estimated_effort=self._estimate_effort(category),
                success_criteria=criteria,
            )
            recommendations.append(rec)

        return recommendations

    def _determine_priority(self, pattern: FailurePattern) -> ImprovementPriority:
        """Determine recommendation priority based on pattern."""
        if pattern.severity == "critical":
            return ImprovementPriority.CRITICAL
        elif pattern.severity == "high" or pattern.error_count >= 15:
            return ImprovementPriority.HIGH
        elif pattern.error_count >= 8:
            return ImprovementPriority.MEDIUM
        return ImprovementPriority.LOW

    def _generate_implementation_steps(
        self, category: RecommendationCategory, pattern: FailurePattern
    ) -> List[str]:
        """Generate implementation steps for a recommendation."""
        base_steps = {
            RecommendationCategory.RULE_ADDITION: [
                "Analyze error examples to identify missing rule patterns",
                "Draft new rules covering the identified gaps",
                "Validate new rules against test cases",
                "Add rules to the rule base with appropriate priority",
                "Monitor performance on similar cases",
            ],
            RecommendationCategory.RULE_MODIFICATION: [
                "Identify rules involved in the failures",
                "Analyze conflicts or incorrect behavior",
                "Modify rules to address issues",
                "Test modified rules for regression",
                "Deploy and monitor changes",
            ],
            RecommendationCategory.PROMPT_IMPROVEMENT: [
                "Review current prompts for the failing step",
                "Identify weakness areas in prompts",
                "Create improved prompt variations",
                "A/B test new prompts against baseline",
                "Deploy winning prompt variant",
            ],
            RecommendationCategory.VALIDATION_ENHANCEMENT: [
                "Review current validation criteria",
                "Identify gaps in validation coverage",
                "Implement additional validation checks",
                "Test validation with known failure cases",
                "Enable enhanced validation in production",
            ],
            RecommendationCategory.DOMAIN_EXPANSION: [
                "Analyze cases at domain boundaries",
                "Define clearer domain criteria",
                "Expand domain coverage where needed",
                "Update domain classification logic",
                "Validate with cross-domain test cases",
            ],
            RecommendationCategory.DATA_AUGMENTATION: [
                "Identify data gaps from failure analysis",
                "Collect or generate additional examples",
                "Validate data quality and labels",
                "Retrain affected components",
                "Evaluate improvement on held-out set",
            ],
            RecommendationCategory.STRATEGY_ADJUSTMENT: [
                "Review current strategy parameters",
                "Analyze strategy performance on failures",
                "Adjust strategy settings",
                "Test adjusted strategy",
                "Monitor for unintended effects",
            ],
        }

        return base_steps.get(category, ["Investigate and implement fix"])

    def _generate_success_criteria(self, pattern: FailurePattern) -> List[str]:
        """Generate success criteria for addressing a pattern."""
        return [
            f"Reduce {pattern.error_category.value} errors by at least 50%",
            (
                f"No new errors of this type in {pattern.affected_domains[0]} domain"
                if pattern.affected_domains
                else "No new errors of this type"
            ),
            "All existing test cases continue to pass",
            "Performance metrics remain stable or improve",
        ]

    def _estimate_effort(self, category: RecommendationCategory) -> str:
        """Estimate effort for implementing a recommendation."""
        effort_map = {
            RecommendationCategory.RULE_ADDITION: "medium",
            RecommendationCategory.RULE_MODIFICATION: "medium",
            RecommendationCategory.PROMPT_IMPROVEMENT: "low",
            RecommendationCategory.VALIDATION_ENHANCEMENT: "medium",
            RecommendationCategory.DOMAIN_EXPANSION: "high",
            RecommendationCategory.DATA_AUGMENTATION: "high",
            RecommendationCategory.STRATEGY_ADJUSTMENT: "low",
        }
        return effort_map.get(category, "medium")

    def prioritize_recommendations(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """
        Prioritize recommendations by expected impact.

        Args:
            recommendations: List of recommendations to prioritize

        Returns:
            Sorted list of recommendations
        """
        # Sort by priority (enum order) then by expected impact
        priority_order = {
            ImprovementPriority.CRITICAL: 0,
            ImprovementPriority.HIGH: 1,
            ImprovementPriority.MEDIUM: 2,
            ImprovementPriority.LOW: 3,
        }

        return sorted(
            recommendations,
            key=lambda r: (priority_order[r.priority], -r.expected_impact),
        )

    def track_recommendation_effectiveness(
        self, recommendation_id: str, actual_improvement: float
    ) -> None:
        """
        Track the effectiveness of an implemented recommendation.

        Args:
            recommendation_id: ID of the recommendation
            actual_improvement: Actual improvement percentage achieved
        """
        self._recommendation_effectiveness[recommendation_id] = actual_improvement

    def get_effectiveness_report(self) -> Dict[str, Any]:
        """Get report on recommendation effectiveness."""
        if not self._recommendation_effectiveness:
            return {"message": "No recommendations have been tracked yet"}

        improvements = list(self._recommendation_effectiveness.values())
        return {
            "total_tracked": len(improvements),
            "average_improvement": sum(improvements) / len(improvements),
            "max_improvement": max(improvements),
            "min_improvement": min(improvements),
            "recommendations": dict(self._recommendation_effectiveness),
        }

    def generate_report(
        self,
        errors: Optional[List[PredictionError]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> FailureAnalysisReport:
        """
        Generate a comprehensive failure analysis report.

        Args:
            errors: Optional list of errors to analyze (uses all if None)
            start_date: Optional start of analysis period
            end_date: Optional end of analysis period

        Returns:
            FailureAnalysisReport with complete analysis
        """
        # Get errors
        if errors is None:
            errors = list(self.analyzer._errors.values())

        # Filter by date if specified
        if start_date:
            errors = [e for e in errors if e.timestamp >= start_date]
        if end_date:
            errors = [e for e in errors if e.timestamp <= end_date]

        if not errors:
            return FailureAnalysisReport(
                report_id=f"report_{uuid.uuid4().hex[:8]}",
                analysis_period_start=start_date or datetime.now(),
                analysis_period_end=end_date or datetime.now(),
                total_errors_analyzed=0,
                error_category_distribution={},
                patterns_identified=[],
                top_root_causes=[],
                recommendations=[],
                overall_error_rate=0.0,
                domains_analyzed=[],
            )

        # Record errors if not already recorded
        for error in errors:
            if error.error_id not in self.analyzer._errors:
                self.analyzer.record_error(error)

        # Get category distribution
        category_dist: Dict[str, int] = defaultdict(int)
        for error in errors:
            cat = self.analyzer.classify_error(error)
            category_dist[cat.value] += 1

        # Find patterns
        patterns = self.analyzer.find_failure_patterns()

        # Get top root causes
        root_causes: List[RootCause] = []
        for error in errors[:20]:  # Sample first 20
            rca = self.analyzer.identify_root_cause(error)
            root_causes.append(rca.primary_cause)

        # Count root cause types
        cause_counts: Dict[str, int] = defaultdict(int)
        for cause in root_causes:
            cause_counts[cause.cause_type.value] += 1

        # Get unique causes sorted by frequency
        seen_types = set()
        unique_causes = []
        for cause in sorted(
            root_causes, key=lambda c: cause_counts[c.cause_type.value], reverse=True
        ):
            if cause.cause_type not in seen_types:
                seen_types.add(cause.cause_type)
                unique_causes.append(cause)

        # Generate recommendations
        recommendations = self.generate_recommendations(patterns)
        prioritized = self.prioritize_recommendations(recommendations)

        # Get domains
        domains = list(set(e.domain for e in errors))

        return FailureAnalysisReport(
            report_id=f"report_{uuid.uuid4().hex[:8]}",
            analysis_period_start=min(e.timestamp for e in errors),
            analysis_period_end=max(e.timestamp for e in errors),
            total_errors_analyzed=len(errors),
            error_category_distribution=dict(category_dist),
            patterns_identified=patterns,
            top_root_causes=unique_causes[:5],
            recommendations=prioritized[:10],
            overall_error_rate=1.0,  # All analyzed errors are failures
            domains_analyzed=domains,
        )


# Factory functions and integration helpers


def create_prediction_error_from_chain(
    chain: ReasoningChain,
    expected_output: Optional[Any] = None,
    actual_output: Optional[Any] = None,
    error_id: Optional[str] = None,
    contributing_rules: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PredictionError:
    """
    Create a PredictionError from a failed ReasoningChain.

    Extracts relevant information from the chain to populate
    the PredictionError fields. This enables seamless integration
    between ReasoningObserver and FailureAnalyzer.

    Args:
        chain: The ReasoningChain containing the failed reasoning
        expected_output: The expected output (uses chain.ground_truth if None)
        actual_output: The actual output (uses chain.prediction if None)
        error_id: Optional custom error ID (generated if None)
        contributing_rules: Optional list of rule IDs that contributed
        metadata: Optional additional metadata

    Returns:
        PredictionError populated from the chain data

    Example:
        >>> chain = observer.observe_reasoning_chain(
        ...     case_id="case_001",
        ...     domain="contracts",
        ...     steps=steps,
        ...     prediction="enforceable",
        ...     ground_truth="unenforceable"
        ... )
        >>> error = create_prediction_error_from_chain(chain)
        >>> analyzer.record_error(error)
    """
    # Use chain values as defaults
    predicted = (
        str(actual_output) if actual_output is not None else (chain.prediction or "")
    )
    actual = (
        str(expected_output)
        if expected_output is not None
        else (chain.ground_truth or "")
    )

    # Generate error_id if not provided
    if error_id is None:
        error_id = f"err_{chain.chain_id}_{uuid.uuid4().hex[:8]}"

    # Extract contributing rules from chain metadata if not provided
    if contributing_rules is None:
        contributing_rules = chain.metadata.get("contributing_rules", [])
        # Also check step metadata for rule references
        if not contributing_rules:
            for step in chain.steps:
                step_rules = step.metadata.get("rules_applied", [])
                contributing_rules.extend(step_rules)
            contributing_rules = list(set(contributing_rules))  # Remove duplicates

    # Build metadata combining chain metadata with any additional
    error_metadata = dict(chain.metadata)
    if metadata:
        error_metadata.update(metadata)

    # Add chain-derived metadata
    error_metadata["chain_success_rate"] = chain.success_rate
    error_metadata["chain_total_duration_ms"] = chain.total_duration_ms
    if chain.failed_steps:
        error_metadata["failed_step_types"] = [
            step.step_type.value for step in chain.failed_steps
        ]

    return PredictionError(
        error_id=error_id,
        case_id=chain.case_id,
        domain=chain.domain,
        predicted=predicted,
        actual=actual,
        reasoning_chain=chain,
        contributing_rules=contributing_rules,
        timestamp=chain.completed_at or datetime.now(),
        metadata=error_metadata,
    )


def extract_failure_patterns(
    errors: List[PredictionError],
    min_occurrences: int = 3,
    analyzer: Optional["FailureAnalyzer"] = None,
) -> List[FailurePattern]:
    """
    Convert a list of errors to failure patterns for the recommendation engine.

    This utility function bridges the gap between raw PredictionError objects
    and the FailurePattern objects expected by RecommendationEngine.generate_recommendations().

    Args:
        errors: List of PredictionError objects to analyze
        min_occurrences: Minimum errors to constitute a pattern
        analyzer: Optional FailureAnalyzer to use (creates new if None)

    Returns:
        List of FailurePattern objects ready for recommendation generation

    Example:
        >>> errors = [create_prediction_error_from_chain(chain) for chain in failed_chains]
        >>> patterns = extract_failure_patterns(errors)
        >>> recommendations = engine.generate_recommendations(patterns)
    """
    if analyzer is None:
        analyzer = FailureAnalyzer()

    # Record all errors in the analyzer
    for error in errors:
        if error.error_id not in analyzer._errors:
            analyzer.record_error(error)

    # Find and return patterns
    return analyzer.find_failure_patterns(min_occurrences=min_occurrences)


def create_failure_analyzer() -> FailureAnalyzer:
    """Create a FailureAnalyzer instance."""
    return FailureAnalyzer()


def create_recommendation_engine(
    analyzer: Optional[FailureAnalyzer] = None,
) -> RecommendationEngine:
    """
    Create a RecommendationEngine instance.

    Args:
        analyzer: Optional FailureAnalyzer (creates new if None)

    Returns:
        Configured RecommendationEngine
    """
    if analyzer is None:
        analyzer = create_failure_analyzer()
    return RecommendationEngine(analyzer)
