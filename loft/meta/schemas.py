"""
Data classes for meta-reasoning module.

Provides schemas for reasoning patterns, observations, bottlenecks,
and improvement recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ReasoningStepType(Enum):
    """Types of reasoning steps in the pipeline."""

    TRANSLATION = "translation"  # NL to ASP or vice versa
    RULE_APPLICATION = "rule_application"  # Applying symbolic rules
    RULE_GENERATION = "rule_generation"  # Generating new rules
    VALIDATION = "validation"  # Validating rules or predictions
    GROUNDING = "grounding"  # Grounding predicates
    INFERENCE = "inference"  # ASP solver inference
    CONSENSUS = "consensus"  # Multi-LLM consensus
    DIALECTICAL = "dialectical"  # Dialectical reasoning cycle


class PatternType(Enum):
    """Types of reasoning patterns that can be identified."""

    SUCCESS = "success"  # Pattern associated with successful reasoning
    FAILURE = "failure"  # Pattern associated with failed reasoning
    BOTTLENECK = "bottleneck"  # Performance bottleneck pattern
    RECURRING = "recurring"  # Frequently occurring pattern
    DOMAIN_SPECIFIC = "domain_specific"  # Pattern specific to a domain


class ImprovementType(Enum):
    """Types of improvements that can be suggested."""

    PROMPT_REFINEMENT = "prompt_refinement"
    RULE_MODIFICATION = "rule_modification"
    STRATEGY_CHANGE = "strategy_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    VALIDATION_THRESHOLD = "validation_threshold"


class ImprovementPriority(Enum):
    """Priority levels for improvements."""

    CRITICAL = "critical"  # Must address immediately
    HIGH = "high"  # Should address soon
    MEDIUM = "medium"  # Address when convenient
    LOW = "low"  # Nice to have


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    step_id: str
    step_type: ReasoningStepType
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    started_at: datetime
    completed_at: datetime
    success: bool
    confidence: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Calculate step duration in milliseconds."""
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "confidence": self.confidence,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            step_type=ReasoningStepType(data["step_type"]),
            description=data["description"],
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            success=data["success"],
            confidence=data.get("confidence", 0.0),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReasoningChain:
    """A complete reasoning chain for a case."""

    chain_id: str
    case_id: str
    domain: str
    steps: List[ReasoningStep]
    prediction: Optional[str] = None
    ground_truth: Optional[str] = None
    overall_success: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        """Calculate total chain duration in milliseconds."""
        if not self.steps:
            return 0.0
        return sum(step.duration_ms for step in self.steps)

    @property
    def success_rate(self) -> float:
        """Calculate success rate of steps."""
        if not self.steps:
            return 0.0
        successful = sum(1 for step in self.steps if step.success)
        return successful / len(self.steps)

    @property
    def failed_steps(self) -> List[ReasoningStep]:
        """Get list of failed steps."""
        return [step for step in self.steps if not step.success]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "case_id": self.case_id,
            "domain": self.domain,
            "steps": [step.to_dict() for step in self.steps],
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "overall_success": self.overall_success,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "total_duration_ms": self.total_duration_ms,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningPattern:
    """A pattern identified in reasoning processes."""

    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str
    frequency: int
    associated_step_types: List[ReasoningStepType]
    success_correlation: float  # -1 to 1, how correlated with success
    domains: List[str] = field(default_factory=list)
    example_chain_ids: List[str] = field(default_factory=list)
    characteristics: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "frequency": self.frequency,
            "associated_step_types": [st.value for st in self.associated_step_types],
            "success_correlation": self.success_correlation,
            "domains": self.domains,
            "example_chain_ids": self.example_chain_ids,
            "characteristics": self.characteristics,
            "discovered_at": self.discovered_at.isoformat(),
        }


@dataclass
class Bottleneck:
    """A performance bottleneck identified in reasoning."""

    bottleneck_id: str
    step_type: ReasoningStepType
    description: str
    avg_duration_ms: float
    max_duration_ms: float
    occurrence_count: int
    percentage_of_total_time: float
    affected_domains: List[str] = field(default_factory=list)
    potential_causes: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bottleneck_id": self.bottleneck_id,
            "step_type": self.step_type.value,
            "description": self.description,
            "avg_duration_ms": self.avg_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "occurrence_count": self.occurrence_count,
            "percentage_of_total_time": self.percentage_of_total_time,
            "affected_domains": self.affected_domains,
            "potential_causes": self.potential_causes,
            "severity": self.severity,
        }


@dataclass
class BottleneckReport:
    """Report of identified bottlenecks."""

    report_id: str
    generated_at: datetime
    total_chains_analyzed: int
    total_steps_analyzed: int
    bottlenecks: List[Bottleneck]
    top_time_consumers: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "total_chains_analyzed": self.total_chains_analyzed,
            "total_steps_analyzed": self.total_steps_analyzed,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "top_time_consumers": self.top_time_consumers,
            "recommendations": self.recommendations,
        }


@dataclass
class FailureDiagnosis:
    """Diagnosis of a reasoning failure."""

    diagnosis_id: str
    chain_id: str
    case_id: str
    prediction: str
    ground_truth: str
    primary_failure_step: Optional[str] = None
    primary_failure_step_type: Optional[ReasoningStepType] = None
    failure_type: str = "unknown"
    root_causes: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    similar_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "diagnosis_id": self.diagnosis_id,
            "chain_id": self.chain_id,
            "case_id": self.case_id,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "primary_failure_step": self.primary_failure_step,
            "primary_failure_step_type": (
                self.primary_failure_step_type.value if self.primary_failure_step_type else None
            ),
            "failure_type": self.failure_type,
            "root_causes": self.root_causes,
            "contributing_factors": self.contributing_factors,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "similar_failures": self.similar_failures,
        }


@dataclass
class Improvement:
    """A suggested improvement to the reasoning system."""

    improvement_id: str
    improvement_type: ImprovementType
    priority: ImprovementPriority
    title: str
    description: str
    expected_impact: str
    target_component: str
    estimated_effort: str = "medium"  # low, medium, high
    supporting_evidence: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    related_diagnoses: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "improvement_id": self.improvement_id,
            "improvement_type": self.improvement_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "expected_impact": self.expected_impact,
            "target_component": self.target_component,
            "estimated_effort": self.estimated_effort,
            "supporting_evidence": self.supporting_evidence,
            "related_patterns": self.related_patterns,
            "related_diagnoses": self.related_diagnoses,
            "implementation_steps": self.implementation_steps,
        }


@dataclass
class ObservationSummary:
    """Summary of observations from the reasoning observer."""

    summary_id: str
    generated_at: datetime
    observation_period_start: datetime
    observation_period_end: datetime
    total_chains_observed: int
    total_steps_observed: int
    success_rate: float
    avg_chain_duration_ms: float
    patterns_identified: int
    bottlenecks_identified: int
    domains_observed: List[str] = field(default_factory=list)
    step_type_distribution: Dict[str, int] = field(default_factory=dict)
    domain_success_rates: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_id": self.summary_id,
            "generated_at": self.generated_at.isoformat(),
            "observation_period_start": self.observation_period_start.isoformat(),
            "observation_period_end": self.observation_period_end.isoformat(),
            "total_chains_observed": self.total_chains_observed,
            "total_steps_observed": self.total_steps_observed,
            "success_rate": self.success_rate,
            "avg_chain_duration_ms": self.avg_chain_duration_ms,
            "patterns_identified": self.patterns_identified,
            "bottlenecks_identified": self.bottlenecks_identified,
            "domains_observed": self.domains_observed,
            "step_type_distribution": self.step_type_distribution,
            "domain_success_rates": self.domain_success_rates,
        }
