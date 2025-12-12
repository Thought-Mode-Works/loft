"""
Data structures for contradiction management (Phase 4.4).

Designed to work with existing critique system and evolution tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from loft.symbolic.stratification import StratificationLevel


class ContradictionType(str, Enum):
    """Types of contradictions that can occur."""

    LOGICAL = "logical"  # Direct logical conflict (A and not A)
    PRECEDENT = "precedent"  # Conflicting precedent interpretations
    CONTEXTUAL = "contextual"  # Context-dependent incompatibility
    TEMPORAL = "temporal"  # Newer rule conflicts with older
    HIERARCHICAL = "hierarchical"  # Different stratification layers
    SEMANTIC = "semantic"  # Same predicate, different meanings


class ContradictionSeverity(str, Enum):
    """Severity levels for contradictions."""

    LOW = "low"  # Minor inconsistency, unlikely to cause issues
    MEDIUM = "medium"  # Notable conflict, should be addressed
    HIGH = "high"  # Significant contradiction, needs resolution
    CRITICAL = "critical"  # System-breaking conflict, immediate resolution required


class ResolutionStrategy(str, Enum):
    """Strategies for resolving contradictions."""

    STRATIFICATION = "stratification"  # Higher layer wins
    TEMPORAL = "temporal"  # Newer rule supersedes older
    CONTEXTUAL = "contextual"  # Select based on context
    SPECIFICITY = "specificity"  # More specific rule wins
    CONFIDENCE = "confidence"  # Higher confidence wins
    HUMAN_DECISION = "human_decision"  # Escalate to human
    MERGE = "merge"  # Combine both rules with conditions
    DEPRECATE_BOTH = "deprecate_both"  # Mark both as problematic


@dataclass
class ContradictionReport:
    """
    Comprehensive report of a detected contradiction.

    Extends the basic Contradiction from critique_schemas with full tracking
    and resolution capabilities.
    """

    contradiction_id: str
    contradiction_type: ContradictionType

    # Rules involved
    rule_a_id: str
    rule_a_text: str
    rule_b_id: str
    rule_b_text: str

    # Severity and context
    severity: ContradictionSeverity
    detected_in_context: Optional[str] = None
    affects_layers: List[StratificationLevel] = field(default_factory=list)

    # Analysis
    explanation: str = ""
    example_conflict_case: Optional[Dict[str, Any]] = None
    confidence: float = 0.8  # Confidence in contradiction detection

    # Resolution
    suggested_resolution: Optional[ResolutionStrategy] = None
    resolution_explanation: str = ""
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_applied: Optional[ResolutionStrategy] = None
    resolution_notes: str = ""

    # Winning rule (if resolved)
    winning_rule_id: Optional[str] = None
    losing_rule_id: Optional[str] = None

    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    detected_by: str = "contradiction_manager"  # System or human identifier
    related_contradictions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_critical(self) -> bool:
        """Check if contradiction is critical severity."""
        return self.severity == ContradictionSeverity.CRITICAL

    def requires_immediate_resolution(self) -> bool:
        """Check if contradiction needs immediate resolution."""
        return self.severity in [
            ContradictionSeverity.CRITICAL,
            ContradictionSeverity.HIGH,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "contradiction_id": self.contradiction_id,
            "contradiction_type": self.contradiction_type.value,
            "rule_a_id": self.rule_a_id,
            "rule_a_text": self.rule_a_text,
            "rule_b_id": self.rule_b_id,
            "rule_b_text": self.rule_b_text,
            "severity": self.severity.value,
            "detected_in_context": self.detected_in_context,
            "affects_layers": [layer.value for layer in self.affects_layers],
            "explanation": self.explanation,
            "example_conflict_case": self.example_conflict_case,
            "confidence": self.confidence,
            "suggested_resolution": (
                self.suggested_resolution.value if self.suggested_resolution else None
            ),
            "resolution_explanation": self.resolution_explanation,
            "resolved": self.resolved,
            "resolution_timestamp": (
                self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
            ),
            "resolution_applied": (
                self.resolution_applied.value if self.resolution_applied else None
            ),
            "resolution_notes": self.resolution_notes,
            "winning_rule_id": self.winning_rule_id,
            "losing_rule_id": self.losing_rule_id,
            "detected_at": self.detected_at.isoformat(),
            "detected_by": self.detected_by,
            "related_contradictions": self.related_contradictions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContradictionReport":
        """Create from dictionary."""
        return cls(
            contradiction_id=data["contradiction_id"],
            contradiction_type=ContradictionType(data["contradiction_type"]),
            rule_a_id=data["rule_a_id"],
            rule_a_text=data["rule_a_text"],
            rule_b_id=data["rule_b_id"],
            rule_b_text=data["rule_b_text"],
            severity=ContradictionSeverity(data["severity"]),
            detected_in_context=data.get("detected_in_context"),
            affects_layers=[StratificationLevel(layer) for layer in data.get("affects_layers", [])],
            explanation=data.get("explanation", ""),
            example_conflict_case=data.get("example_conflict_case"),
            confidence=data.get("confidence", 0.8),
            suggested_resolution=(
                ResolutionStrategy(data["suggested_resolution"])
                if data.get("suggested_resolution")
                else None
            ),
            resolution_explanation=data.get("resolution_explanation", ""),
            resolved=data.get("resolved", False),
            resolution_timestamp=(
                datetime.fromisoformat(data["resolution_timestamp"])
                if data.get("resolution_timestamp")
                else None
            ),
            resolution_applied=(
                ResolutionStrategy(data["resolution_applied"])
                if data.get("resolution_applied")
                else None
            ),
            resolution_notes=data.get("resolution_notes", ""),
            winning_rule_id=data.get("winning_rule_id"),
            losing_rule_id=data.get("losing_rule_id"),
            detected_at=datetime.fromisoformat(data["detected_at"]),
            detected_by=data.get("detected_by", "contradiction_manager"),
            related_contradictions=data.get("related_contradictions", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RuleInterpretation:
    """
    A specific interpretation of a legal/policy principle.

    Supports tracking multiple competing interpretations of the same principle,
    each applicable in different contexts.
    """

    interpretation_id: str
    principle: str  # e.g., "substantial performance doctrine"
    interpretation_text: str
    asp_rules: List[str] = field(default_factory=list)

    # Context applicability
    applicable_contexts: List[str] = field(default_factory=list)
    exclusion_contexts: List[str] = field(default_factory=list)

    # Evidence and confidence
    supporting_precedents: List[str] = field(default_factory=list)
    supporting_rationale: str = ""
    confidence_score: float = 0.5

    # Competing interpretations
    competing_interpretations: List[str] = field(default_factory=list)
    dominance_conditions: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "interpretation_id": self.interpretation_id,
            "principle": self.principle,
            "interpretation_text": self.interpretation_text,
            "asp_rules": self.asp_rules,
            "applicable_contexts": self.applicable_contexts,
            "exclusion_contexts": self.exclusion_contexts,
            "supporting_precedents": self.supporting_precedents,
            "supporting_rationale": self.supporting_rationale,
            "confidence_score": self.confidence_score,
            "competing_interpretations": self.competing_interpretations,
            "dominance_conditions": self.dominance_conditions,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleInterpretation":
        """Create from dictionary."""
        return cls(
            interpretation_id=data["interpretation_id"],
            principle=data["principle"],
            interpretation_text=data["interpretation_text"],
            asp_rules=data.get("asp_rules", []),
            applicable_contexts=data.get("applicable_contexts", []),
            exclusion_contexts=data.get("exclusion_contexts", []),
            supporting_precedents=data.get("supporting_precedents", []),
            supporting_rationale=data.get("supporting_rationale", ""),
            confidence_score=data.get("confidence_score", 0.5),
            competing_interpretations=data.get("competing_interpretations", []),
            dominance_conditions=data.get("dominance_conditions", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "system"),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ResolutionResult:
    """Result of applying a resolution strategy to a contradiction."""

    contradiction_id: str
    strategy_applied: ResolutionStrategy
    success: bool

    # Resolution outcome
    winning_rule_id: Optional[str] = None
    losing_rule_id: Optional[str] = None
    merged_rule: Optional[str] = None

    # Actions taken
    rules_deprecated: List[str] = field(default_factory=list)
    rules_modified: List[str] = field(default_factory=list)
    new_rules_created: List[str] = field(default_factory=list)

    # Metadata
    resolution_notes: str = ""
    confidence: float = 0.8
    requires_validation: bool = True
    requires_human_review: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextClassification:
    """
    Classification of a case context for rule selection.

    Used to determine which rules/interpretations are applicable given
    the specific facts and circumstances of a case.
    """

    context_id: str
    context_type: str  # e.g., "contract_formation", "breach_remedy"
    confidence: float

    # Context features
    key_features: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    jurisdiction: Optional[str] = None

    # Applicable rules
    applicable_rule_ids: List[str] = field(default_factory=list)
    excluded_rule_ids: List[str] = field(default_factory=list)

    # Applicable interpretations
    applicable_interpretations: List[str] = field(default_factory=list)
    interpretation_preferences: Dict[str, float] = field(
        default_factory=dict
    )  # interpretation_id -> preference score

    # Metadata
    classified_at: datetime = field(default_factory=datetime.now)
    classifier_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "context_type": self.context_type,
            "confidence": self.confidence,
            "key_features": self.key_features,
            "domain": self.domain,
            "jurisdiction": self.jurisdiction,
            "applicable_rule_ids": self.applicable_rule_ids,
            "excluded_rule_ids": self.excluded_rule_ids,
            "applicable_interpretations": self.applicable_interpretations,
            "interpretation_preferences": self.interpretation_preferences,
            "classified_at": self.classified_at.isoformat(),
            "classifier_version": self.classifier_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextClassification":
        """Create from dictionary."""
        return cls(
            context_id=data["context_id"],
            context_type=data["context_type"],
            confidence=data["confidence"],
            key_features=data.get("key_features", {}),
            domain=data.get("domain", "general"),
            jurisdiction=data.get("jurisdiction"),
            applicable_rule_ids=data.get("applicable_rule_ids", []),
            excluded_rule_ids=data.get("excluded_rule_ids", []),
            applicable_interpretations=data.get("applicable_interpretations", []),
            interpretation_preferences=data.get("interpretation_preferences", {}),
            classified_at=datetime.fromisoformat(data["classified_at"]),
            classifier_version=data.get("classifier_version", "1.0.0"),
            metadata=data.get("metadata", {}),
        )
