"""
Data structures for rule evolution tracking (Phase 4.3).

Designed with LinkedASP future in mind - metadata fields align with
MAINTAINABILITY.md ontology for seamless migration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class EvolutionMethod(Enum):
    """How a rule version was created."""

    DIALECTICAL = "dialectical"  # From debate framework
    MANUAL = "manual"  # Human-written
    LLM_DIRECT = "llm_direct"  # Direct LLM generation
    GAP_FILL = "gap_fill"  # Gap filling from test failure
    PRINCIPLE = "principle"  # Generated from legal principle
    HUMAN_REVISION = "human_revision"  # Human edited LLM output
    REFINEMENT = "refinement"  # Iterative refinement


class StratificationLevel(Enum):
    """ASP stratification levels (from ROADMAP.md)."""

    CONSTITUTIONAL = "constitutional"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"


@dataclass
class PerformanceSnapshot:
    """Performance metrics at a point in time."""

    timestamp: datetime
    accuracy: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    test_cases_passed: int
    test_cases_total: int
    success_rate: float  # 0.0-1.0

    # Optional detailed metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Validation metadata
    validation_report_id: Optional[str] = None
    validation_method: Optional[str] = None


@dataclass
class EvolutionContext:
    """Context about how a rule version evolved."""

    evolution_method: EvolutionMethod

    # Dialectical debate context (if applicable)
    dialectical_cycle_id: Optional[str] = None
    debate_round: Optional[int] = None
    thesis_rule: Optional[str] = None
    critique_summary: Optional[str] = None

    # LLM provenance (LinkedASP-aligned)
    source_llm: Optional[str] = None  # e.g., "anthropic/claude-3-5-haiku"
    source_prompt_version: Optional[str] = None
    source_text: Optional[str] = None  # Natural language source

    # Legal sources
    legal_sources: List[str] = field(default_factory=list)
    jurisdiction: Optional[str] = None

    # Modification metadata
    reasoning: Optional[str] = None  # Why this version was created
    critique_addressed: Optional[str] = None  # What issues were fixed
    test_case_triggered: Optional[str] = None  # Gap-fill scenario


@dataclass
class RuleVersion:
    """
    A single version of a rule in its evolution history.

    Metadata fields align with LinkedASP ontology (MAINTAINABILITY.md)
    for future migration to RDF-based tracking.
    """

    # Identity
    rule_id: str  # Unique ID for this version
    rule_family_id: str  # ID shared by all versions
    version: str  # Semantic version (e.g., "1.2.3")

    # Rule content
    asp_rule: str  # The actual ASP code
    predicates_used: List[str] = field(default_factory=list)

    # Versioning
    parent_version: Optional[str] = None  # Parent rule ID
    children_versions: List[str] = field(default_factory=list)

    # Evolution context
    evolution_context: Optional[EvolutionContext] = None

    # Performance
    performance: Optional[PerformanceSnapshot] = None
    improvement_over_parent: float = 0.0  # Delta confidence/accuracy

    # Stratification (LinkedASP-aligned)
    stratification_level: StratificationLevel = StratificationLevel.TACTICAL
    modification_authority: str = "llm_with_validation"  # Who can modify

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    incorporated_at: Optional[datetime] = None  # When added to ASP core
    incorporated_by: str = "autonomous_system"

    # Status
    is_active: bool = True  # Currently in use
    is_deprecated: bool = False
    replaced_by: Optional[str] = None  # Rule ID that replaced this
    deprecation_reason: Optional[str] = None

    # Additional metadata for LinkedASP compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_family_id": self.rule_family_id,
            "version": self.version,
            "asp_rule": self.asp_rule,
            "predicates_used": self.predicates_used,
            "parent_version": self.parent_version,
            "children_versions": self.children_versions,
            "evolution_context": self._context_to_dict(),
            "performance": self._performance_to_dict(),
            "improvement_over_parent": self.improvement_over_parent,
            "stratification_level": self.stratification_level.value,
            "modification_authority": self.modification_authority,
            "created_at": self.created_at.isoformat(),
            "incorporated_at": (self.incorporated_at.isoformat() if self.incorporated_at else None),
            "incorporated_by": self.incorporated_by,
            "is_active": self.is_active,
            "is_deprecated": self.is_deprecated,
            "replaced_by": self.replaced_by,
            "deprecation_reason": self.deprecation_reason,
            "metadata": self.metadata,
        }

    def _context_to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert evolution context to dict."""
        if not self.evolution_context:
            return None

        ctx = self.evolution_context
        return {
            "evolution_method": ctx.evolution_method.value,
            "dialectical_cycle_id": ctx.dialectical_cycle_id,
            "debate_round": ctx.debate_round,
            "thesis_rule": ctx.thesis_rule,
            "critique_summary": ctx.critique_summary,
            "source_llm": ctx.source_llm,
            "source_prompt_version": ctx.source_prompt_version,
            "source_text": ctx.source_text,
            "legal_sources": ctx.legal_sources,
            "jurisdiction": ctx.jurisdiction,
            "reasoning": ctx.reasoning,
            "critique_addressed": ctx.critique_addressed,
            "test_case_triggered": ctx.test_case_triggered,
        }

    def _performance_to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert performance snapshot to dict."""
        if not self.performance:
            return None

        perf = self.performance
        return {
            "timestamp": perf.timestamp.isoformat(),
            "accuracy": perf.accuracy,
            "confidence": perf.confidence,
            "test_cases_passed": perf.test_cases_passed,
            "test_cases_total": perf.test_cases_total,
            "success_rate": perf.success_rate,
            "precision": perf.precision,
            "recall": perf.recall,
            "f1_score": perf.f1_score,
            "validation_report_id": perf.validation_report_id,
            "validation_method": perf.validation_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleVersion":
        """Create from dictionary."""
        # Parse evolution context
        ctx_data = data.get("evolution_context")
        evolution_context = None
        if ctx_data:
            evolution_context = EvolutionContext(
                evolution_method=EvolutionMethod(ctx_data["evolution_method"]),
                dialectical_cycle_id=ctx_data.get("dialectical_cycle_id"),
                debate_round=ctx_data.get("debate_round"),
                thesis_rule=ctx_data.get("thesis_rule"),
                critique_summary=ctx_data.get("critique_summary"),
                source_llm=ctx_data.get("source_llm"),
                source_prompt_version=ctx_data.get("source_prompt_version"),
                source_text=ctx_data.get("source_text"),
                legal_sources=ctx_data.get("legal_sources", []),
                jurisdiction=ctx_data.get("jurisdiction"),
                reasoning=ctx_data.get("reasoning"),
                critique_addressed=ctx_data.get("critique_addressed"),
                test_case_triggered=ctx_data.get("test_case_triggered"),
            )

        # Parse performance
        perf_data = data.get("performance")
        performance = None
        if perf_data:
            performance = PerformanceSnapshot(
                timestamp=datetime.fromisoformat(perf_data["timestamp"]),
                accuracy=perf_data["accuracy"],
                confidence=perf_data["confidence"],
                test_cases_passed=perf_data["test_cases_passed"],
                test_cases_total=perf_data["test_cases_total"],
                success_rate=perf_data["success_rate"],
                precision=perf_data.get("precision"),
                recall=perf_data.get("recall"),
                f1_score=perf_data.get("f1_score"),
                validation_report_id=perf_data.get("validation_report_id"),
                validation_method=perf_data.get("validation_method"),
            )

        return cls(
            rule_id=data["rule_id"],
            rule_family_id=data["rule_family_id"],
            version=data["version"],
            asp_rule=data["asp_rule"],
            predicates_used=data.get("predicates_used", []),
            parent_version=data.get("parent_version"),
            children_versions=data.get("children_versions", []),
            evolution_context=evolution_context,
            performance=performance,
            improvement_over_parent=data.get("improvement_over_parent", 0.0),
            stratification_level=StratificationLevel(data.get("stratification_level", "tactical")),
            modification_authority=data.get("modification_authority", "llm_with_validation"),
            created_at=datetime.fromisoformat(data["created_at"]),
            incorporated_at=(
                datetime.fromisoformat(data["incorporated_at"])
                if data.get("incorporated_at")
                else None
            ),
            incorporated_by=data.get("incorporated_by", "autonomous_system"),
            is_active=data.get("is_active", True),
            is_deprecated=data.get("is_deprecated", False),
            replaced_by=data.get("replaced_by"),
            deprecation_reason=data.get("deprecation_reason"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RuleLineage:
    """
    Complete evolution history of a rule family.

    Tracks all versions from root to current, including branches.
    """

    rule_family_id: str
    root_version: RuleVersion
    all_versions: List[RuleVersion]
    current_version: RuleVersion

    # Aggregate metrics
    total_iterations: int
    overall_improvement: float  # From root to current

    # Metadata
    created_at: datetime
    last_updated: datetime

    # Analysis cache
    _version_map: Dict[str, RuleVersion] = field(default_factory=dict, repr=False)
    _evolution_tree: Dict[str, List[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Build version map and tree."""
        self._version_map = {v.rule_id: v for v in self.all_versions}
        self._evolution_tree = self._build_tree()

    def _build_tree(self) -> Dict[str, List[str]]:
        """Build parent -> children mapping."""
        tree = {}
        for version in self.all_versions:
            if version.parent_version:
                if version.parent_version not in tree:
                    tree[version.parent_version] = []
                tree[version.parent_version].append(version.rule_id)
        return tree

    def get_evolution_path(self, to_version_id: Optional[str] = None) -> List[RuleVersion]:
        """Get path from root to specified version (or current)."""
        target_id = to_version_id or self.current_version.rule_id
        target = self._version_map.get(target_id)

        if not target:
            return []

        # Walk backwards from target to root
        path = []
        current = target
        while current:
            path.insert(0, current)
            if current.parent_version:
                current = self._version_map.get(current.parent_version)
            else:
                current = None

        return path

    def get_version_tree(self) -> Dict[str, List[str]]:
        """Get complete branching structure."""
        return self._evolution_tree.copy()

    def get_branches(self) -> List[List[RuleVersion]]:
        """Get all branches from root to leaves."""
        leaves = [v for v in self.all_versions if v.rule_id not in self._evolution_tree]
        return [self.get_evolution_path(leaf.rule_id) for leaf in leaves]

    def get_version(self, version_id: str) -> Optional[RuleVersion]:
        """Get a specific version by ID."""
        return self._version_map.get(version_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_family_id": self.rule_family_id,
            "root_version": self.root_version.to_dict(),
            "all_versions": [v.to_dict() for v in self.all_versions],
            "current_version": self.current_version.to_dict(),
            "total_iterations": self.total_iterations,
            "overall_improvement": self.overall_improvement,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleLineage":
        """Create from dictionary."""
        root = RuleVersion.from_dict(data["root_version"])
        all_versions = [RuleVersion.from_dict(v) for v in data["all_versions"]]
        current = RuleVersion.from_dict(data["current_version"])

        return cls(
            rule_family_id=data["rule_family_id"],
            root_version=root,
            all_versions=all_versions,
            current_version=current,
            total_iterations=data["total_iterations"],
            overall_improvement=data["overall_improvement"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )
