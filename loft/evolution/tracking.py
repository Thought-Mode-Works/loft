"""
Rule metadata and evolution tracking.

Provides dataclasses and tracking logic for rule evolution,
including provenance, validation history, and dialectical refinement.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger


class StratificationLayer(Enum):
    """Stratification layers for rules."""

    CONSTITUTIONAL = "constitutional"  # Core axioms, rarely change
    STRATEGIC = "strategic"  # High-level policies
    TACTICAL = "tactical"  # Domain-specific rules
    OPERATIONAL = "operational"  # Case-specific, frequently updated


class RuleStatus(Enum):
    """Lifecycle status of a rule."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    ROLLBACK = "rollback"
    AB_TESTING = "ab_testing"
    CANDIDATE = "candidate"


@dataclass
class ValidationResult:
    """Result of validating a rule against test cases."""

    timestamp: datetime
    test_cases_evaluated: int
    passed: int
    failed: int
    accuracy: float
    error_cases: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "test_cases_evaluated": self.test_cases_evaluated,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "error_cases": self.error_cases,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ValidationResult":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            test_cases_evaluated=data["test_cases_evaluated"],
            passed=data["passed"],
            failed=data["failed"],
            accuracy=data["accuracy"],
            error_cases=data.get("error_cases", []),
            notes=data.get("notes"),
        )


@dataclass
class ABTestResult:
    """Result of an A/B test comparing rule variants."""

    test_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    variant_a_id: str = ""
    variant_b_id: str = ""
    variant_a_accuracy: float = 0.0
    variant_b_accuracy: float = 0.0
    cases_evaluated: int = 0
    p_value: Optional[float] = None
    winner: Optional[str] = None  # "a", "b", or None if inconclusive
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "variant_a_id": self.variant_a_id,
            "variant_b_id": self.variant_b_id,
            "variant_a_accuracy": self.variant_a_accuracy,
            "variant_b_accuracy": self.variant_b_accuracy,
            "cases_evaluated": self.cases_evaluated,
            "p_value": self.p_value,
            "winner": self.winner,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ABTestResult":
        """Create from dictionary."""
        return cls(
            test_id=data["test_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            variant_a_id=data.get("variant_a_id", ""),
            variant_b_id=data.get("variant_b_id", ""),
            variant_a_accuracy=data.get("variant_a_accuracy", 0.0),
            variant_b_accuracy=data.get("variant_b_accuracy", 0.0),
            cases_evaluated=data.get("cases_evaluated", 0),
            p_value=data.get("p_value"),
            winner=data.get("winner"),
            notes=data.get("notes"),
        )


@dataclass
class DialecticalHistory:
    """History of dialectical refinement for a rule."""

    thesis_rule: Optional[str] = None
    thesis_reasoning: Optional[str] = None
    antithesis_critiques: List[str] = field(default_factory=list)
    synthesis_rule: Optional[str] = None
    synthesis_reasoning: Optional[str] = None
    cycles_completed: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "thesis_rule": self.thesis_rule,
            "thesis_reasoning": self.thesis_reasoning,
            "antithesis_critiques": self.antithesis_critiques,
            "synthesis_rule": self.synthesis_rule,
            "synthesis_reasoning": self.synthesis_reasoning,
            "cycles_completed": self.cycles_completed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DialecticalHistory":
        """Create from dictionary."""
        return cls(
            thesis_rule=data.get("thesis_rule"),
            thesis_reasoning=data.get("thesis_reasoning"),
            antithesis_critiques=data.get("antithesis_critiques", []),
            synthesis_rule=data.get("synthesis_rule"),
            synthesis_reasoning=data.get("synthesis_reasoning"),
            cycles_completed=data.get("cycles_completed", 0),
        )


@dataclass
class RuleMetadata:
    """Complete provenance and evolution history for a rule."""

    # Identity
    rule_id: str
    rule_text: str
    natural_language: str
    version: str = "1.0"

    # Provenance
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "unknown"
    source_gap: Optional[str] = None
    source_case: Optional[str] = None
    parent_rule_id: Optional[str] = None

    # Dialectical history
    dialectical: DialecticalHistory = field(default_factory=DialecticalHistory)

    # Validation history
    validation_results: List[ValidationResult] = field(default_factory=list)

    # Performance tracking
    test_case_results: Dict[str, bool] = field(default_factory=dict)
    accuracy_history: List[Tuple[datetime, float]] = field(default_factory=list)

    # A/B testing
    ab_test_id: Optional[str] = None
    ab_test_result: Optional[ABTestResult] = None

    # Stratification
    current_layer: StratificationLayer = StratificationLayer.OPERATIONAL
    layer_history: List[Tuple[datetime, StratificationLayer]] = field(
        default_factory=list
    )
    stability_score: float = 0.0

    # Impact analysis
    downstream_rules: List[str] = field(default_factory=list)
    upstream_rules: List[str] = field(default_factory=list)
    cases_affected: Set[str] = field(default_factory=set)

    # Lifecycle
    status: RuleStatus = RuleStatus.CANDIDATE
    superseded_by: Optional[str] = None
    deprecation_reason: Optional[str] = None

    @property
    def current_accuracy(self) -> float:
        """Get the most recent accuracy."""
        if self.validation_results:
            return self.validation_results[-1].accuracy
        return 0.0

    @property
    def total_validations(self) -> int:
        """Get total number of validations performed."""
        return len(self.validation_results)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "natural_language": self.natural_language,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "source_gap": self.source_gap,
            "source_case": self.source_case,
            "parent_rule_id": self.parent_rule_id,
            "dialectical": self.dialectical.to_dict(),
            "validation_results": [v.to_dict() for v in self.validation_results],
            "test_case_results": self.test_case_results,
            "accuracy_history": [
                (ts.isoformat(), acc) for ts, acc in self.accuracy_history
            ],
            "ab_test_id": self.ab_test_id,
            "ab_test_result": (
                self.ab_test_result.to_dict() if self.ab_test_result else None
            ),
            "current_layer": self.current_layer.value,
            "layer_history": [
                (ts.isoformat(), layer.value) for ts, layer in self.layer_history
            ],
            "stability_score": self.stability_score,
            "downstream_rules": self.downstream_rules,
            "upstream_rules": self.upstream_rules,
            "cases_affected": list(self.cases_affected),
            "status": self.status.value,
            "superseded_by": self.superseded_by,
            "deprecation_reason": self.deprecation_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RuleMetadata":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            rule_text=data["rule_text"],
            natural_language=data["natural_language"],
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "unknown"),
            source_gap=data.get("source_gap"),
            source_case=data.get("source_case"),
            parent_rule_id=data.get("parent_rule_id"),
            dialectical=DialecticalHistory.from_dict(data.get("dialectical", {})),
            validation_results=[
                ValidationResult.from_dict(v)
                for v in data.get("validation_results", [])
            ],
            test_case_results=data.get("test_case_results", {}),
            accuracy_history=[
                (datetime.fromisoformat(ts), acc)
                for ts, acc in data.get("accuracy_history", [])
            ],
            ab_test_id=data.get("ab_test_id"),
            ab_test_result=(
                ABTestResult.from_dict(data["ab_test_result"])
                if data.get("ab_test_result")
                else None
            ),
            current_layer=StratificationLayer(data.get("current_layer", "operational")),
            layer_history=[
                (datetime.fromisoformat(ts), StratificationLayer(layer))
                for ts, layer in data.get("layer_history", [])
            ],
            stability_score=data.get("stability_score", 0.0),
            downstream_rules=data.get("downstream_rules", []),
            upstream_rules=data.get("upstream_rules", []),
            cases_affected=set(data.get("cases_affected", [])),
            status=RuleStatus(data.get("status", "candidate")),
            superseded_by=data.get("superseded_by"),
            deprecation_reason=data.get("deprecation_reason"),
        )


def generate_rule_id(rule_text: str) -> str:
    """Generate a unique rule ID based on content hash."""
    content_hash = hashlib.sha256(rule_text.encode()).hexdigest()[:8]
    unique_part = uuid.uuid4().hex[:4]
    return f"rule_{content_hash}_{unique_part}"


def parse_version(version: str) -> Tuple[int, int]:
    """Parse version string into (major, minor) tuple."""
    parts = version.split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def increment_version(version: str, major: bool = False) -> str:
    """Increment version string."""
    major_v, minor_v = parse_version(version)
    if major:
        return f"{major_v + 1}.0"
    return f"{major_v}.{minor_v + 1}"


class RuleEvolutionTracker:
    """
    Track rule evolution through creation, validation, and refinement.

    Example:
        tracker = RuleEvolutionTracker()

        # Create initial rule
        meta = tracker.create_rule(
            rule_text="valid(C) :- signed(C).",
            natural_language="A contract is valid if signed",
            created_by="llm_generator",
        )

        # Record validation
        tracker.record_validation(
            meta.rule_id,
            test_cases=10,
            passed=8,
            failed=2,
        )

        # Create new version
        new_meta = tracker.create_version(
            parent_id=meta.rule_id,
            new_rule_text="valid(C) :- signed(C), has_consideration(C).",
            change_reason="Added consideration requirement",
        )
    """

    def __init__(self):
        """Initialize rule evolution tracker."""
        self._rules: Dict[str, RuleMetadata] = {}
        self._ab_tests: Dict[str, ABTestResult] = {}
        self._version_chains: Dict[str, List[str]] = {}  # root_id -> [v1, v2, v3, ...]

    def create_rule(
        self,
        rule_text: str,
        natural_language: str,
        created_by: str = "unknown",
        source_gap: Optional[str] = None,
        source_case: Optional[str] = None,
        layer: StratificationLayer = StratificationLayer.OPERATIONAL,
    ) -> RuleMetadata:
        """
        Create and track a new rule.

        Args:
            rule_text: ASP rule code
            natural_language: Human-readable description
            created_by: Creator identifier (e.g., "llm_generator", "human")
            source_gap: Knowledge gap that prompted this rule
            source_case: Case that triggered rule generation
            layer: Initial stratification layer

        Returns:
            RuleMetadata for the new rule
        """
        rule_id = generate_rule_id(rule_text)

        metadata = RuleMetadata(
            rule_id=rule_id,
            rule_text=rule_text,
            natural_language=natural_language,
            version="1.0",
            created_at=datetime.now(),
            created_by=created_by,
            source_gap=source_gap,
            source_case=source_case,
            current_layer=layer,
            layer_history=[(datetime.now(), layer)],
            status=RuleStatus.CANDIDATE,
        )

        self._rules[rule_id] = metadata
        self._version_chains[rule_id] = [rule_id]

        logger.info(f"Created rule {rule_id}: {rule_text[:50]}...")
        return metadata

    def record_validation(
        self,
        rule_id: str,
        test_cases: int,
        passed: int,
        failed: int,
        error_cases: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> ValidationResult:
        """
        Record a validation result for a rule.

        Args:
            rule_id: ID of the rule being validated
            test_cases: Total test cases evaluated
            passed: Number passed
            failed: Number failed
            error_cases: IDs of cases that failed
            notes: Optional validation notes

        Returns:
            The created ValidationResult
        """
        if rule_id not in self._rules:
            raise ValueError(f"Rule {rule_id} not found")

        metadata = self._rules[rule_id]
        accuracy = passed / test_cases if test_cases > 0 else 0.0

        result = ValidationResult(
            timestamp=datetime.now(),
            test_cases_evaluated=test_cases,
            passed=passed,
            failed=failed,
            accuracy=accuracy,
            error_cases=error_cases or [],
            notes=notes,
        )

        metadata.validation_results.append(result)
        metadata.accuracy_history.append((datetime.now(), accuracy))

        # Update status based on validation
        if metadata.status == RuleStatus.CANDIDATE and accuracy >= 0.8:
            metadata.status = RuleStatus.ACTIVE

        logger.info(f"Validation for {rule_id}: {accuracy:.1%} ({passed}/{test_cases})")
        return result

    def create_version(
        self,
        parent_id: str,
        new_rule_text: str,
        change_reason: str,
        major_version: bool = False,
        new_natural_language: Optional[str] = None,
    ) -> RuleMetadata:
        """
        Create a new version of an existing rule.

        Args:
            parent_id: ID of the parent rule
            new_rule_text: Updated ASP rule code
            change_reason: Reason for the change
            major_version: If True, increment major version
            new_natural_language: Updated description (optional)

        Returns:
            RuleMetadata for the new version
        """
        if parent_id not in self._rules:
            raise ValueError(f"Parent rule {parent_id} not found")

        parent = self._rules[parent_id]
        new_version = increment_version(parent.version, major=major_version)
        new_id = generate_rule_id(new_rule_text)

        # Find root of version chain
        root_id = parent_id
        for root, chain in self._version_chains.items():
            if parent_id in chain:
                root_id = root
                break

        metadata = RuleMetadata(
            rule_id=new_id,
            rule_text=new_rule_text,
            natural_language=new_natural_language or parent.natural_language,
            version=new_version,
            created_at=datetime.now(),
            created_by=parent.created_by,
            source_gap=parent.source_gap,
            source_case=parent.source_case,
            parent_rule_id=parent_id,
            current_layer=parent.current_layer,
            layer_history=[(datetime.now(), parent.current_layer)],
            upstream_rules=parent.upstream_rules.copy(),
            downstream_rules=parent.downstream_rules.copy(),
            status=RuleStatus.CANDIDATE,
        )

        # Copy dialectical history if available
        if parent.dialectical.cycles_completed > 0:
            metadata.dialectical = DialecticalHistory(
                thesis_rule=parent.dialectical.thesis_rule,
                thesis_reasoning=parent.dialectical.thesis_reasoning,
                antithesis_critiques=parent.dialectical.antithesis_critiques.copy(),
                synthesis_rule=new_rule_text,
                synthesis_reasoning=change_reason,
                cycles_completed=parent.dialectical.cycles_completed,
            )

        self._rules[new_id] = metadata

        # Update version chain
        if root_id in self._version_chains:
            self._version_chains[root_id].append(new_id)
        else:
            self._version_chains[root_id] = [parent_id, new_id]

        logger.info(
            f"Created version {new_version} of rule {parent_id[:12]}... -> {new_id[:12]}..."
        )
        return metadata

    def record_dialectical_cycle(
        self,
        rule_id: str,
        thesis: str,
        thesis_reasoning: str,
        critiques: List[str],
        synthesis: str,
        synthesis_reasoning: str,
    ) -> RuleMetadata:
        """
        Record a dialectical refinement cycle.

        Args:
            rule_id: ID of the rule being refined
            thesis: Initial rule proposal
            thesis_reasoning: Reasoning for thesis
            critiques: List of criticisms
            synthesis: Final synthesized rule
            synthesis_reasoning: Reasoning for synthesis

        Returns:
            Updated or new RuleMetadata
        """
        if rule_id not in self._rules:
            raise ValueError(f"Rule {rule_id} not found")

        metadata = self._rules[rule_id]

        # Update dialectical history
        metadata.dialectical = DialecticalHistory(
            thesis_rule=thesis,
            thesis_reasoning=thesis_reasoning,
            antithesis_critiques=critiques,
            synthesis_rule=synthesis,
            synthesis_reasoning=synthesis_reasoning,
            cycles_completed=metadata.dialectical.cycles_completed + 1,
        )

        # If synthesis differs from current rule, create new version
        if synthesis != metadata.rule_text:
            return self.create_version(
                parent_id=rule_id,
                new_rule_text=synthesis,
                change_reason=f"Dialectical synthesis: {synthesis_reasoning}",
            )

        return metadata

    def promote_rule(
        self,
        rule_id: str,
        to_layer: StratificationLayer,
        reason: str,
    ) -> None:
        """
        Promote a rule to a higher stratification layer.

        Args:
            rule_id: ID of the rule to promote
            to_layer: Target layer
            reason: Reason for promotion
        """
        if rule_id not in self._rules:
            raise ValueError(f"Rule {rule_id} not found")

        metadata = self._rules[rule_id]
        old_layer = metadata.current_layer

        metadata.current_layer = to_layer
        metadata.layer_history.append((datetime.now(), to_layer))

        logger.info(
            f"Promoted rule {rule_id}: {old_layer.value} -> {to_layer.value} ({reason})"
        )

    def deprecate_rule(
        self,
        rule_id: str,
        reason: str,
        superseded_by: Optional[str] = None,
    ) -> None:
        """
        Deprecate a rule.

        Args:
            rule_id: ID of the rule to deprecate
            reason: Reason for deprecation
            superseded_by: ID of replacing rule (optional)
        """
        if rule_id not in self._rules:
            raise ValueError(f"Rule {rule_id} not found")

        metadata = self._rules[rule_id]
        metadata.status = (
            RuleStatus.SUPERSEDED if superseded_by else RuleStatus.DEPRECATED
        )
        metadata.deprecation_reason = reason
        metadata.superseded_by = superseded_by

        logger.info(f"Deprecated rule {rule_id}: {reason}")

    def start_ab_test(
        self,
        variant_a_id: str,
        variant_b_id: str,
        test_id: Optional[str] = None,
    ) -> ABTestResult:
        """
        Start an A/B test between two rule variants.

        Args:
            variant_a_id: ID of first variant
            variant_b_id: ID of second variant
            test_id: Optional test ID (auto-generated if not provided)

        Returns:
            ABTestResult for the new test
        """
        if variant_a_id not in self._rules:
            raise ValueError(f"Variant A {variant_a_id} not found")
        if variant_b_id not in self._rules:
            raise ValueError(f"Variant B {variant_b_id} not found")

        test_id = test_id or f"ab_{uuid.uuid4().hex[:8]}"

        result = ABTestResult(
            test_id=test_id,
            started_at=datetime.now(),
            variant_a_id=variant_a_id,
            variant_b_id=variant_b_id,
        )

        self._ab_tests[test_id] = result

        # Update rule statuses
        self._rules[variant_a_id].status = RuleStatus.AB_TESTING
        self._rules[variant_a_id].ab_test_id = test_id
        self._rules[variant_b_id].status = RuleStatus.AB_TESTING
        self._rules[variant_b_id].ab_test_id = test_id

        logger.info(f"Started A/B test {test_id}: {variant_a_id} vs {variant_b_id}")
        return result

    def complete_ab_test(
        self,
        test_id: str,
        winner: Optional[str],
        p_value: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> ABTestResult:
        """
        Complete an A/B test and declare winner.

        Args:
            test_id: ID of the A/B test
            winner: "a", "b", or None if inconclusive
            p_value: Statistical significance
            notes: Optional notes

        Returns:
            Updated ABTestResult
        """
        if test_id not in self._ab_tests:
            raise ValueError(f"A/B test {test_id} not found")

        result = self._ab_tests[test_id]
        result.completed_at = datetime.now()
        result.winner = winner
        result.p_value = p_value
        result.notes = notes

        # Update rule statuses
        if winner == "a":
            self._rules[result.variant_a_id].status = RuleStatus.ACTIVE
            self.deprecate_rule(
                result.variant_b_id,
                reason=f"Lost A/B test {test_id}",
                superseded_by=result.variant_a_id,
            )
        elif winner == "b":
            self._rules[result.variant_b_id].status = RuleStatus.ACTIVE
            self.deprecate_rule(
                result.variant_a_id,
                reason=f"Lost A/B test {test_id}",
                superseded_by=result.variant_b_id,
            )
        else:
            # Inconclusive - keep variant A as default
            self._rules[result.variant_a_id].status = RuleStatus.ACTIVE
            self._rules[result.variant_b_id].status = RuleStatus.DEPRECATED

        logger.info(f"Completed A/B test {test_id}: winner = {winner}")
        return result

    def get_rule(self, rule_id: str) -> Optional[RuleMetadata]:
        """Get rule metadata by ID."""
        return self._rules.get(rule_id)

    def get_version_history(self, rule_id: str) -> List[RuleMetadata]:
        """Get all versions of a rule."""
        # Find root of version chain
        root_id = rule_id
        for root, chain in self._version_chains.items():
            if rule_id in chain:
                root_id = root
                break

        chain = self._version_chains.get(root_id, [rule_id])
        return [self._rules[rid] for rid in chain if rid in self._rules]

    def get_active_ab_tests(self) -> List[ABTestResult]:
        """Get all currently running A/B tests."""
        return [test for test in self._ab_tests.values() if test.completed_at is None]

    def get_rules_by_status(self, status: RuleStatus) -> List[RuleMetadata]:
        """Get all rules with a given status."""
        return [meta for meta in self._rules.values() if meta.status == status]

    def get_rules_by_layer(self, layer: StratificationLayer) -> List[RuleMetadata]:
        """Get all rules in a given stratification layer."""
        return [meta for meta in self._rules.values() if meta.current_layer == layer]

    def get_all_rules(self) -> List[RuleMetadata]:
        """Get all tracked rules."""
        return list(self._rules.values())

    def calculate_stability_scores(self) -> Dict[str, float]:
        """
        Calculate stability scores for all rules.

        Stability = 1 / (1 + number of version changes + layer changes)

        Returns:
            Dictionary mapping rule_id to stability score
        """
        scores = {}
        for rule_id, metadata in self._rules.items():
            version_changes = len(self.get_version_history(rule_id)) - 1
            layer_changes = len(metadata.layer_history) - 1
            total_changes = version_changes + layer_changes
            stability = 1.0 / (1.0 + total_changes)
            scores[rule_id] = stability
            metadata.stability_score = stability
        return scores
