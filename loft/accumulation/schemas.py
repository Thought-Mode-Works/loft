"""
Data schemas for rule accumulation pipeline.

Defines data structures for accumulation results, conflicts, and reports.

Issue #273: Continuous Rule Accumulation Pipeline
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Conflict:
    """
    Represents a conflict between rules.

    Types of conflicts:
    - contradiction: Rules directly contradict each other
    - subsumption: One rule makes the other redundant
    - inconsistency: Rules lead to inconsistent conclusions
    """

    conflict_type: str  # contradiction, subsumption, inconsistency
    new_rule: str  # The new rule being considered
    existing_rule_id: str  # ID of conflicting existing rule
    existing_rule: str  # Text of conflicting existing rule
    explanation: str  # Why they conflict
    severity: float  # 0.0-1.0, how severe the conflict is

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Conflict({self.conflict_type}): "
            f"{self.new_rule[:50]}... vs {self.existing_rule_id[:8]}"
        )


@dataclass
class ConflictResolution:
    """
    Resolution strategy for a conflict.

    Determines whether and how to add a new rule that conflicts
    with existing rules.
    """

    should_add: bool  # Whether to add the new rule
    action: str  # "add", "skip", "replace", "merge"
    reason: str  # Explanation of decision
    rules_to_archive: List[str] = field(
        default_factory=list
    )  # Rule IDs to archive if adding
    modified_rule: Optional[str] = None  # Modified version if merging

    def __str__(self) -> str:
        """String representation."""
        return f"Resolution({self.action}): {self.reason}"


@dataclass
class RuleCandidate:
    """
    A candidate rule extracted from a case.

    Contains the rule text, metadata, and extraction context.
    """

    asp_rule: str  # The ASP rule text
    domain: str  # Legal domain
    confidence: float  # Confidence in this rule (0.0-1.0)
    reasoning: str  # Why this rule was extracted
    source_case_id: str  # ID of source case
    principle: Optional[str] = None  # Legal principle it represents

    def __str__(self) -> str:
        """String representation."""
        return f"RuleCandidate({self.domain}): {self.asp_rule[:60]}..."


@dataclass
class AccumulationResult:
    """
    Result of accumulating rules from a single case.

    Tracks what rules were added, skipped, and why.
    """

    case_id: str  # ID of processed case
    rules_added: int  # Number of rules added to database
    rules_skipped: int  # Number of rules skipped
    rule_ids: List[str] = field(default_factory=list)  # IDs of added rules
    skipped_reasons: List[str] = field(default_factory=list)  # Why rules were skipped
    conflicts_found: List[Conflict] = field(default_factory=list)  # Conflicts detected
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0  # Time taken to process case

    @property
    def success_rate(self) -> float:
        """Calculate success rate of rule addition."""
        total = self.rules_added + self.rules_skipped
        return self.rules_added / total if total > 0 else 0.0

    def __str__(self) -> str:
        """String representation."""
        return (
            f"AccumulationResult({self.case_id}): "
            f"+{self.rules_added} rules, -{self.rules_skipped} skipped"
        )


@dataclass
class BatchAccumulationReport:
    """
    Report from batch accumulation over multiple cases.

    Aggregates statistics across all processed cases.
    """

    results: List[AccumulationResult]
    total_cases: int = 0
    total_rules_added: int = 0
    total_rules_skipped: int = 0
    total_conflicts: int = 0
    domains: dict = field(default_factory=dict)  # Stats by domain
    avg_processing_time_ms: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    def __post_init__(self):
        """Calculate aggregate statistics."""
        self.total_cases = len(self.results)

        if self.total_cases == 0:
            return

        # Aggregate counts
        for result in self.results:
            self.total_rules_added += result.rules_added
            self.total_rules_skipped += result.rules_skipped
            self.total_conflicts += len(result.conflicts_found)

        # Calculate average processing time
        times = [r.processing_time_ms for r in self.results if r.processing_time_ms > 0]
        self.avg_processing_time_ms = sum(times) / len(times) if times else 0.0

    @property
    def overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.total_rules_added + self.total_rules_skipped
        return self.total_rules_added / total if total > 0 else 0.0

    @property
    def rules_per_case(self) -> float:
        """Average rules added per case."""
        return (
            self.total_rules_added / self.total_cases if self.total_cases > 0 else 0.0
        )

    def to_string(self) -> str:
        """Format report as string."""
        report = "Batch Accumulation Report\n"
        report += "=" * 50 + "\n\n"
        report += f"Cases Processed: {self.total_cases}\n"
        report += f"Rules Added: {self.total_rules_added}\n"
        report += f"Rules Skipped: {self.total_rules_skipped}\n"
        report += f"Conflicts Detected: {self.total_conflicts}\n"
        report += f"Success Rate: {self.overall_success_rate:.1%}\n"
        report += f"Rules per Case: {self.rules_per_case:.2f}\n"
        report += f"Avg Processing Time: {self.avg_processing_time_ms:.0f}ms\n"

        if self.domains:
            report += "\nBy Domain:\n"
            for domain, stats in sorted(self.domains.items()):
                report += f"  {domain}: {stats}\n"

        return report

    def __str__(self) -> str:
        """String representation."""
        return (
            f"BatchReport: {self.total_cases} cases, "
            f"+{self.total_rules_added} rules ({self.overall_success_rate:.1%})"
        )


@dataclass
class CaseData:
    """
    Structured representation of a legal case.

    Loaded from JSON case files in datasets.
    """

    case_id: str
    description: str
    facts: List[str]
    asp_facts: str
    question: str
    ground_truth: str
    rationale: str
    legal_citations: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    domain: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CaseData":
        """Create CaseData from dictionary."""
        return cls(
            case_id=data.get("id", "unknown"),
            description=data.get("description", ""),
            facts=data.get("facts", []),
            asp_facts=data.get("asp_facts", ""),
            question=data.get("question", ""),
            ground_truth=data.get("ground_truth", ""),
            rationale=data.get("rationale", ""),
            legal_citations=data.get("legal_citations", []),
            difficulty=data.get("difficulty", "medium"),
            domain=data.get("domain"),
        )

    def __str__(self) -> str:
        """String representation."""
        return f"CaseData({self.case_id}): {self.description[:60]}..."
