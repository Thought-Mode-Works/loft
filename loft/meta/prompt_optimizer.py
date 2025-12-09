"""
Prompt Optimization Module for Meta-Reasoning.

This module provides automatic prompt optimization through:
- Performance tracking and analysis
- A/B testing framework with statistical significance
- Automatic refinement suggestions
- Prompt improvement candidate generation

The optimizer builds on the PromptRegistry in loft/neural/ and integrates
with the meta-reasoning layer for self-improvement capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import math
import random
import re
import uuid


class PromptCategory(Enum):
    """Categories of prompts for optimization."""

    RULE_GENERATION = "rule_generation"
    VALIDATION = "validation"
    TRANSLATION = "translation"
    CRITIC = "critic"
    SYNTHESIS = "synthesis"
    GENERAL = "general"


class TestStatus(Enum):
    """Status of an A/B test."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class PromptVersion:
    """
    A versioned prompt with metadata and performance tracking.

    Attributes:
        prompt_id: Unique identifier for the prompt
        version: Version number
        template: The prompt template string
        category: Category of the prompt
        created_at: When this version was created
        parent_version: ID of the prompt this was derived from
        modification_reason: Why this version was created
        variables: Template variables used
    """

    prompt_id: str
    version: int
    template: str
    category: PromptCategory = PromptCategory.GENERAL
    created_at: datetime = field(default_factory=datetime.now)
    parent_version: Optional[str] = None
    modification_reason: str = ""
    variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Extract variables from template if not provided."""
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        pattern = r"\{(\w+)\}"
        return list(set(re.findall(pattern, self.template)))

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided arguments."""
        return self.template.format(**kwargs)

    @property
    def full_id(self) -> str:
        """Full identifier including version."""
        return f"{self.prompt_id}_v{self.version}"

    @property
    def template_hash(self) -> str:
        """Hash of the template for deduplication."""
        return hashlib.md5(self.template.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "template": self.template,
            "category": self.category.value,
            "created_at": self.created_at.isoformat(),
            "parent_version": self.parent_version,
            "modification_reason": self.modification_reason,
            "variables": self.variables,
            "full_id": self.full_id,
            "template_hash": self.template_hash,
        }


@dataclass
class PromptMetrics:
    """
    Performance metrics for a prompt version.

    Tracks empirical data about how well a prompt performs
    across different usage scenarios.
    """

    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    total_confidence: float = 0.0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    syntax_valid_count: int = 0
    domain_performance: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"success": 0, "total": 0})
    )

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        if self.total_uses == 0:
            return 0.0
        return self.total_confidence / self.total_uses

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_uses == 0:
            return 0.0
        return self.total_latency_ms / self.total_uses

    @property
    def syntax_validity_rate(self) -> float:
        """Calculate syntax validity rate."""
        if self.total_uses == 0:
            return 0.0
        return self.syntax_valid_count / self.total_uses

    def record(
        self,
        success: bool,
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        syntax_valid: bool = True,
        domain: Optional[str] = None,
    ) -> None:
        """Record a single use of the prompt."""
        self.total_uses += 1
        if success:
            self.successful_uses += 1
        else:
            self.failed_uses += 1

        self.total_confidence += confidence
        self.total_latency_ms += latency_ms
        self.total_cost_usd += cost_usd

        if syntax_valid:
            self.syntax_valid_count += 1

        if domain:
            self.domain_performance[domain]["total"] += 1
            if success:
                self.domain_performance[domain]["success"] += 1

    def get_domain_success_rate(self, domain: str) -> float:
        """Get success rate for a specific domain."""
        if domain not in self.domain_performance:
            return 0.0
        domain_data = self.domain_performance[domain]
        if domain_data["total"] == 0:
            return 0.0
        return domain_data["success"] / domain_data["total"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_uses": self.total_uses,
            "successful_uses": self.successful_uses,
            "failed_uses": self.failed_uses,
            "success_rate": self.success_rate,
            "average_confidence": self.average_confidence,
            "average_latency_ms": self.average_latency_ms,
            "syntax_validity_rate": self.syntax_validity_rate,
            "total_cost_usd": self.total_cost_usd,
            "domain_performance": dict(self.domain_performance),
        }


@dataclass
class EffectivenessReport:
    """
    Report on prompt effectiveness over time.

    Provides analysis of how well a prompt performs and
    identifies areas for improvement.
    """

    prompt_id: str
    analysis_period_start: datetime
    analysis_period_end: datetime
    total_samples: int
    overall_effectiveness: float
    confidence_interval: Tuple[float, float]
    trend: str  # "improving", "stable", "degrading"
    domain_breakdown: Dict[str, float]
    weakness_areas: List[str]
    strength_areas: List[str]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "analysis_period_start": self.analysis_period_start.isoformat(),
            "analysis_period_end": self.analysis_period_end.isoformat(),
            "total_samples": self.total_samples,
            "overall_effectiveness": self.overall_effectiveness,
            "confidence_interval": self.confidence_interval,
            "trend": self.trend,
            "domain_breakdown": self.domain_breakdown,
            "weakness_areas": self.weakness_areas,
            "strength_areas": self.strength_areas,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class ABTestConfig:
    """
    Configuration for an A/B test.

    Attributes:
        test_id: Unique identifier for this test
        prompt_a: First prompt version (control)
        prompt_b: Second prompt version (treatment)
        min_samples_per_variant: Minimum samples needed per variant
        significance_level: Statistical significance threshold (default 0.05)
        allocation_ratio: Traffic split ratio (default 0.5)
    """

    test_id: str
    prompt_a: PromptVersion
    prompt_b: PromptVersion
    min_samples_per_variant: int = 30
    significance_level: float = 0.05
    allocation_ratio: float = 0.5


@dataclass
class ABTestResult:
    """
    Results of an A/B test.

    Contains statistical analysis comparing two prompt versions.
    """

    test_id: str
    prompt_a_id: str
    prompt_b_id: str
    prompt_a_metrics: PromptMetrics
    prompt_b_metrics: PromptMetrics
    winner: Optional[str]  # None if no significant difference
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    recommendation: str
    completed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "prompt_a_id": self.prompt_a_id,
            "prompt_b_id": self.prompt_b_id,
            "prompt_a_metrics": self.prompt_a_metrics.to_dict(),
            "prompt_b_metrics": self.prompt_b_metrics.to_dict(),
            "winner": self.winner,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "is_significant": self.is_significant,
            "recommendation": self.recommendation,
            "completed_at": self.completed_at.isoformat(),
        }


@dataclass
class ImprovementCandidate:
    """
    A candidate improvement for a prompt.

    Generated based on performance analysis and improvement heuristics.
    """

    candidate_id: str
    original_prompt_id: str
    new_template: str
    improvement_type: str  # "clarification", "structure", "examples", "constraints"
    expected_improvement: float
    rationale: str
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "original_prompt_id": self.original_prompt_id,
            "new_template": self.new_template,
            "improvement_type": self.improvement_type,
            "expected_improvement": self.expected_improvement,
            "rationale": self.rationale,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class ImprovementSuggestion:
    """
    A specific improvement suggestion for a prompt.

    Contains actionable advice for improving prompt performance
    based on analysis of its metrics and weaknesses.
    """

    suggestion_id: str
    prompt_id: str
    suggestion_type: str  # "wording", "structure", "examples", "constraints", "domain"
    description: str
    rationale: str
    expected_improvement_percentage: float
    confidence: float
    priority: str = "medium"  # "high", "medium", "low"
    implementation_hint: str = ""
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "prompt_id": self.prompt_id,
            "suggestion_type": self.suggestion_type,
            "description": self.description,
            "rationale": self.rationale,
            "expected_improvement_percentage": self.expected_improvement_percentage,
            "confidence": self.confidence,
            "priority": self.priority,
            "implementation_hint": self.implementation_hint,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class AggregateEffectivenessReport:
    """
    Aggregate effectiveness report across all prompts.

    Provides a comprehensive overview of the prompt optimization system's
    performance with per-prompt summaries and overall metrics.
    """

    report_id: str
    generated_at: datetime
    total_prompts_tracked: int
    total_uses_recorded: int
    overall_success_rate: float
    overall_average_confidence: float
    overall_average_latency_ms: float
    prompt_reports: Dict[str, Dict[str, Any]]  # prompt_id -> summary dict
    top_performers: List[Tuple[str, float]]  # (prompt_id, success_rate)
    bottom_performers: List[Tuple[str, float]]  # (prompt_id, success_rate)
    category_breakdown: Dict[str, Dict[str, float]]  # category -> metrics
    domain_breakdown: Dict[str, Dict[str, float]]  # domain -> metrics
    trends: Dict[str, str]  # prompt_id -> trend ("improving", "stable", "degrading")
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "total_prompts_tracked": self.total_prompts_tracked,
            "total_uses_recorded": self.total_uses_recorded,
            "overall_success_rate": self.overall_success_rate,
            "overall_average_confidence": self.overall_average_confidence,
            "overall_average_latency_ms": self.overall_average_latency_ms,
            "prompt_reports": self.prompt_reports,
            "top_performers": self.top_performers,
            "bottom_performers": self.bottom_performers,
            "category_breakdown": self.category_breakdown,
            "domain_breakdown": self.domain_breakdown,
            "trends": self.trends,
            "recommendations": self.recommendations,
        }


class PromptOptimizer:
    """
    Optimizes prompts based on empirical performance data.

    This class provides:
    - Performance tracking across prompt versions
    - Effectiveness analysis over time
    - Automatic generation of improvement candidates
    - Integration with A/B testing for validation
    """

    def __init__(
        self,
        improvement_threshold: float = 0.15,
        min_samples_for_analysis: int = 10,
    ):
        """
        Initialize the prompt optimizer.

        Args:
            improvement_threshold: Target improvement percentage (default 15%)
            min_samples_for_analysis: Minimum samples needed for analysis
        """
        self.improvement_threshold = improvement_threshold
        self.min_samples_for_analysis = min_samples_for_analysis

        # Store prompt versions and their metrics
        self._prompts: Dict[str, PromptVersion] = {}
        self._metrics: Dict[str, PromptMetrics] = {}

        # Track version history
        self._version_history: Dict[str, List[str]] = defaultdict(list)

        # Performance history over time (for trend analysis)
        self._performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def register_prompt(self, prompt: PromptVersion) -> None:
        """
        Register a prompt version for tracking.

        Args:
            prompt: The prompt version to register
        """
        self._prompts[prompt.full_id] = prompt
        self._metrics[prompt.full_id] = PromptMetrics()
        self._version_history[prompt.prompt_id].append(prompt.full_id)

    def get_prompt(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get a registered prompt by ID."""
        return self._prompts.get(prompt_id)

    def get_metrics(self, prompt_id: str) -> Optional[PromptMetrics]:
        """Get metrics for a prompt."""
        return self._metrics.get(prompt_id)

    def get_active_version(self, prompt_id: str) -> Optional[int]:
        """Get the active version number for a prompt.

        The active version is the latest version in the version history,
        or can be explicitly set via set_active_version().

        Args:
            prompt_id: Base prompt ID (without version suffix)

        Returns:
            Active version number, or None if prompt not found
        """
        # Check if we have an explicitly set active version
        if hasattr(self, "_active_versions") and prompt_id in self._active_versions:
            return self._active_versions[prompt_id]

        # Default to latest version
        versions = self._version_history.get(prompt_id, [])
        if not versions:
            return None

        # Get the highest version number
        max_version = 0
        for full_id in versions:
            prompt = self._prompts.get(full_id)
            if prompt and prompt.version > max_version:
                max_version = prompt.version

        return max_version if max_version > 0 else None

    def set_active_version(self, prompt_id: str, version: int) -> bool:
        """Set the active version for a prompt.

        This method designates which version of a prompt should be used
        as the current "live" version for the system.

        Args:
            prompt_id: Base prompt ID (without version suffix)
            version: Version number to set as active

        Returns:
            True if the version was successfully set, False otherwise
        """
        # Initialize active versions dict if needed
        if not hasattr(self, "_active_versions"):
            self._active_versions: Dict[str, int] = {}

        # Verify the version exists
        full_id = f"{prompt_id}_v{version}"
        if full_id not in self._prompts:
            return False

        self._active_versions[prompt_id] = version
        return True

    def get_active_prompt(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get the active prompt version.

        Args:
            prompt_id: Base prompt ID (without version suffix)

        Returns:
            Active PromptVersion, or None if not found
        """
        active_version = self.get_active_version(prompt_id)
        if active_version is None:
            return None

        full_id = f"{prompt_id}_v{active_version}"
        return self._prompts.get(full_id)

    def track_prompt_performance(
        self,
        prompt_id: str,
        success: bool,
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        syntax_valid: bool = True,
        domain: Optional[str] = None,
    ) -> None:
        """
        Record performance for a prompt use.

        Args:
            prompt_id: ID of the prompt (full_id format)
            success: Whether the generation was successful
            confidence: Confidence score of the result
            latency_ms: Latency in milliseconds
            cost_usd: Cost in USD
            syntax_valid: Whether the output had valid syntax
            domain: Optional domain context
        """
        if prompt_id not in self._metrics:
            self._metrics[prompt_id] = PromptMetrics()

        metrics = self._metrics[prompt_id]
        metrics.record(
            success=success,
            confidence=confidence,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            syntax_valid=syntax_valid,
            domain=domain,
        )

        # Record snapshot for trend analysis
        self._performance_history[prompt_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "confidence": confidence,
                "cumulative_success_rate": metrics.success_rate,
            }
        )

    def analyze_prompt_effectiveness(self, prompt_id: str) -> EffectivenessReport:
        """
        Analyze effectiveness of a prompt over time.

        Args:
            prompt_id: ID of the prompt to analyze

        Returns:
            EffectivenessReport with detailed analysis

        Raises:
            ValueError: If insufficient data for analysis
        """
        metrics = self._metrics.get(prompt_id)
        if not metrics or metrics.total_uses < self.min_samples_for_analysis:
            raise ValueError(
                f"Insufficient data for analysis. Need {self.min_samples_for_analysis} "
                f"samples, have {metrics.total_uses if metrics else 0}"
            )

        # Calculate overall effectiveness
        overall_effectiveness = metrics.success_rate

        # Calculate confidence interval using Wilson score
        confidence_interval = self._wilson_confidence_interval(
            metrics.successful_uses, metrics.total_uses
        )

        # Determine trend from history
        history = self._performance_history.get(prompt_id, [])
        trend = self._calculate_trend(history)

        # Domain breakdown
        domain_breakdown = {}
        for domain, data in metrics.domain_performance.items():
            if data["total"] > 0:
                domain_breakdown[domain] = data["success"] / data["total"]

        # Identify weaknesses and strengths
        weakness_areas = []
        strength_areas = []

        for domain, rate in domain_breakdown.items():
            if rate < overall_effectiveness - 0.1:
                weakness_areas.append(
                    f"Low performance in {domain} domain ({rate:.1%})"
                )
            elif rate > overall_effectiveness + 0.1:
                strength_areas.append(
                    f"High performance in {domain} domain ({rate:.1%})"
                )

        if metrics.syntax_validity_rate < 0.9:
            weakness_areas.append(
                f"Syntax validity issues ({metrics.syntax_validity_rate:.1%})"
            )

        if metrics.average_latency_ms > 5000:
            weakness_areas.append(
                f"High latency ({metrics.average_latency_ms:.0f}ms average)"
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_effectiveness, weakness_areas, trend
        )

        # Determine analysis period
        if history:
            start = datetime.fromisoformat(history[0]["timestamp"])
            end = datetime.fromisoformat(history[-1]["timestamp"])
        else:
            start = end = datetime.now()

        return EffectivenessReport(
            prompt_id=prompt_id,
            analysis_period_start=start,
            analysis_period_end=end,
            total_samples=metrics.total_uses,
            overall_effectiveness=overall_effectiveness,
            confidence_interval=confidence_interval,
            trend=trend,
            domain_breakdown=domain_breakdown,
            weakness_areas=weakness_areas,
            strength_areas=strength_areas,
            recommendations=recommendations,
        )

    def generate_improvement_candidates(
        self, prompt_id: str, num_candidates: int = 3
    ) -> List[ImprovementCandidate]:
        """
        Generate improvement candidates for a prompt.

        This method analyzes the prompt and its performance to
        generate potential improvements.

        Args:
            prompt_id: ID of the prompt to improve
            num_candidates: Number of candidates to generate

        Returns:
            List of ImprovementCandidate objects
        """
        prompt = self._prompts.get(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        metrics = self._metrics.get(prompt_id)
        candidates = []

        improvement_strategies = [
            self._add_clarity_instructions,
            self._add_output_structure,
            self._add_constraints,
            self._add_examples_placeholder,
        ]

        for i, strategy in enumerate(improvement_strategies[:num_candidates]):
            candidate = strategy(prompt, metrics)
            if candidate:
                candidates.append(candidate)

        return candidates

    def identify_underperforming_prompts(
        self, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Identify prompts performing below threshold.

        Args:
            threshold: Minimum acceptable success rate

        Returns:
            List of (prompt_id, success_rate) tuples for underperformers
        """
        underperformers = []

        for prompt_id, metrics in self._metrics.items():
            if (
                metrics.total_uses >= self.min_samples_for_analysis
                and metrics.success_rate < threshold
            ):
                underperformers.append((prompt_id, metrics.success_rate))

        return sorted(underperformers, key=lambda x: x[1])

    def generate_effectiveness_report(self) -> AggregateEffectivenessReport:
        """
        Generate comprehensive report aggregating all prompt performance data.

        Returns:
            AggregateEffectivenessReport with:
            - Overall metrics across all prompts
            - Per-prompt performance summaries
            - Trends over time
            - Top/bottom performing prompts
        """
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        generated_at = datetime.now()

        # Calculate overall metrics
        total_uses = sum(m.total_uses for m in self._metrics.values())
        total_successes = sum(m.successful_uses for m in self._metrics.values())
        total_confidence = sum(m.total_confidence for m in self._metrics.values())
        total_latency = sum(m.total_latency_ms for m in self._metrics.values())

        overall_success_rate = total_successes / total_uses if total_uses > 0 else 0.0
        overall_avg_confidence = (
            total_confidence / total_uses if total_uses > 0 else 0.0
        )
        overall_avg_latency = total_latency / total_uses if total_uses > 0 else 0.0

        # Build per-prompt reports
        prompt_reports: Dict[str, Dict[str, Any]] = {}
        prompt_rates: List[Tuple[str, float]] = []

        for prompt_id, metrics in self._metrics.items():
            if metrics.total_uses > 0:
                prompt_reports[prompt_id] = {
                    "total_uses": metrics.total_uses,
                    "success_rate": metrics.success_rate,
                    "average_confidence": metrics.average_confidence,
                    "average_latency_ms": metrics.average_latency_ms,
                    "syntax_validity_rate": metrics.syntax_validity_rate,
                    "domain_performance": dict(metrics.domain_performance),
                }
                prompt_rates.append((prompt_id, metrics.success_rate))

        # Sort for top/bottom performers
        sorted_rates = sorted(prompt_rates, key=lambda x: x[1], reverse=True)
        top_performers = sorted_rates[:5]
        bottom_performers = (
            sorted_rates[-5:][::-1] if len(sorted_rates) >= 5 else sorted_rates[::-1]
        )

        # Category breakdown
        category_breakdown: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total_uses": 0, "successes": 0, "success_rate": 0.0}
        )
        for prompt_id, metrics in self._metrics.items():
            prompt = self._prompts.get(prompt_id)
            if prompt and metrics.total_uses > 0:
                cat = prompt.category.value
                category_breakdown[cat]["total_uses"] += metrics.total_uses
                category_breakdown[cat]["successes"] += metrics.successful_uses

        for cat, data in category_breakdown.items():
            if data["total_uses"] > 0:
                data["success_rate"] = data["successes"] / data["total_uses"]

        # Domain breakdown
        domain_breakdown: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total_uses": 0, "successes": 0, "success_rate": 0.0}
        )
        for metrics in self._metrics.values():
            for domain, perf in metrics.domain_performance.items():
                domain_breakdown[domain]["total_uses"] += perf["total"]
                domain_breakdown[domain]["successes"] += perf["success"]

        for domain, data in domain_breakdown.items():
            if data["total_uses"] > 0:
                data["success_rate"] = data["successes"] / data["total_uses"]

        # Calculate trends
        trends: Dict[str, str] = {}
        for prompt_id in self._metrics:
            history = self._performance_history.get(prompt_id, [])
            trends[prompt_id] = self._calculate_trend(history)

        # Generate recommendations
        recommendations = self._generate_aggregate_recommendations(
            overall_success_rate,
            bottom_performers,
            category_breakdown,
            domain_breakdown,
        )

        return AggregateEffectivenessReport(
            report_id=report_id,
            generated_at=generated_at,
            total_prompts_tracked=len(self._prompts),
            total_uses_recorded=total_uses,
            overall_success_rate=overall_success_rate,
            overall_average_confidence=overall_avg_confidence,
            overall_average_latency_ms=overall_avg_latency,
            prompt_reports=prompt_reports,
            top_performers=top_performers,
            bottom_performers=bottom_performers,
            category_breakdown=dict(category_breakdown),
            domain_breakdown=dict(domain_breakdown),
            trends=trends,
            recommendations=recommendations,
        )

    def suggest_improvements(self, prompt_id: str) -> List[ImprovementSuggestion]:
        """
        Generate specific improvement suggestions for a prompt.

        Analyzes the prompt's performance metrics and generates actionable
        suggestions for improving its effectiveness.

        Args:
            prompt_id: The ID of the prompt to analyze

        Returns:
            List of ImprovementSuggestion objects with:
            - Suggested modification
            - Rationale
            - Expected improvement percentage
            - Confidence score
        """
        prompt = self._prompts.get(prompt_id)
        metrics = self._metrics.get(prompt_id)

        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        suggestions: List[ImprovementSuggestion] = []

        # If no metrics yet, suggest collecting more data
        if not metrics or metrics.total_uses < self.min_samples_for_analysis:
            suggestions.append(
                ImprovementSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    prompt_id=prompt_id,
                    suggestion_type="data_collection",
                    description="Collect more usage data before optimization",
                    rationale=f"Only {metrics.total_uses if metrics else 0} samples collected, "
                    f"need at least {self.min_samples_for_analysis} for reliable analysis",
                    expected_improvement_percentage=0.0,
                    confidence=0.9,
                    priority="high",
                    implementation_hint="Use this prompt in more scenarios to gather performance data",
                )
            )
            return suggestions

        # Analyze success rate
        if metrics.success_rate < 0.7:
            suggestions.append(
                ImprovementSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    prompt_id=prompt_id,
                    suggestion_type="wording",
                    description="Rewrite prompt with clearer instructions",
                    rationale=f"Success rate is {metrics.success_rate:.1%}, below 70% threshold",
                    expected_improvement_percentage=15.0,
                    confidence=0.7,
                    priority="high",
                    implementation_hint="Add explicit step-by-step instructions and expected output format",
                )
            )

        # Analyze syntax validity
        if metrics.syntax_validity_rate < 0.9:
            suggestions.append(
                ImprovementSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    prompt_id=prompt_id,
                    suggestion_type="constraints",
                    description="Add output format constraints to reduce syntax errors",
                    rationale=f"Syntax validity is {metrics.syntax_validity_rate:.1%}, "
                    f"{(1 - metrics.syntax_validity_rate) * 100:.0f}% of outputs have syntax issues",
                    expected_improvement_percentage=10.0,
                    confidence=0.8,
                    priority="high",
                    implementation_hint="Add explicit format requirements: 'Output must be valid JSON/ASP/etc.'",
                )
            )

        # Analyze domain-specific performance
        for domain, perf in metrics.domain_performance.items():
            if perf["total"] >= 5:
                domain_rate = perf["success"] / perf["total"]
                if domain_rate < metrics.success_rate - 0.15:
                    suggestions.append(
                        ImprovementSuggestion(
                            suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                            prompt_id=prompt_id,
                            suggestion_type="domain",
                            description=f"Add domain-specific guidance for '{domain}'",
                            rationale=f"Performance in {domain} ({domain_rate:.1%}) is significantly "
                            f"below average ({metrics.success_rate:.1%})",
                            expected_improvement_percentage=12.0,
                            confidence=0.65,
                            priority="medium",
                            implementation_hint=f"Include {domain}-specific examples or terminology",
                        )
                    )

        # Analyze latency
        if metrics.average_latency_ms > 5000:
            suggestions.append(
                ImprovementSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    prompt_id=prompt_id,
                    suggestion_type="structure",
                    description="Simplify prompt to reduce latency",
                    rationale=f"Average latency is {metrics.average_latency_ms:.0f}ms, "
                    f"which may indicate overly complex prompts",
                    expected_improvement_percentage=5.0,
                    confidence=0.5,
                    priority="low",
                    implementation_hint="Remove redundant instructions and use more concise wording",
                )
            )

        # Analyze confidence scores
        if metrics.average_confidence < 0.7:
            suggestions.append(
                ImprovementSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    prompt_id=prompt_id,
                    suggestion_type="examples",
                    description="Add few-shot examples to improve confidence",
                    rationale=f"Average confidence is {metrics.average_confidence:.1%}, "
                    f"examples can help the model understand expected output",
                    expected_improvement_percentage=8.0,
                    confidence=0.75,
                    priority="medium",
                    implementation_hint="Include 2-3 input/output examples demonstrating the expected format",
                )
            )

        # Check trend
        history = self._performance_history.get(prompt_id, [])
        trend = self._calculate_trend(history)
        if trend == "degrading":
            suggestions.append(
                ImprovementSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    prompt_id=prompt_id,
                    suggestion_type="investigation",
                    description="Investigate recent performance degradation",
                    rationale="Performance trend shows degradation over recent samples",
                    expected_improvement_percentage=0.0,
                    confidence=0.6,
                    priority="high",
                    implementation_hint="Review recent failed cases to identify patterns or data drift",
                )
            )

        # If no issues found, suggest A/B testing
        if not suggestions:
            suggestions.append(
                ImprovementSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    prompt_id=prompt_id,
                    suggestion_type="testing",
                    description="Consider A/B testing prompt variations",
                    rationale=f"Prompt is performing well ({metrics.success_rate:.1%}), "
                    f"but may benefit from optimization",
                    expected_improvement_percentage=5.0,
                    confidence=0.4,
                    priority="low",
                    implementation_hint="Try variations with different wording, structure, or examples",
                )
            )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 1))

        return suggestions

    def _generate_aggregate_recommendations(
        self,
        overall_rate: float,
        bottom_performers: List[Tuple[str, float]],
        category_breakdown: Dict[str, Dict[str, float]],
        domain_breakdown: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """Generate recommendations for aggregate report."""
        recommendations = []

        # Overall performance
        if overall_rate < 0.7:
            recommendations.append(
                f"Overall success rate ({overall_rate:.1%}) is below target. "
                "Consider systematic prompt review."
            )
        elif overall_rate > 0.85:
            recommendations.append(
                f"Overall success rate ({overall_rate:.1%}) is excellent. "
                "Focus on optimizing underperformers."
            )

        # Bottom performers
        if bottom_performers:
            worst_prompt, worst_rate = bottom_performers[0]
            if worst_rate < 0.5:
                recommendations.append(
                    f"Prompt '{worst_prompt}' has very low success rate ({worst_rate:.1%}). "
                    "Prioritize for improvement."
                )

        # Category issues
        for cat, data in category_breakdown.items():
            if data.get("success_rate", 0) < 0.6 and data.get("total_uses", 0) >= 10:
                recommendations.append(
                    f"Category '{cat}' prompts underperforming ({data['success_rate']:.1%}). "
                    "Review category-specific patterns."
                )

        # Domain issues
        for domain, data in domain_breakdown.items():
            if data.get("success_rate", 0) < 0.6 and data.get("total_uses", 0) >= 10:
                recommendations.append(
                    f"Domain '{domain}' shows low performance ({data['success_rate']:.1%}). "
                    "Consider domain-specific prompt tuning."
                )

        if not recommendations:
            recommendations.append("System is performing well. Continue monitoring.")

        return recommendations

    def get_version_history(self, prompt_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for a prompt.

        Args:
            prompt_id: Base prompt ID (without version)

        Returns:
            List of version information dictionaries
        """
        versions = self._version_history.get(prompt_id, [])
        history = []

        for full_id in versions:
            prompt = self._prompts.get(full_id)
            metrics = self._metrics.get(full_id)

            if prompt and metrics:
                history.append(
                    {
                        "full_id": full_id,
                        "version": prompt.version,
                        "created_at": prompt.created_at.isoformat(),
                        "modification_reason": prompt.modification_reason,
                        "success_rate": metrics.success_rate,
                        "total_uses": metrics.total_uses,
                    }
                )

        return history

    def create_new_version(
        self,
        original_prompt_id: str,
        new_template: str,
        modification_reason: str,
    ) -> PromptVersion:
        """
        Create a new version of a prompt.

        Args:
            original_prompt_id: ID of the original prompt
            new_template: New template string
            modification_reason: Why this version was created

        Returns:
            New PromptVersion
        """
        original = self._prompts.get(original_prompt_id)
        if not original:
            raise ValueError(f"Original prompt {original_prompt_id} not found")

        new_version = PromptVersion(
            prompt_id=original.prompt_id,
            version=original.version + 1,
            template=new_template,
            category=original.category,
            parent_version=original.full_id,
            modification_reason=modification_reason,
        )

        self.register_prompt(new_version)
        return new_version

    def _wilson_confidence_interval(
        self, successes: int, total: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if total == 0:
            return (0.0, 0.0)

        z = 1.96 if confidence == 0.95 else 1.645  # z-score for confidence level
        p_hat = successes / total

        denominator = 1 + z**2 / total
        center = p_hat + z**2 / (2 * total)
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)

        lower = max(0.0, (center - spread) / denominator)
        upper = min(1.0, (center + spread) / denominator)

        return (lower, upper)

    def _calculate_trend(self, history: List[Dict[str, Any]]) -> str:
        """Calculate performance trend from history."""
        if len(history) < 5:
            return "stable"

        # Compare recent performance to earlier performance
        mid_point = len(history) // 2
        early_successes = sum(1 for h in history[:mid_point] if h["success"])
        late_successes = sum(1 for h in history[mid_point:] if h["success"])

        early_rate = early_successes / mid_point if mid_point > 0 else 0
        late_rate = (
            late_successes / (len(history) - mid_point)
            if len(history) > mid_point
            else 0
        )

        if late_rate > early_rate + 0.05:
            return "improving"
        elif late_rate < early_rate - 0.05:
            return "degrading"
        else:
            return "stable"

    def _generate_recommendations(
        self, effectiveness: float, weaknesses: List[str], trend: str
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if effectiveness < 0.7:
            recommendations.append(
                "Consider major prompt restructuring - effectiveness is below 70%"
            )
        elif effectiveness < 0.85:
            recommendations.append(
                "Prompt has room for improvement - target 85%+ effectiveness"
            )

        if weaknesses:
            recommendations.append(
                f"Address identified weaknesses: {'; '.join(weaknesses[:2])}"
            )

        if trend == "degrading":
            recommendations.append(
                "Performance is degrading - investigate recent changes or data drift"
            )
        elif trend == "improving":
            recommendations.append(
                "Performance is improving - continue current optimization approach"
            )

        if not recommendations:
            recommendations.append(
                "Prompt is performing well - consider A/B testing variations"
            )

        return recommendations

    def _add_clarity_instructions(
        self, prompt: PromptVersion, metrics: Optional[PromptMetrics]
    ) -> ImprovementCandidate:
        """Generate candidate with clearer instructions."""
        clarity_prefix = "Please follow these instructions carefully and precisely.\n\n"

        new_template = clarity_prefix + prompt.template

        return ImprovementCandidate(
            candidate_id=f"clarity_{uuid.uuid4().hex[:8]}",
            original_prompt_id=prompt.full_id,
            new_template=new_template,
            improvement_type="clarification",
            expected_improvement=0.05,
            rationale="Added explicit instruction for careful following of directions",
        )

    def _add_output_structure(
        self, prompt: PromptVersion, metrics: Optional[PromptMetrics]
    ) -> ImprovementCandidate:
        """Generate candidate with structured output instructions."""
        structure_suffix = (
            "\n\nProvide your response in a structured format. "
            "Be specific and precise in your output."
        )

        new_template = prompt.template + structure_suffix

        return ImprovementCandidate(
            candidate_id=f"structure_{uuid.uuid4().hex[:8]}",
            original_prompt_id=prompt.full_id,
            new_template=new_template,
            improvement_type="structure",
            expected_improvement=0.08,
            rationale="Added structured output instructions to improve format consistency",
        )

    def _add_constraints(
        self, prompt: PromptVersion, metrics: Optional[PromptMetrics]
    ) -> ImprovementCandidate:
        """Generate candidate with additional constraints."""
        constraint_suffix = (
            "\n\nConstraints:\n"
            "- Follow the specified format exactly\n"
            "- Do not include explanations unless requested\n"
            "- Ensure output is syntactically valid"
        )

        new_template = prompt.template + constraint_suffix

        return ImprovementCandidate(
            candidate_id=f"constraints_{uuid.uuid4().hex[:8]}",
            original_prompt_id=prompt.full_id,
            new_template=new_template,
            improvement_type="constraints",
            expected_improvement=0.10,
            rationale="Added explicit constraints to reduce invalid outputs",
        )

    def _add_examples_placeholder(
        self, prompt: PromptVersion, metrics: Optional[PromptMetrics]
    ) -> ImprovementCandidate:
        """Generate candidate with example placeholder."""
        example_section = (
            "\n\nExample:\n"
            "Input: [example input]\n"
            "Output: [example output]\n\n"
            "Now process the following:"
        )

        # Insert before the last part of the prompt
        new_template = prompt.template + example_section

        return ImprovementCandidate(
            candidate_id=f"examples_{uuid.uuid4().hex[:8]}",
            original_prompt_id=prompt.full_id,
            new_template=new_template,
            improvement_type="examples",
            expected_improvement=0.12,
            rationale="Added example section to demonstrate expected format",
        )


class PromptABTester:
    """
    A/B testing framework for prompt variations.

    Provides statistical comparison of prompt versions
    with significance testing and automatic winner selection.
    """

    def __init__(
        self,
        optimizer: PromptOptimizer,
        min_samples_per_variant: int = 30,
        significance_level: float = 0.05,
    ):
        """
        Initialize the A/B tester.

        Args:
            optimizer: PromptOptimizer instance for tracking
            min_samples_per_variant: Minimum samples needed per variant
            significance_level: P-value threshold for significance
        """
        self.optimizer = optimizer
        self.min_samples_per_variant = min_samples_per_variant
        self.significance_level = significance_level

        # Active tests
        self._active_tests: Dict[str, ABTestConfig] = {}
        self._test_results: Dict[str, ABTestResult] = {}

    def create_test(
        self,
        prompt_a: PromptVersion,
        prompt_b: PromptVersion,
        allocation_ratio: float = 0.5,
    ) -> ABTestConfig:
        """
        Create a new A/B test.

        Args:
            prompt_a: Control prompt version
            prompt_b: Treatment prompt version
            allocation_ratio: Traffic allocation to variant B

        Returns:
            ABTestConfig for the test
        """
        test_id = f"test_{uuid.uuid4().hex[:8]}"

        config = ABTestConfig(
            test_id=test_id,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            min_samples_per_variant=self.min_samples_per_variant,
            significance_level=self.significance_level,
            allocation_ratio=allocation_ratio,
        )

        self._active_tests[test_id] = config

        # Ensure prompts are registered
        self.optimizer.register_prompt(prompt_a)
        self.optimizer.register_prompt(prompt_b)

        return config

    def get_variant(self, test_id: str) -> PromptVersion:
        """
        Get a variant for a test using allocation ratio.

        Args:
            test_id: ID of the test

        Returns:
            Selected prompt variant

        Raises:
            ValueError: If test not found
        """
        config = self._active_tests.get(test_id)
        if not config:
            raise ValueError(f"Test {test_id} not found")

        if random.random() < config.allocation_ratio:
            return config.prompt_b
        return config.prompt_a

    def record_outcome(
        self,
        test_id: str,
        prompt_id: str,
        success: bool,
        confidence: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Record an outcome for a test.

        Args:
            test_id: ID of the test
            prompt_id: ID of the prompt used
            success: Whether the outcome was successful
            confidence: Confidence score
            latency_ms: Latency in milliseconds
        """
        if test_id not in self._active_tests:
            raise ValueError(f"Test {test_id} not found")

        self.optimizer.track_prompt_performance(
            prompt_id=prompt_id,
            success=success,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    def check_test_completion(self, test_id: str) -> bool:
        """
        Check if a test has enough samples to conclude.

        Args:
            test_id: ID of the test

        Returns:
            True if test can be concluded
        """
        config = self._active_tests.get(test_id)
        if not config:
            return False

        metrics_a = self.optimizer.get_metrics(config.prompt_a.full_id)
        metrics_b = self.optimizer.get_metrics(config.prompt_b.full_id)

        if not metrics_a or not metrics_b:
            return False

        return (
            metrics_a.total_uses >= config.min_samples_per_variant
            and metrics_b.total_uses >= config.min_samples_per_variant
        )

    def analyze_test(self, test_id: str) -> ABTestResult:
        """
        Analyze an A/B test and determine winner.

        Args:
            test_id: ID of the test

        Returns:
            ABTestResult with statistical analysis

        Raises:
            ValueError: If test not found or insufficient data
        """
        config = self._active_tests.get(test_id)
        if not config:
            raise ValueError(f"Test {test_id} not found")

        metrics_a = self.optimizer.get_metrics(config.prompt_a.full_id)
        metrics_b = self.optimizer.get_metrics(config.prompt_b.full_id)

        if not metrics_a or not metrics_b:
            raise ValueError("Metrics not available for test variants")

        # Calculate z-test for proportions
        p_value, effect_size = self._two_proportion_z_test(
            metrics_a.successful_uses,
            metrics_a.total_uses,
            metrics_b.successful_uses,
            metrics_b.total_uses,
        )

        is_significant = p_value < config.significance_level

        # Determine winner
        winner = None
        if is_significant:
            if metrics_b.success_rate > metrics_a.success_rate:
                winner = config.prompt_b.full_id
            else:
                winner = config.prompt_a.full_id

        # Calculate confidence interval for difference
        diff_ci = self._difference_confidence_interval(
            metrics_a.successful_uses,
            metrics_a.total_uses,
            metrics_b.successful_uses,
            metrics_b.total_uses,
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            metrics_a, metrics_b, is_significant, winner
        )

        result = ABTestResult(
            test_id=test_id,
            prompt_a_id=config.prompt_a.full_id,
            prompt_b_id=config.prompt_b.full_id,
            prompt_a_metrics=metrics_a,
            prompt_b_metrics=metrics_b,
            winner=winner,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=diff_ci,
            is_significant=is_significant,
            recommendation=recommendation,
        )

        self._test_results[test_id] = result
        return result

    def select_winner(self, test_id: str) -> Optional[PromptVersion]:
        """
        Select the winning prompt from a test.

        Args:
            test_id: ID of the test

        Returns:
            Winning PromptVersion or None if no significant difference
        """
        if test_id not in self._test_results:
            self.analyze_test(test_id)

        result = self._test_results.get(test_id)
        if not result or not result.winner:
            return None

        return self.optimizer.get_prompt(result.winner)

    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get list of active tests with status."""
        tests = []
        for test_id, config in self._active_tests.items():
            metrics_a = self.optimizer.get_metrics(config.prompt_a.full_id)
            metrics_b = self.optimizer.get_metrics(config.prompt_b.full_id)

            tests.append(
                {
                    "test_id": test_id,
                    "prompt_a": config.prompt_a.full_id,
                    "prompt_b": config.prompt_b.full_id,
                    "samples_a": metrics_a.total_uses if metrics_a else 0,
                    "samples_b": metrics_b.total_uses if metrics_b else 0,
                    "can_conclude": self.check_test_completion(test_id),
                }
            )

        return tests

    def _two_proportion_z_test(
        self, successes_a: int, total_a: int, successes_b: int, total_b: int
    ) -> Tuple[float, float]:
        """
        Perform two-proportion z-test.

        Returns:
            Tuple of (p_value, effect_size)
        """
        if total_a == 0 or total_b == 0:
            return (1.0, 0.0)

        p_a = successes_a / total_a
        p_b = successes_b / total_b

        # Pooled proportion
        p_pool = (successes_a + successes_b) / (total_a + total_b)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))

        if se == 0:
            return (1.0, 0.0)

        # Z statistic
        z = (p_b - p_a) / se

        # Two-tailed p-value (using normal approximation)
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        # Effect size (Cohen's h)
        effect_size = abs(2 * math.asin(math.sqrt(p_b)) - 2 * math.asin(math.sqrt(p_a)))

        return (p_value, effect_size)

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function."""
        return (1 + math.erf(x / math.sqrt(2))) / 2

    def _difference_confidence_interval(
        self, successes_a: int, total_a: int, successes_b: int, total_b: int
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference in proportions."""
        if total_a == 0 or total_b == 0:
            return (0.0, 0.0)

        p_a = successes_a / total_a
        p_b = successes_b / total_b
        diff = p_b - p_a

        # Standard error of difference
        se = math.sqrt(p_a * (1 - p_a) / total_a + p_b * (1 - p_b) / total_b)

        z = 1.96  # 95% confidence
        return (diff - z * se, diff + z * se)

    def _generate_recommendation(
        self,
        metrics_a: PromptMetrics,
        metrics_b: PromptMetrics,
        is_significant: bool,
        winner: Optional[str],
    ) -> str:
        """Generate recommendation based on test results."""
        if not is_significant:
            return (
                "No significant difference detected. Consider extending the test "
                "or testing a more different variation."
            )

        improvement = abs(metrics_b.success_rate - metrics_a.success_rate)

        if winner and "b" in winner.lower():
            return (
                f"Variant B shows significant improvement ({improvement:.1%}). "
                f"Recommend adopting variant B as the new default."
            )
        else:
            return (
                f"Variant A performs significantly better ({improvement:.1%}). "
                f"Recommend keeping variant A."
            )


# Factory functions


def create_prompt_optimizer(
    improvement_threshold: float = 0.15,
    min_samples: int = 10,
) -> PromptOptimizer:
    """
    Create a PromptOptimizer instance.

    Args:
        improvement_threshold: Target improvement percentage
        min_samples: Minimum samples for analysis

    Returns:
        Configured PromptOptimizer
    """
    return PromptOptimizer(
        improvement_threshold=improvement_threshold,
        min_samples_for_analysis=min_samples,
    )


def create_ab_tester(
    optimizer: Optional[PromptOptimizer] = None,
    min_samples: int = 30,
    significance_level: float = 0.05,
) -> PromptABTester:
    """
    Create a PromptABTester instance.

    Args:
        optimizer: Optional PromptOptimizer (creates new if None)
        min_samples: Minimum samples per variant
        significance_level: P-value threshold

    Returns:
        Configured PromptABTester
    """
    if optimizer is None:
        optimizer = create_prompt_optimizer()

    return PromptABTester(
        optimizer=optimizer,
        min_samples_per_variant=min_samples,
        significance_level=significance_level,
    )
