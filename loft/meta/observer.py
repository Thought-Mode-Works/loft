"""
Meta-reasoning observer for pattern observation and analysis.

Provides the ReasoningObserver for tracking reasoning chains and
MetaReasoner for second-order reasoning about reasoning processes.
"""

import uuid
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loft.meta.schemas import (
    Bottleneck,
    BottleneckReport,
    FailureDiagnosis,
    Improvement,
    ImprovementPriority,
    ImprovementType,
    ObservationSummary,
    PatternType,
    ReasoningChain,
    ReasoningPattern,
    ReasoningStep,
    ReasoningStepType,
)


class ReasoningObserver:
    """Observes and records reasoning patterns.

    Monitors reasoning pipeline execution, tracks which patterns succeed/fail,
    and identifies recurring patterns across cases.
    """

    def __init__(
        self,
        observer_id: Optional[str] = None,
        enable_detailed_tracking: bool = True,
    ):
        """Initialize the reasoning observer.

        Args:
            observer_id: Unique identifier for this observer instance
            enable_detailed_tracking: Whether to store detailed step data
        """
        self.observer_id = observer_id or f"observer_{uuid.uuid4().hex[:8]}"
        self.enable_detailed_tracking = enable_detailed_tracking

        self.chains: Dict[str, ReasoningChain] = {}
        self.patterns: Dict[str, ReasoningPattern] = {}
        self.started_at = datetime.now()

        # Aggregated statistics
        self._step_durations: Dict[ReasoningStepType, List[float]] = defaultdict(list)
        self._step_success_rates: Dict[ReasoningStepType, List[bool]] = defaultdict(list)
        self._domain_outcomes: Dict[str, List[bool]] = defaultdict(list)

        # Callbacks for real-time observation
        self._on_chain_complete: Optional[Callable[[ReasoningChain], None]] = None
        self._on_pattern_discovered: Optional[Callable[[ReasoningPattern], None]] = None

    def set_callbacks(
        self,
        on_chain_complete: Optional[Callable[[ReasoningChain], None]] = None,
        on_pattern_discovered: Optional[Callable[[ReasoningPattern], None]] = None,
    ) -> None:
        """Set callbacks for real-time observation.

        Args:
            on_chain_complete: Called when a reasoning chain completes
            on_pattern_discovered: Called when a new pattern is discovered
        """
        self._on_chain_complete = on_chain_complete
        self._on_pattern_discovered = on_pattern_discovered

    def observe_reasoning_chain(
        self,
        case_id: str,
        domain: str,
        steps: List[ReasoningStep],
        prediction: Optional[str] = None,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningChain:
        """Track a full reasoning chain for a case.

        Args:
            case_id: Identifier for the case
            domain: Legal domain of the case
            steps: List of reasoning steps in the chain
            prediction: System's prediction
            ground_truth: Actual correct answer
            metadata: Additional metadata

        Returns:
            The recorded reasoning chain
        """
        chain_id = f"chain_{uuid.uuid4().hex[:8]}"
        overall_success = prediction == ground_truth if ground_truth else False

        started_at = steps[0].started_at if steps else datetime.now()
        completed_at = steps[-1].completed_at if steps else datetime.now()

        chain = ReasoningChain(
            chain_id=chain_id,
            case_id=case_id,
            domain=domain,
            steps=steps,
            prediction=prediction,
            ground_truth=ground_truth,
            overall_success=overall_success,
            started_at=started_at,
            completed_at=completed_at,
            metadata=metadata or {},
        )

        self.chains[chain_id] = chain

        # Update aggregated statistics
        self._update_statistics(chain)

        if self._on_chain_complete:
            self._on_chain_complete(chain)

        return chain

    def _update_statistics(self, chain: ReasoningChain) -> None:
        """Update aggregated statistics with a new chain.

        Args:
            chain: The reasoning chain to incorporate
        """
        for step in chain.steps:
            self._step_durations[step.step_type].append(step.duration_ms)
            self._step_success_rates[step.step_type].append(step.success)

        self._domain_outcomes[chain.domain].append(chain.overall_success)

    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Retrieve a specific reasoning chain.

        Args:
            chain_id: The chain identifier

        Returns:
            The reasoning chain or None if not found
        """
        return self.chains.get(chain_id)

    def get_chain_id(self, chain_or_id: Union[str, ReasoningChain]) -> str:
        """Extract chain ID from a chain object or return the string as-is.

        Args:
            chain_or_id: Either a chain_id string or a ReasoningChain object

        Returns:
            The chain ID as a string

        Raises:
            TypeError: If the input is neither a string nor a ReasoningChain
        """
        if isinstance(chain_or_id, str):
            return chain_or_id
        elif isinstance(chain_or_id, ReasoningChain):
            return chain_or_id.chain_id
        else:
            raise TypeError(f"Expected str or ReasoningChain, got {type(chain_or_id).__name__}")

    def resolve_chain(self, chain_or_id: Union[str, ReasoningChain]) -> Optional[ReasoningChain]:
        """Resolve a chain object or ID to a ReasoningChain.

        Accepts either a chain_id string or a ReasoningChain object and returns
        the corresponding ReasoningChain from storage (if found).

        Args:
            chain_or_id: Either a chain_id string or a ReasoningChain object

        Returns:
            The ReasoningChain if found, None otherwise
        """
        if isinstance(chain_or_id, ReasoningChain):
            # If it's already a chain, return it directly if in storage
            # or just return it (for chains not yet stored)
            return self.chains.get(chain_or_id.chain_id, chain_or_id)
        elif isinstance(chain_or_id, str):
            return self.chains.get(chain_or_id)
        else:
            return None

    def get_chains_by_domain(self, domain: str) -> List[ReasoningChain]:
        """Get all chains for a specific domain.

        Args:
            domain: The domain to filter by

        Returns:
            List of chains in that domain
        """
        return [c for c in self.chains.values() if c.domain == domain]

    def get_failed_chains(self) -> List[ReasoningChain]:
        """Get all chains that resulted in incorrect predictions.

        Returns:
            List of failed chains
        """
        return [c for c in self.chains.values() if not c.overall_success]

    def identify_patterns(
        self,
        min_frequency: int = 3,
        min_correlation: float = 0.3,
    ) -> List[ReasoningPattern]:
        """Identify recurring patterns across observed cases.

        Args:
            min_frequency: Minimum occurrences to consider a pattern
            min_correlation: Minimum success correlation to include

        Returns:
            List of identified patterns
        """
        patterns = []

        # Pattern 1: Step type success patterns
        for step_type, successes in self._step_success_rates.items():
            if len(successes) < min_frequency:
                continue

            success_rate = sum(successes) / len(successes)
            # Convert to correlation (-1 to 1 scale)
            correlation = (success_rate - 0.5) * 2

            if abs(correlation) >= min_correlation:
                pattern = ReasoningPattern(
                    pattern_id=f"pattern_step_{step_type.value}",
                    pattern_type=(PatternType.SUCCESS if correlation > 0 else PatternType.FAILURE),
                    name=f"{step_type.value}_pattern",
                    description=(
                        f"Steps of type '{step_type.value}' have {success_rate:.1%} success rate"
                    ),
                    frequency=len(successes),
                    associated_step_types=[step_type],
                    success_correlation=correlation,
                    characteristics={"success_rate": success_rate},
                )
                patterns.append(pattern)
                self.patterns[pattern.pattern_id] = pattern

        # Pattern 2: Domain-specific patterns
        for domain, outcomes in self._domain_outcomes.items():
            if len(outcomes) < min_frequency:
                continue

            success_rate = sum(outcomes) / len(outcomes)
            correlation = (success_rate - 0.5) * 2

            if abs(correlation) >= min_correlation:
                pattern = ReasoningPattern(
                    pattern_id=f"pattern_domain_{domain}",
                    pattern_type=PatternType.DOMAIN_SPECIFIC,
                    name=f"{domain}_success_pattern",
                    description=(f"Domain '{domain}' has {success_rate:.1%} success rate"),
                    frequency=len(outcomes),
                    associated_step_types=[],
                    success_correlation=correlation,
                    domains=[domain],
                    characteristics={"success_rate": success_rate},
                )
                patterns.append(pattern)
                self.patterns[pattern.pattern_id] = pattern

        # Pattern 3: Failure mode patterns
        failure_steps = self._analyze_failure_steps()
        for step_type, count in failure_steps.items():
            if count >= min_frequency:
                pattern = ReasoningPattern(
                    pattern_id=f"pattern_failure_{step_type.value}",
                    pattern_type=PatternType.FAILURE,
                    name=f"{step_type.value}_failure_pattern",
                    description=(f"Failures frequently occur at '{step_type.value}' step"),
                    frequency=count,
                    associated_step_types=[step_type],
                    success_correlation=-0.8,
                    characteristics={"failure_count": count},
                )
                patterns.append(pattern)
                self.patterns[pattern.pattern_id] = pattern

        # Pattern 4: Duration-based bottleneck patterns
        bottleneck_patterns = self._identify_bottleneck_patterns(min_frequency)
        patterns.extend(bottleneck_patterns)

        # Pattern 5: Error-message-based failure patterns
        error_patterns = self._identify_error_patterns(min_frequency)
        patterns.extend(error_patterns)

        # Notify callbacks for new patterns
        if self._on_pattern_discovered:
            for pattern in patterns:
                if pattern.pattern_id not in self.patterns:
                    self._on_pattern_discovered(pattern)

        return patterns

    def _analyze_failure_steps(self) -> Dict[ReasoningStepType, int]:
        """Analyze which step types are associated with failures.

        Returns:
            Dictionary mapping step types to failure counts
        """
        failure_counts: Dict[ReasoningStepType, int] = defaultdict(int)

        for chain in self.chains.values():
            if not chain.overall_success:
                # Count failed steps in failed chains
                for step in chain.steps:
                    if not step.success:
                        failure_counts[step.step_type] += 1

        return failure_counts

    def _identify_bottleneck_patterns(self, min_frequency: int) -> List[ReasoningPattern]:
        """Identify patterns related to performance bottlenecks.

        Args:
            min_frequency: Minimum occurrences to consider

        Returns:
            List of bottleneck patterns
        """
        patterns = []

        for step_type, durations in self._step_durations.items():
            if len(durations) < min_frequency:
                continue

            avg_duration = mean(durations)
            # Consider it a bottleneck if average is > 1000ms
            if avg_duration > 1000:
                pattern = ReasoningPattern(
                    pattern_id=f"pattern_bottleneck_{step_type.value}",
                    pattern_type=PatternType.BOTTLENECK,
                    name=f"{step_type.value}_bottleneck",
                    description=(f"Step type '{step_type.value}' averages {avg_duration:.0f}ms"),
                    frequency=len(durations),
                    associated_step_types=[step_type],
                    success_correlation=0.0,
                    characteristics={
                        "avg_duration_ms": avg_duration,
                        "max_duration_ms": max(durations),
                    },
                )
                patterns.append(pattern)
                self.patterns[pattern.pattern_id] = pattern

        return patterns

    def _identify_error_patterns(self, min_frequency: int) -> List[ReasoningPattern]:
        """Identify failure patterns based on error message similarity.

        Groups failures by error message content and step type to find
        recurring failure modes that may indicate systematic issues.

        Args:
            min_frequency: Minimum occurrences to consider (uses min of 1 for
                single-chain detection support)

        Returns:
            List of error-based failure patterns
        """
        patterns = []
        error_groups = self._group_failures_by_error()

        # Use min of 1 for error patterns to support single-chain detection
        # when the error is distinct enough to warrant a pattern
        effective_min = min(min_frequency, 1)

        for error_key, failure_data in error_groups.items():
            if failure_data["count"] < effective_min:
                continue

            step_type, error_message = error_key
            affected_domains = failure_data["domains"]
            chain_ids = failure_data["chain_ids"]

            # Create a sanitized pattern ID from the error message
            error_slug = self._create_error_slug(error_message)
            pattern_id = f"pattern_error_{step_type.value}_{error_slug}"

            # Skip if already exists
            if pattern_id in self.patterns:
                continue

            pattern = ReasoningPattern(
                pattern_id=pattern_id,
                pattern_type=PatternType.FAILURE,
                name=f"{step_type.value}_error_pattern",
                description=(f"Recurring error in '{step_type.value}': {error_message}"),
                frequency=failure_data["count"],
                associated_step_types=[step_type],
                success_correlation=-0.9,
                domains=list(affected_domains),
                characteristics={
                    "error_message": error_message,
                    "affected_chain_ids": chain_ids,
                    "failure_count": failure_data["count"],
                },
            )
            patterns.append(pattern)
            self.patterns[pattern_id] = pattern

        return patterns

    def _group_failures_by_error(
        self,
    ) -> Dict[Tuple[ReasoningStepType, str], Dict[str, Any]]:
        """Group failed steps by their error messages and step types.

        Returns:
            Dictionary mapping (step_type, error_message) to failure data
            containing count, affected domains, and chain IDs.
        """
        error_groups: Dict[tuple, Dict[str, Any]] = {}

        for chain in self.chains.values():
            if not chain.overall_success:
                for step in chain.steps:
                    if not step.success and step.error_message:
                        key = (step.step_type, step.error_message)
                        if key not in error_groups:
                            error_groups[key] = {
                                "count": 0,
                                "domains": set(),
                                "chain_ids": [],
                            }
                        error_groups[key]["count"] += 1
                        error_groups[key]["domains"].add(chain.domain)
                        error_groups[key]["chain_ids"].append(chain.chain_id)

        return error_groups

    def _create_error_slug(self, error_message: str) -> str:
        """Create a URL-safe slug from an error message.

        Args:
            error_message: The error message to slugify

        Returns:
            A shortened, sanitized version suitable for pattern IDs
        """
        # Take first 30 chars, lowercase, replace spaces with underscores
        slug = error_message[:30].lower().replace(" ", "_")
        # Remove non-alphanumeric chars except underscores
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        return slug

    def analyze_bottlenecks(
        self,
        threshold_percentage: float = 20.0,
    ) -> BottleneckReport:
        """Identify reasoning bottlenecks from observations.

        Args:
            threshold_percentage: Minimum percentage of total time to be a bottleneck

        Returns:
            Report of identified bottlenecks
        """
        bottlenecks = []
        total_time = sum(sum(durations) for durations in self._step_durations.values())

        if total_time == 0:
            return BottleneckReport(
                report_id=f"bottleneck_{uuid.uuid4().hex[:8]}",
                generated_at=datetime.now(),
                total_chains_analyzed=len(self.chains),
                total_steps_analyzed=sum(len(c.steps) for c in self.chains.values()),
                bottlenecks=[],
                recommendations=["Insufficient data for bottleneck analysis"],
            )

        # Calculate time contribution per step type
        time_by_type: List[Dict[str, Any]] = []
        for step_type, durations in self._step_durations.items():
            type_total = sum(durations)
            percentage = (type_total / total_time) * 100

            time_by_type.append(
                {
                    "step_type": step_type.value,
                    "total_time_ms": type_total,
                    "percentage": percentage,
                    "avg_duration_ms": mean(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "count": len(durations),
                }
            )

            if percentage >= threshold_percentage:
                # Determine affected domains
                affected_domains = set()
                for chain in self.chains.values():
                    for step in chain.steps:
                        if step.step_type == step_type:
                            affected_domains.add(chain.domain)

                bottleneck = Bottleneck(
                    bottleneck_id=f"bottleneck_{step_type.value}",
                    step_type=step_type,
                    description=(f"{step_type.value} consumes {percentage:.1f}% of total time"),
                    avg_duration_ms=mean(durations),
                    max_duration_ms=max(durations),
                    occurrence_count=len(durations),
                    percentage_of_total_time=percentage,
                    affected_domains=list(affected_domains),
                    potential_causes=self._infer_bottleneck_causes(step_type),
                    severity="high" if percentage > 40 else "medium",
                )
                bottlenecks.append(bottleneck)

        # Sort time_by_type by percentage descending
        time_by_type.sort(key=lambda x: x["percentage"], reverse=True)

        # Generate recommendations
        recommendations = self._generate_bottleneck_recommendations(bottlenecks)

        return BottleneckReport(
            report_id=f"bottleneck_{uuid.uuid4().hex[:8]}",
            generated_at=datetime.now(),
            total_chains_analyzed=len(self.chains),
            total_steps_analyzed=sum(len(c.steps) for c in self.chains.values()),
            bottlenecks=bottlenecks,
            top_time_consumers=time_by_type[:5],
            recommendations=recommendations,
        )

    def _infer_bottleneck_causes(self, step_type: ReasoningStepType) -> List[str]:
        """Infer potential causes for a bottleneck.

        Args:
            step_type: The step type that is a bottleneck

        Returns:
            List of potential causes
        """
        causes = {
            ReasoningStepType.TRANSLATION: [
                "Complex natural language input",
                "Large predicate set to process",
                "LLM API latency",
            ],
            ReasoningStepType.INFERENCE: [
                "Large rule base",
                "Complex ASP program",
                "Many variables to ground",
            ],
            ReasoningStepType.VALIDATION: [
                "Multiple validation stages",
                "Large test case set",
                "Complex consistency checks",
            ],
            ReasoningStepType.RULE_GENERATION: [
                "Complex case facts",
                "LLM response time",
                "Multiple generation attempts",
            ],
            ReasoningStepType.CONSENSUS: [
                "Multiple LLM calls required",
                "Slow LLM responses",
                "Large consensus group",
            ],
            ReasoningStepType.DIALECTICAL: [
                "Multiple dialectical rounds",
                "Complex synthesis required",
                "Large argument space",
            ],
        }
        return causes.get(step_type, ["Unknown cause"])

    def _generate_bottleneck_recommendations(self, bottlenecks: List[Bottleneck]) -> List[str]:
        """Generate recommendations for addressing bottlenecks.

        Args:
            bottlenecks: List of identified bottlenecks

        Returns:
            List of recommendations
        """
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck.step_type == ReasoningStepType.TRANSLATION:
                recommendations.append("Consider caching translation results for common patterns")
            elif bottleneck.step_type == ReasoningStepType.INFERENCE:
                recommendations.append("Optimize ASP program or consider rule pruning")
            elif bottleneck.step_type == ReasoningStepType.VALIDATION:
                recommendations.append("Parallelize validation steps where possible")
            elif bottleneck.step_type == ReasoningStepType.CONSENSUS:
                recommendations.append(
                    "Consider reducing consensus group size or using faster models"
                )

        if not recommendations:
            recommendations.append("No specific optimizations identified")

        return recommendations

    def get_summary(self) -> ObservationSummary:
        """Get a summary of all observations.

        Returns:
            Summary of observations
        """
        total_steps = sum(len(c.steps) for c in self.chains.values())
        successful_chains = sum(1 for c in self.chains.values() if c.overall_success)
        success_rate = successful_chains / len(self.chains) if self.chains else 0.0

        avg_duration = (
            mean(c.total_duration_ms for c in self.chains.values()) if self.chains else 0.0
        )

        # Step type distribution
        step_distribution: Dict[str, int] = defaultdict(int)
        for chain in self.chains.values():
            for step in chain.steps:
                step_distribution[step.step_type.value] += 1

        # Domain success rates
        domain_rates: Dict[str, float] = {}
        for domain, outcomes in self._domain_outcomes.items():
            if outcomes:
                domain_rates[domain] = sum(outcomes) / len(outcomes)

        return ObservationSummary(
            summary_id=f"summary_{uuid.uuid4().hex[:8]}",
            generated_at=datetime.now(),
            observation_period_start=self.started_at,
            observation_period_end=datetime.now(),
            total_chains_observed=len(self.chains),
            total_steps_observed=total_steps,
            success_rate=success_rate,
            avg_chain_duration_ms=avg_duration,
            patterns_identified=len(self.patterns),
            bottlenecks_identified=len(
                [p for p in self.patterns.values() if p.pattern_type == PatternType.BOTTLENECK]
            ),
            domains_observed=list(self._domain_outcomes.keys()),
            step_type_distribution=dict(step_distribution),
            domain_success_rates=domain_rates,
        )

    def clear(self) -> None:
        """Clear all observations and reset state."""
        self.chains.clear()
        self.patterns.clear()
        self._step_durations.clear()
        self._step_success_rates.clear()
        self._domain_outcomes.clear()
        self.started_at = datetime.now()


class MetaReasoner:
    """Second-order reasoning about reasoning processes.

    Enables the system to reason about its own reasoning, diagnose failures,
    and suggest improvements based on observed patterns.
    """

    def __init__(self, observer: ReasoningObserver):
        """Initialize the meta-reasoner.

        Args:
            observer: The reasoning observer to analyze
        """
        self.observer = observer
        self.diagnoses: Dict[str, FailureDiagnosis] = {}
        self.improvements: Dict[str, Improvement] = {}

    def diagnose_reasoning_failure(
        self,
        chain_or_id: Union[str, ReasoningChain],
    ) -> Optional[FailureDiagnosis]:
        """Explain why reasoning failed for a specific case.

        Args:
            chain_or_id: The chain to diagnose - can be either a chain_id string
                or a ReasoningChain object directly

        Returns:
            Diagnosis of the failure or None if chain not found
        """
        # Resolve chain from string ID or use directly if ReasoningChain
        chain = self.observer.resolve_chain(chain_or_id)
        if not chain:
            return None

        # Get the chain_id for storage
        chain_id = chain.chain_id

        if chain.overall_success:
            return None  # Not a failure

        diagnosis_id = f"diagnosis_{uuid.uuid4().hex[:8]}"

        # Identify primary failure step and its type
        primary_failure_step_id: Optional[str] = None
        primary_failure_step_type: Optional[ReasoningStepType] = None
        for step in chain.steps:
            if not step.success:
                primary_failure_step_id = step.step_id
                primary_failure_step_type = step.step_type
                break

        # Determine failure type
        failure_type = self._categorize_failure(chain)

        # Identify root causes
        root_causes = self._identify_root_causes(chain)

        # Find contributing factors
        contributing_factors = self._find_contributing_factors(chain)

        # Find similar failures
        similar = self._find_similar_failures(chain)

        # Generate explanation
        explanation = self._generate_failure_explanation(chain, failure_type, root_causes)

        diagnosis = FailureDiagnosis(
            diagnosis_id=diagnosis_id,
            chain_id=chain_id,
            case_id=chain.case_id,
            prediction=chain.prediction or "unknown",
            ground_truth=chain.ground_truth or "unknown",
            primary_failure_step=primary_failure_step_id,
            primary_failure_step_type=primary_failure_step_type,
            failure_type=failure_type,
            root_causes=root_causes,
            contributing_factors=contributing_factors,
            confidence=self._calculate_diagnosis_confidence(chain),
            explanation=explanation,
            similar_failures=[c.chain_id for c in similar],
        )

        self.diagnoses[diagnosis_id] = diagnosis
        return diagnosis

    def _categorize_failure(self, chain: ReasoningChain) -> str:
        """Categorize the type of failure.

        Args:
            chain: The failed chain

        Returns:
            Category string
        """
        failed_steps = chain.failed_steps
        if not failed_steps:
            return "prediction_mismatch"

        # Check primary failure step type
        primary = failed_steps[0]
        if primary.step_type == ReasoningStepType.TRANSLATION:
            return "translation_error"
        elif primary.step_type == ReasoningStepType.RULE_APPLICATION:
            return "rule_gap"
        elif primary.step_type == ReasoningStepType.VALIDATION:
            return "validation_failure"
        elif primary.step_type == ReasoningStepType.INFERENCE:
            return "inference_error"
        elif primary.step_type == ReasoningStepType.CONSENSUS:
            return "consensus_disagreement"
        else:
            return "unknown"

    def _identify_root_causes(self, chain: ReasoningChain) -> List[str]:
        """Identify root causes of the failure.

        Args:
            chain: The failed chain

        Returns:
            List of root causes
        """
        causes = []

        # Check for low confidence steps
        low_confidence_steps = [s for s in chain.steps if s.confidence < 0.5 and s.success]
        if low_confidence_steps:
            causes.append("Low confidence in intermediate steps")

        # Check for failed steps
        for step in chain.failed_steps:
            if step.error_message:
                causes.append(f"Error in {step.step_type.value}: {step.error_message}")
            else:
                causes.append(f"Step {step.step_type.value} failed without error")

        # Check for domain-specific issues
        domain_success_rate = self.observer._domain_outcomes.get(chain.domain, [])
        if domain_success_rate:
            rate = sum(domain_success_rate) / len(domain_success_rate)
            if rate < 0.5:
                causes.append(f"Domain '{chain.domain}' has low success rate ({rate:.1%})")

        if not causes:
            causes.append("Unable to determine specific root cause")

        return causes

    def _find_contributing_factors(self, chain: ReasoningChain) -> List[str]:
        """Find contributing factors to the failure.

        Args:
            chain: The failed chain

        Returns:
            List of contributing factors
        """
        factors = []

        # Check chain duration
        if chain.total_duration_ms > 5000:
            factors.append("Long processing time may indicate complexity issues")

        # Check step count
        if len(chain.steps) > 10:
            factors.append("Many reasoning steps increase error probability")

        # Check for patterns associated with failures
        for pattern in self.observer.patterns.values():
            if pattern.pattern_type == PatternType.FAILURE and pattern.success_correlation < -0.3:
                for step in chain.steps:
                    if step.step_type in pattern.associated_step_types:
                        factors.append(f"Matches failure pattern: {pattern.name}")
                        break

        return factors

    def _find_similar_failures(
        self, chain: ReasoningChain, max_results: int = 5
    ) -> List[ReasoningChain]:
        """Find similar failure cases.

        Args:
            chain: The chain to compare against
            max_results: Maximum number of similar failures to return

        Returns:
            List of similar failed chains
        """
        similar = []

        failed_chains = self.observer.get_failed_chains()
        for other in failed_chains:
            if other.chain_id == chain.chain_id:
                continue

            # Check similarity
            similarity_score = 0

            # Same domain
            if other.domain == chain.domain:
                similarity_score += 2

            # Similar failure type
            if self._categorize_failure(other) == self._categorize_failure(chain):
                similarity_score += 3

            # Similar step types failed
            chain_failed_types = {s.step_type for s in chain.failed_steps}
            other_failed_types = {s.step_type for s in other.failed_steps}
            overlap = len(chain_failed_types & other_failed_types)
            similarity_score += overlap

            if similarity_score >= 3:
                similar.append((similarity_score, other))

        # Sort by similarity and return top results
        similar.sort(key=lambda x: x[0], reverse=True)
        return [chain for _, chain in similar[:max_results]]

    def _generate_failure_explanation(
        self,
        chain: ReasoningChain,
        failure_type: str,
        root_causes: List[str],
    ) -> str:
        """Generate a natural language explanation of the failure.

        Args:
            chain: The failed chain
            failure_type: Category of failure
            root_causes: Identified root causes

        Returns:
            Human-readable explanation
        """
        explanation_parts = [
            f"The reasoning about case '{chain.case_id}' was flawed.",
            f"Failure type: {failure_type}.",
        ]

        if root_causes:
            explanation_parts.append(f"Root causes identified: {'; '.join(root_causes[:3])}.")

        if chain.prediction and chain.ground_truth:
            explanation_parts.append(
                f"Predicted '{chain.prediction}' but correct answer was '{chain.ground_truth}'."
            )

        return " ".join(explanation_parts)

    def _calculate_diagnosis_confidence(self, chain: ReasoningChain) -> float:
        """Calculate confidence in the diagnosis.

        Args:
            chain: The chain being diagnosed

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Increase if we found failed steps
        if chain.failed_steps:
            confidence += 0.2

        # Increase if similar failures exist
        similar = self._find_similar_failures(chain, max_results=1)
        if similar:
            confidence += 0.1

        # Increase if we have domain data
        if chain.domain in self.observer._domain_outcomes:
            confidence += 0.1

        return min(confidence, 1.0)

    def suggest_improvements(
        self,
        max_improvements: int = 10,
    ) -> List[Improvement]:
        """Suggest improvements based on observed patterns.

        Args:
            max_improvements: Maximum number of improvements to suggest

        Returns:
            List of suggested improvements
        """
        improvements = []

        # Analyze patterns for improvements
        patterns = self.observer.identify_patterns()
        for pattern in patterns:
            if pattern.pattern_type == PatternType.FAILURE:
                improvement = self._create_improvement_from_failure_pattern(pattern)
                if improvement:
                    improvements.append(improvement)
            elif pattern.pattern_type == PatternType.BOTTLENECK:
                improvement = self._create_improvement_from_bottleneck(pattern)
                if improvement:
                    improvements.append(improvement)

        # Analyze bottleneck report
        bottleneck_report = self.observer.analyze_bottlenecks()
        for bottleneck in bottleneck_report.bottlenecks:
            improvement = self._create_improvement_from_bottleneck_detail(bottleneck)
            if improvement and improvement.improvement_id not in [
                i.improvement_id for i in improvements
            ]:
                improvements.append(improvement)

        # Analyze diagnoses for common issues
        if self.diagnoses:
            diagnosis_improvements = self._analyze_diagnoses_for_improvements()
            improvements.extend(diagnosis_improvements)

        # Sort by priority and limit
        priority_order = {
            ImprovementPriority.CRITICAL: 0,
            ImprovementPriority.HIGH: 1,
            ImprovementPriority.MEDIUM: 2,
            ImprovementPriority.LOW: 3,
        }
        improvements.sort(key=lambda x: priority_order[x.priority])

        # Store and return
        for imp in improvements[:max_improvements]:
            self.improvements[imp.improvement_id] = imp

        return improvements[:max_improvements]

    def _create_improvement_from_failure_pattern(
        self, pattern: ReasoningPattern
    ) -> Optional[Improvement]:
        """Create an improvement suggestion from a failure pattern.

        Args:
            pattern: The failure pattern

        Returns:
            Improvement suggestion or None
        """
        if not pattern.associated_step_types:
            return None

        step_type = pattern.associated_step_types[0]
        improvement_id = f"imp_failure_{step_type.value}"

        improvement_map = {
            ReasoningStepType.TRANSLATION: (
                ImprovementType.PROMPT_REFINEMENT,
                "Improve Translation Prompts",
                "Refine NL↔ASP translation prompts to reduce errors",
                "translation",
            ),
            ReasoningStepType.RULE_APPLICATION: (
                ImprovementType.RULE_MODIFICATION,
                "Expand Rule Coverage",
                "Add rules to cover identified gaps",
                "rule_base",
            ),
            ReasoningStepType.VALIDATION: (
                ImprovementType.VALIDATION_THRESHOLD,
                "Adjust Validation Thresholds",
                "Review and adjust validation thresholds",
                "validation",
            ),
        }

        if step_type not in improvement_map:
            return None

        imp_type, title, desc, target = improvement_map[step_type]

        return Improvement(
            improvement_id=improvement_id,
            improvement_type=imp_type,
            priority=ImprovementPriority.HIGH,
            title=title,
            description=desc,
            expected_impact=f"Reduce {step_type.value} failures",
            target_component=target,
            supporting_evidence=[f"Pattern '{pattern.name}' shows {pattern.frequency} occurrences"],
            related_patterns=[pattern.pattern_id],
        )

    def _create_improvement_from_bottleneck(
        self, pattern: ReasoningPattern
    ) -> Optional[Improvement]:
        """Create an improvement suggestion from a bottleneck pattern.

        Args:
            pattern: The bottleneck pattern

        Returns:
            Improvement suggestion or None
        """
        if not pattern.associated_step_types:
            return None

        step_type = pattern.associated_step_types[0]

        return Improvement(
            improvement_id=f"imp_perf_{step_type.value}",
            improvement_type=ImprovementType.RESOURCE_ALLOCATION,
            priority=ImprovementPriority.MEDIUM,
            title=f"Optimize {step_type.value} Performance",
            description=f"Address performance bottleneck in {step_type.value}",
            expected_impact="Reduce processing time",
            target_component=step_type.value,
            supporting_evidence=[
                f"Average duration: {pattern.characteristics.get('avg_duration_ms', 0):.0f}ms"
            ],
            related_patterns=[pattern.pattern_id],
        )

    def _create_improvement_from_bottleneck_detail(self, bottleneck: Bottleneck) -> Improvement:
        """Create an improvement from detailed bottleneck analysis.

        Args:
            bottleneck: The bottleneck details

        Returns:
            Improvement suggestion
        """
        priority = (
            ImprovementPriority.HIGH
            if bottleneck.severity == "high"
            else ImprovementPriority.MEDIUM
        )

        return Improvement(
            improvement_id=f"imp_bn_{bottleneck.bottleneck_id}",
            improvement_type=ImprovementType.RESOURCE_ALLOCATION,
            priority=priority,
            title=f"Optimize {bottleneck.step_type.value}",
            description=bottleneck.description,
            expected_impact=f"Reduce {bottleneck.percentage_of_total_time:.1f}% of processing time",
            target_component=bottleneck.step_type.value,
            estimated_effort="medium",
            supporting_evidence=bottleneck.potential_causes,
            implementation_steps=[
                f"Analyze {bottleneck.step_type.value} implementation",
                "Identify optimization opportunities",
                "Implement and measure improvements",
            ],
        )

    def _analyze_diagnoses_for_improvements(self) -> List[Improvement]:
        """Analyze stored diagnoses to find common improvement opportunities.

        Returns:
            List of improvements based on diagnosis patterns
        """
        improvements = []

        # Count failure types
        failure_type_counts: Dict[str, int] = defaultdict(int)
        for diagnosis in self.diagnoses.values():
            failure_type_counts[diagnosis.failure_type] += 1

        # Create improvements for common failure types
        for failure_type, count in failure_type_counts.items():
            if count >= 3:  # Minimum threshold
                improvement = Improvement(
                    improvement_id=f"imp_diag_{failure_type}",
                    improvement_type=ImprovementType.STRATEGY_CHANGE,
                    priority=ImprovementPriority.HIGH if count >= 5 else ImprovementPriority.MEDIUM,
                    title=f"Address {failure_type} Failures",
                    description=f"Multiple cases ({count}) failing with {failure_type}",
                    expected_impact=f"Fix {count} failure cases",
                    target_component=failure_type,
                    supporting_evidence=[f"{count} diagnoses identified this failure type"],
                    related_diagnoses=[
                        d.diagnosis_id
                        for d in self.diagnoses.values()
                        if d.failure_type == failure_type
                    ],
                )
                improvements.append(improvement)

        return improvements

    def explain_reasoning_quality(self, domain: Optional[str] = None) -> str:
        """Generate a natural language explanation of reasoning quality.

        Args:
            domain: Optional domain to focus on

        Returns:
            Human-readable quality explanation
        """
        summary = self.observer.get_summary()

        parts = [
            f"Observed {summary.total_chains_observed} reasoning chains "
            f"with {summary.success_rate:.1%} overall success rate."
        ]

        if domain and domain in summary.domain_success_rates:
            domain_rate = summary.domain_success_rates[domain]
            parts.append(f"Domain '{domain}' has {domain_rate:.1%} success rate.")
        elif summary.domain_success_rates:
            best_domain = max(summary.domain_success_rates.items(), key=lambda x: x[1])
            worst_domain = min(summary.domain_success_rates.items(), key=lambda x: x[1])
            parts.append(
                f"Best performing domain: '{best_domain[0]}' ({best_domain[1]:.1%}). "
                f"Worst: '{worst_domain[0]}' ({worst_domain[1]:.1%})."
            )

        if summary.patterns_identified > 0:
            parts.append(f"Identified {summary.patterns_identified} reasoning patterns.")

        if summary.bottlenecks_identified > 0:
            parts.append(f"Found {summary.bottlenecks_identified} performance bottlenecks.")

        return " ".join(parts)
