"""
Strategy evaluation framework for meta-reasoning.

Provides classes for defining, evaluating, and selecting reasoning strategies
based on problem characteristics and historical performance.
"""

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Protocol


class StrategyType(Enum):
    """Types of reasoning strategies."""

    CHECKLIST = "checklist"  # Element-by-element satisfaction
    CAUSAL_CHAIN = "causal_chain"  # Trace cause-effect relationships
    BALANCING_TEST = "balancing_test"  # Weigh competing factors
    RULE_BASED = "rule_based"  # Apply formal rules
    DIALECTICAL = "dialectical"  # Thesis-antithesis-synthesis
    ANALOGICAL = "analogical"  # Reason by analogy to precedent
    DEFAULT = "default"  # Fallback strategy


@dataclass
class StrategyCharacteristics:
    """Characteristics of a reasoning strategy."""

    speed: str = "medium"  # fast, medium, slow
    accuracy_profile: str = "balanced"  # high_precision, high_recall, balanced
    resource_usage: str = "medium"  # low, medium, high
    complexity_handling: str = "medium"  # simple, medium, complex
    llm_calls_typical: int = 1  # Typical number of LLM calls
    best_for: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speed": self.speed,
            "accuracy_profile": self.accuracy_profile,
            "resource_usage": self.resource_usage,
            "complexity_handling": self.complexity_handling,
            "llm_calls_typical": self.llm_calls_typical,
            "best_for": self.best_for,
            "limitations": self.limitations,
        }


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""

    strategy_name: str
    domain: Optional[str] = None
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    accuracy: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    std_duration_ms: float = 0.0
    avg_confidence: float = 0.0
    recorded_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_cases == 0:
            return 0.0
        return self.successful_cases / self.total_cases

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "domain": self.domain,
            "total_cases": self.total_cases,
            "successful_cases": self.successful_cases,
            "failed_cases": self.failed_cases,
            "accuracy": self.accuracy,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "std_duration_ms": self.std_duration_ms,
            "avg_confidence": self.avg_confidence,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass
class ComparisonReport:
    """Report comparing multiple strategies."""

    report_id: str
    domain: str
    strategies_compared: List[str]
    best_strategy: str
    best_accuracy: float
    generated_at: datetime = field(default_factory=datetime.now)
    strategy_rankings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    statistical_significance: bool = False

    @property
    def rankings(self) -> List[Dict[str, Any]]:
        """Alias for strategy_rankings for consistent API."""
        return self.strategy_rankings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "domain": self.domain,
            "strategies_compared": self.strategies_compared,
            "best_strategy": self.best_strategy,
            "best_accuracy": self.best_accuracy,
            "generated_at": self.generated_at.isoformat(),
            "strategy_rankings": self.strategy_rankings,
            "rankings": self.rankings,  # Include alias in dict
            "recommendations": self.recommendations,
            "statistical_significance": self.statistical_significance,
        }


@dataclass
class CounterfactualAnalysis:
    """Analysis of why an alternative strategy was not selected.

    Provides "why not" reasoning for strategies that were considered
    but ultimately rejected in favor of the selected strategy.
    """

    alternative: str
    why_not_selected: str
    hypothetical_performance: float
    confidence: float
    comparison_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alternative": self.alternative,
            "why_not_selected": self.why_not_selected,
            "hypothetical_performance": self.hypothetical_performance,
            "confidence": self.confidence,
            "comparison_factors": self.comparison_factors,
        }


@dataclass
class SelectionExplanation:
    """Explanation for why a strategy was selected.

    Includes counterfactual analysis of alternative strategies, enabling
    the system to reason about "what if" scenarios for strategy selection.

    Attributes:
        strategy_name: Name of the selected strategy
        case_id: Identifier of the case being analyzed
        domain: Legal domain of the case
        reasons: List of reasons for the selection
        confidence: Confidence score for the selection (0-1)
        alternative_strategies: List of alternative strategy names considered
        domain_performance: Historical accuracy in this domain (if available)
        counterfactuals: Detailed counterfactual analyses of alternatives

    Example:
        >>> explanation = selector.explain_selection(case)
        >>> print(f"Selected: {explanation.strategy_name}")
        >>> for cf in explanation.alternatives_considered:
        ...     print(f"  Alternative: {cf.alternative}")
        ...     print(f"    Why not: {cf.why_not_selected}")
        ...     print(f"    Expected performance: {cf.hypothetical_performance:.1%}")
    """

    strategy_name: str
    case_id: str
    domain: str
    reasons: List[str]
    confidence: float
    alternative_strategies: List[str] = field(default_factory=list)
    domain_performance: Optional[float] = None
    counterfactuals: List[CounterfactualAnalysis] = field(default_factory=list)

    @property
    def alternatives_considered(self) -> List[CounterfactualAnalysis]:
        """Alias for counterfactuals for API consistency.

        Returns the list of counterfactual analyses, allowing callers to use
        either `.counterfactuals` or `.alternatives_considered` interchangeably.

        Returns:
            List of CounterfactualAnalysis objects for alternative strategies
        """
        return self.counterfactuals

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        counterfactual_dicts = [c.to_dict() for c in self.counterfactuals]
        return {
            "strategy_name": self.strategy_name,
            "case_id": self.case_id,
            "domain": self.domain,
            "reasons": self.reasons,
            "confidence": self.confidence,
            "alternative_strategies": self.alternative_strategies,
            "domain_performance": self.domain_performance,
            "counterfactuals": counterfactual_dicts,
            "alternatives_considered": counterfactual_dicts,  # Alias for API consistency
        }

    def explain(self) -> str:
        """Generate human-readable explanation."""
        parts = [
            f"Selected strategy '{self.strategy_name}' for case '{self.case_id}'.",
            f"Domain: {self.domain}.",
        ]
        if self.reasons:
            parts.append(f"Reasons: {'; '.join(self.reasons)}.")
        if self.domain_performance is not None:
            parts.append(
                f"Historical accuracy in this domain: {self.domain_performance:.1%}."
            )
        parts.append(f"Selection confidence: {self.confidence:.1%}.")

        # Add counterfactual reasoning
        if self.counterfactuals:
            parts.append("Alternatives considered:")
            for cf in self.counterfactuals[:3]:  # Limit to top 3
                parts.append(f"  - {cf.alternative}: {cf.why_not_selected}")

        return " ".join(parts)


class CaseProtocol(Protocol):
    """Protocol for case objects that can be used with strategies."""

    @property
    def case_id(self) -> str:
        """Case identifier."""
        ...

    @property
    def domain(self) -> str:
        """Legal domain of the case."""
        ...


@dataclass
class SimpleCase:
    """Simple case implementation for testing."""

    case_id: str
    domain: str
    facts: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningStrategy(ABC):
    """Base class for reasoning strategies."""

    def __init__(
        self,
        name: str,
        strategy_type: StrategyType,
        description: str,
        applicable_domains: Optional[List[str]] = None,
        characteristics: Optional[StrategyCharacteristics] = None,
    ):
        """Initialize the strategy.

        Args:
            name: Unique name for the strategy
            strategy_type: Type of strategy
            description: Human-readable description
            applicable_domains: Domains where this strategy works (None = all)
            characteristics: Performance characteristics
        """
        self.name = name
        self.strategy_type = strategy_type
        self.description = description
        self.applicable_domains = applicable_domains
        self.characteristics = characteristics or StrategyCharacteristics()

    @abstractmethod
    def apply(self, case: CaseProtocol) -> Dict[str, Any]:
        """Apply this strategy to a case.

        Args:
            case: The case to reason about

        Returns:
            Result dictionary with prediction and metadata
        """
        pass

    def is_applicable(self, domain: str) -> bool:
        """Check if this strategy is applicable to a domain.

        Args:
            domain: The domain to check

        Returns:
            True if applicable
        """
        if self.applicable_domains is None:
            return True
        return domain in self.applicable_domains

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "strategy_type": self.strategy_type.value,
            "description": self.description,
            "applicable_domains": self.applicable_domains,
            "characteristics": self.characteristics.to_dict(),
        }


class ChecklistStrategy(ReasoningStrategy):
    """Strategy that checks elements one by one."""

    def __init__(
        self,
        name: str = "checklist",
        applicable_domains: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            strategy_type=StrategyType.CHECKLIST,
            description="Check legal elements one by one for satisfaction",
            applicable_domains=applicable_domains or ["contracts", "statute_of_frauds"],
            characteristics=StrategyCharacteristics(
                speed="fast",
                accuracy_profile="high_precision",
                resource_usage="low",
                complexity_handling="simple",
                llm_calls_typical=1,
                best_for=["element-based legal tests", "contract formation"],
                limitations=["May miss nuanced interactions between elements"],
            ),
        )

    def apply(self, case: CaseProtocol) -> Dict[str, Any]:
        """Apply checklist reasoning."""
        return {
            "strategy": self.name,
            "approach": "element_satisfaction",
            "domain": case.domain,
        }


class CausalChainStrategy(ReasoningStrategy):
    """Strategy that traces cause-effect relationships."""

    def __init__(
        self,
        name: str = "causal_chain",
        applicable_domains: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            strategy_type=StrategyType.CAUSAL_CHAIN,
            description="Trace causal chains from action to harm",
            applicable_domains=applicable_domains or ["torts"],
            characteristics=StrategyCharacteristics(
                speed="medium",
                accuracy_profile="balanced",
                resource_usage="medium",
                complexity_handling="complex",
                llm_calls_typical=2,
                best_for=["negligence", "causation analysis", "tort liability"],
                limitations=["Struggles with multiple concurrent causes"],
            ),
        )

    def apply(self, case: CaseProtocol) -> Dict[str, Any]:
        """Apply causal chain reasoning."""
        return {
            "strategy": self.name,
            "approach": "trace_causation",
            "domain": case.domain,
        }


class BalancingTestStrategy(ReasoningStrategy):
    """Strategy that weighs competing factors."""

    def __init__(
        self,
        name: str = "balancing_test",
        applicable_domains: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            strategy_type=StrategyType.BALANCING_TEST,
            description="Weigh competing factors and interests",
            applicable_domains=applicable_domains or ["procedural", "constitutional"],
            characteristics=StrategyCharacteristics(
                speed="medium",
                accuracy_profile="balanced",
                resource_usage="medium",
                complexity_handling="complex",
                llm_calls_typical=2,
                best_for=["standing", "jurisdiction", "constitutional scrutiny"],
                limitations=["Subjective weighting can vary"],
            ),
        )

    def apply(self, case: CaseProtocol) -> Dict[str, Any]:
        """Apply balancing test reasoning."""
        return {
            "strategy": self.name,
            "approach": "weigh_factors",
            "domain": case.domain,
        }


class RuleBasedStrategy(ReasoningStrategy):
    """Strategy that applies formal rules directly."""

    def __init__(
        self,
        name: str = "rule_based",
        applicable_domains: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            strategy_type=StrategyType.RULE_BASED,
            description="Apply formal symbolic rules via ASP solver",
            applicable_domains=applicable_domains,  # Applicable to all
            characteristics=StrategyCharacteristics(
                speed="fast",
                accuracy_profile="high_precision",
                resource_usage="low",
                complexity_handling="medium",
                llm_calls_typical=0,
                best_for=["well-defined rules", "statutory interpretation"],
                limitations=["Requires pre-existing rules"],
            ),
        )

    def apply(self, case: CaseProtocol) -> Dict[str, Any]:
        """Apply rule-based reasoning."""
        return {
            "strategy": self.name,
            "approach": "formal_rules",
            "domain": case.domain,
        }


class DialecticalStrategy(ReasoningStrategy):
    """Strategy using thesis-antithesis-synthesis."""

    def __init__(
        self,
        name: str = "dialectical",
        applicable_domains: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            strategy_type=StrategyType.DIALECTICAL,
            description="Use dialectical reasoning with multiple perspectives",
            applicable_domains=applicable_domains,  # Applicable to all
            characteristics=StrategyCharacteristics(
                speed="slow",
                accuracy_profile="high_recall",
                resource_usage="high",
                complexity_handling="complex",
                llm_calls_typical=3,
                best_for=["ambiguous cases", "novel legal questions"],
                limitations=["Resource intensive", "May over-complicate simple cases"],
            ),
        )

    def apply(self, case: CaseProtocol) -> Dict[str, Any]:
        """Apply dialectical reasoning."""
        return {
            "strategy": self.name,
            "approach": "thesis_antithesis_synthesis",
            "domain": case.domain,
        }


class AnalogicalStrategy(ReasoningStrategy):
    """Strategy that reasons by analogy to precedent."""

    def __init__(
        self,
        name: str = "analogical",
        applicable_domains: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            strategy_type=StrategyType.ANALOGICAL,
            description="Reason by analogy to similar precedent cases",
            applicable_domains=applicable_domains
            or ["property_law", "adverse_possession"],
            characteristics=StrategyCharacteristics(
                speed="medium",
                accuracy_profile="balanced",
                resource_usage="medium",
                complexity_handling="complex",
                llm_calls_typical=2,
                best_for=["property rights", "common law reasoning"],
                limitations=["Requires good precedent database"],
            ),
        )

    def apply(self, case: CaseProtocol) -> Dict[str, Any]:
        """Apply analogical reasoning."""
        return {
            "strategy": self.name,
            "approach": "precedent_analogy",
            "domain": case.domain,
        }


# Default strategy catalog
DEFAULT_STRATEGIES: Dict[str, ReasoningStrategy] = {
    "checklist": ChecklistStrategy(),
    "causal_chain": CausalChainStrategy(),
    "balancing_test": BalancingTestStrategy(),
    "rule_based": RuleBasedStrategy(),
    "dialectical": DialecticalStrategy(),
    "analogical": AnalogicalStrategy(),
}


class StrategyEvaluator:
    """Evaluates and compares reasoning strategies."""

    def __init__(
        self,
        strategies: Optional[Dict[str, ReasoningStrategy]] = None,
    ):
        """Initialize the evaluator.

        Args:
            strategies: Available strategies (defaults to DEFAULT_STRATEGIES)
        """
        self.strategies = strategies or dict(DEFAULT_STRATEGIES)
        self._performance_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = (
            defaultdict(lambda: defaultdict(list))
        )

    def register_strategy(self, strategy: ReasoningStrategy) -> None:
        """Register a new strategy.

        Args:
            strategy: The strategy to register
        """
        self.strategies[strategy.name] = strategy

    def get_strategy(self, name: str) -> Optional[ReasoningStrategy]:
        """Get a strategy by name.

        Args:
            name: Strategy name

        Returns:
            The strategy or None
        """
        return self.strategies.get(name)

    def get_applicable_strategies(self, domain: str) -> List[ReasoningStrategy]:
        """Get strategies applicable to a domain.

        Args:
            domain: The domain to filter by

        Returns:
            List of applicable strategies
        """
        return [s for s in self.strategies.values() if s.is_applicable(domain)]

    def record_result(
        self,
        strategy_name: str,
        domain: str,
        success: bool,
        duration_ms: float,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a strategy execution result.

        Args:
            strategy_name: Name of the strategy used
            domain: Domain of the case
            success: Whether the strategy produced correct result
            duration_ms: Execution duration in milliseconds
            confidence: Confidence score of the result
            metadata: Additional metadata
        """
        result = {
            "success": success,
            "duration_ms": duration_ms,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }
        self._performance_history[strategy_name][domain].append(result)

    def evaluate_strategy(
        self,
        strategy: ReasoningStrategy,
        domain: Optional[str] = None,
    ) -> StrategyMetrics:
        """Evaluate a strategy's performance.

        Args:
            strategy: The strategy to evaluate
            domain: Optional domain to filter by

        Returns:
            Strategy metrics
        """
        history = self._performance_history.get(strategy.name, {})

        if domain:
            results = history.get(domain, [])
        else:
            results = []
            for domain_results in history.values():
                results.extend(domain_results)

        if not results:
            return StrategyMetrics(
                strategy_name=strategy.name,
                domain=domain,
            )

        successful = sum(1 for r in results if r["success"])
        durations = [r["duration_ms"] for r in results]
        confidences = [r["confidence"] for r in results if "confidence" in r]

        return StrategyMetrics(
            strategy_name=strategy.name,
            domain=domain,
            total_cases=len(results),
            successful_cases=successful,
            failed_cases=len(results) - successful,
            accuracy=successful / len(results) if results else 0.0,
            avg_duration_ms=mean(durations) if durations else 0.0,
            min_duration_ms=min(durations) if durations else 0.0,
            max_duration_ms=max(durations) if durations else 0.0,
            std_duration_ms=stdev(durations) if len(durations) > 1 else 0.0,
            avg_confidence=mean(confidences) if confidences else 0.0,
        )

    def compare_strategies(
        self,
        strategy_names: List[str],
        domain: str,
    ) -> ComparisonReport:
        """Compare multiple strategies for a domain.

        Args:
            strategy_names: Names of strategies to compare
            domain: Domain to compare in

        Returns:
            Comparison report
        """
        rankings = []
        for name in strategy_names:
            strategy = self.strategies.get(name)
            if not strategy:
                continue

            metrics = self.evaluate_strategy(strategy, domain)
            rankings.append(
                {
                    "strategy_name": name,
                    "accuracy": metrics.accuracy,
                    "total_cases": metrics.total_cases,
                    "avg_duration_ms": metrics.avg_duration_ms,
                    "success_rate": metrics.success_rate,
                }
            )

        # Sort by accuracy descending
        rankings.sort(key=lambda x: x["accuracy"], reverse=True)

        best_strategy = rankings[0]["strategy_name"] if rankings else "none"
        best_accuracy = rankings[0]["accuracy"] if rankings else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(rankings, domain)

        # Check statistical significance (simplified)
        significant = (
            len(rankings) >= 2
            and rankings[0]["total_cases"] >= 10
            and rankings[0]["accuracy"] - rankings[1]["accuracy"] > 0.1
        )

        return ComparisonReport(
            report_id=f"cmp_{uuid.uuid4().hex[:8]}",
            domain=domain,
            strategies_compared=strategy_names,
            best_strategy=best_strategy,
            best_accuracy=best_accuracy,
            strategy_rankings=rankings,
            recommendations=recommendations,
            statistical_significance=significant,
        )

    def _generate_recommendations(
        self, rankings: List[Dict[str, Any]], domain: str
    ) -> List[str]:
        """Generate recommendations based on comparison.

        Args:
            rankings: Strategy rankings
            domain: The domain being analyzed

        Returns:
            List of recommendations
        """
        recommendations = []

        if not rankings:
            recommendations.append(f"No strategies evaluated for domain '{domain}'")
            return recommendations

        best = rankings[0]
        if best["total_cases"] < 10:
            recommendations.append(
                f"More data needed: only {best['total_cases']} cases evaluated"
            )

        if best["accuracy"] < 0.7:
            recommendations.append(
                f"Best strategy has only {best['accuracy']:.1%} accuracy - "
                "consider developing new strategies"
            )

        if len(rankings) >= 2:
            diff = best["accuracy"] - rankings[1]["accuracy"]
            if diff < 0.05:
                recommendations.append(
                    f"Top strategies have similar accuracy (diff={diff:.1%}) - "
                    "consider speed/resource tradeoffs"
                )

        return recommendations


class StrategySelector:
    """Dynamically selects optimal strategy for a case."""

    def __init__(
        self,
        evaluator: StrategyEvaluator,
        selection_policy: str = "best_accuracy",
    ):
        """Initialize the selector.

        Args:
            evaluator: Strategy evaluator with performance history
            selection_policy: Policy for selection (best_accuracy, balanced, fast)
        """
        self.evaluator = evaluator
        self.selection_policy = selection_policy

        # Domain to default strategy mapping
        self._domain_defaults: Dict[str, str] = {
            "contracts": "checklist",
            "statute_of_frauds": "checklist",
            "torts": "causal_chain",
            "procedural": "balancing_test",
            "property_law": "analogical",
            "adverse_possession": "analogical",
        }

        # Selection callbacks
        self._on_selection: Optional[Callable[[str, str, ReasoningStrategy], None]] = (
            None
        )

        # Track most recent selection for explain_selection() without args
        self._last_selection: Optional[Dict[str, Any]] = None

    def set_callback(
        self,
        on_selection: Optional[Callable[[str, str, ReasoningStrategy], None]] = None,
    ) -> None:
        """Set callback for selection events.

        Args:
            on_selection: Called with (case_id, domain, strategy) on selection
        """
        self._on_selection = on_selection

    def set_domain_default(self, domain: str, strategy_name: str) -> None:
        """Set the default strategy for a domain.

        Args:
            domain: The domain
            strategy_name: Default strategy name
        """
        self._domain_defaults[domain] = strategy_name

    def get_domain_default(self, domain: str) -> Optional[str]:
        """Get the default strategy for a domain.

        Args:
            domain: The domain to look up

        Returns:
            Strategy name, or None if no default set
        """
        return self._domain_defaults.get(domain)

    def select_strategy(
        self,
        case: CaseProtocol,
        available_strategies: Optional[List[str]] = None,
    ) -> ReasoningStrategy:
        """Select the best strategy for a case.

        Args:
            case: The case to reason about
            available_strategies: Optional list of strategies to consider

        Returns:
            Selected strategy
        """
        domain = case.domain

        # Get applicable strategies
        if available_strategies:
            candidates = [
                self.evaluator.strategies[name]
                for name in available_strategies
                if name in self.evaluator.strategies
                and self.evaluator.strategies[name].is_applicable(domain)
            ]
        else:
            candidates = self.evaluator.get_applicable_strategies(domain)

        if not candidates:
            # Fallback to rule_based if no candidates
            return self.evaluator.strategies.get(
                "rule_based", list(self.evaluator.strategies.values())[0]
            )

        # Select based on policy
        if self.selection_policy == "best_accuracy":
            selected = self._select_by_accuracy(candidates, domain)
        elif self.selection_policy == "fast":
            selected = self._select_by_speed(candidates)
        elif self.selection_policy == "balanced":
            selected = self._select_balanced(candidates, domain)
        else:
            selected = self._select_by_accuracy(candidates, domain)

        # Track selection for explain_selection() without args
        self._last_selection = {
            "case": case,
            "strategy": selected,
            "candidates": candidates,
            "domain": domain,
        }

        if self._on_selection:
            self._on_selection(case.case_id, domain, selected)

        return selected

    def _select_by_accuracy(
        self,
        candidates: List[ReasoningStrategy],
        domain: str,
    ) -> ReasoningStrategy:
        """Select strategy with best accuracy in domain.

        Args:
            candidates: Available strategies
            domain: The domain

        Returns:
            Best accuracy strategy
        """
        best_strategy = None
        best_accuracy = -1.0

        for strategy in candidates:
            metrics = self.evaluator.evaluate_strategy(strategy, domain)
            # Use accuracy if we have data, otherwise use domain default
            if metrics.total_cases > 0:
                if metrics.accuracy > best_accuracy:
                    best_accuracy = metrics.accuracy
                    best_strategy = strategy
            elif strategy.name == self._domain_defaults.get(domain):
                if best_strategy is None:
                    best_strategy = strategy

        return best_strategy or candidates[0]

    def _select_by_speed(
        self, candidates: List[ReasoningStrategy]
    ) -> ReasoningStrategy:
        """Select fastest strategy.

        Args:
            candidates: Available strategies

        Returns:
            Fastest strategy
        """
        speed_order = {"fast": 0, "medium": 1, "slow": 2}
        return min(
            candidates,
            key=lambda s: speed_order.get(s.characteristics.speed, 1),
        )

    def _select_balanced(
        self,
        candidates: List[ReasoningStrategy],
        domain: str,
    ) -> ReasoningStrategy:
        """Select strategy balancing accuracy and speed.

        Args:
            candidates: Available strategies
            domain: The domain

        Returns:
            Balanced strategy
        """
        speed_scores = {"fast": 1.0, "medium": 0.5, "slow": 0.0}

        best_strategy = None
        best_score = -1.0

        for strategy in candidates:
            metrics = self.evaluator.evaluate_strategy(strategy, domain)
            accuracy = metrics.accuracy if metrics.total_cases > 0 else 0.5
            speed = speed_scores.get(strategy.characteristics.speed, 0.5)

            # Combined score: 70% accuracy, 30% speed
            score = 0.7 * accuracy + 0.3 * speed

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy or candidates[0]

    def explain_selection(
        self,
        case: Optional[CaseProtocol] = None,
        strategy: Optional[ReasoningStrategy] = None,
        include_counterfactuals: bool = True,
    ) -> SelectionExplanation:
        """Explain why a strategy was selected.

        Can be called in three ways:
        1. explain_selection() - Explains the most recent selection
        2. explain_selection(case, strategy) - Explains a specific selection
        3. explain_selection(strategy=some_strategy) - Explains with last case

        Args:
            case: The case (uses last selection if None)
            strategy: The selected strategy (uses last selection if None)
            include_counterfactuals: Include "why not" reasoning for alternatives

        Returns:
            Selection explanation with optional counterfactual analysis

        Raises:
            ValueError: If no selection has been made and no args provided
        """
        # Resolve case and strategy from last selection if not provided
        if case is None and strategy is None:
            if self._last_selection is None:
                raise ValueError(
                    "No selection to explain. Call select_strategy() first "
                    "or provide case and strategy arguments."
                )
            case = self._last_selection["case"]
            strategy = self._last_selection["strategy"]
        elif case is None:
            if self._last_selection is None:
                raise ValueError("No case provided and no previous selection.")
            case = self._last_selection["case"]
        elif strategy is None:
            if self._last_selection is None:
                raise ValueError("No strategy provided and no previous selection.")
            strategy = self._last_selection["strategy"]

        domain = case.domain
        reasons = []

        # Get performance metrics
        metrics = self.evaluator.evaluate_strategy(strategy, domain)

        # Build reasons
        if strategy.is_applicable(domain):
            reasons.append(f"Strategy is designed for '{domain}' domain")

        if metrics.total_cases > 0:
            reasons.append(
                f"Historical accuracy in domain: {metrics.accuracy:.1%} "
                f"({metrics.total_cases} cases)"
            )
        else:
            if strategy.name == self._domain_defaults.get(domain):
                reasons.append(f"Default strategy for '{domain}' domain")

        reasons.append(f"Speed profile: {strategy.characteristics.speed}")

        if strategy.characteristics.best_for:
            reasons.append(
                f"Best for: {', '.join(strategy.characteristics.best_for[:2])}"
            )

        # Get alternatives
        alternatives = [
            s.name
            for s in self.evaluator.get_applicable_strategies(domain)
            if s.name != strategy.name
        ][:3]

        # Calculate confidence
        confidence = 0.5  # Base confidence
        if metrics.total_cases >= 10:
            confidence += 0.2
        if metrics.accuracy >= 0.8:
            confidence += 0.2
        if strategy.name == self._domain_defaults.get(domain):
            confidence += 0.1

        # Generate counterfactual analysis if requested
        counterfactuals = []
        if include_counterfactuals:
            counterfactuals = self._generate_counterfactuals(
                strategy, domain, metrics, alternatives
            )

        return SelectionExplanation(
            strategy_name=strategy.name,
            case_id=case.case_id,
            domain=domain,
            reasons=reasons,
            confidence=min(confidence, 1.0),
            alternative_strategies=alternatives,
            domain_performance=metrics.accuracy if metrics.total_cases > 0 else None,
            counterfactuals=counterfactuals,
        )

    def _generate_counterfactuals(
        self,
        selected: ReasoningStrategy,
        domain: str,
        selected_metrics: StrategyMetrics,
        alternatives: List[str],
    ) -> List[CounterfactualAnalysis]:
        """Generate counterfactual analysis for alternative strategies.

        Args:
            selected: The selected strategy
            domain: The domain
            selected_metrics: Performance metrics for selected strategy
            alternatives: Alternative strategy names to analyze

        Returns:
            List of counterfactual analyses
        """
        counterfactuals = []

        for alt_name in alternatives[:3]:  # Limit to top 3 alternatives
            alt_strategy = self.evaluator.get_strategy(alt_name)
            if not alt_strategy:
                continue

            alt_metrics = self.evaluator.evaluate_strategy(alt_strategy, domain)

            # Determine why this alternative wasn't selected
            why_not_reasons = []
            comparison_factors = []

            # Compare accuracy
            if alt_metrics.total_cases > 0 and selected_metrics.total_cases > 0:
                accuracy_diff = selected_metrics.accuracy - alt_metrics.accuracy
                if accuracy_diff > 0:
                    why_not_reasons.append(
                        f"Lower accuracy ({alt_metrics.accuracy:.1%} vs {selected_metrics.accuracy:.1%})"
                    )
                    comparison_factors.append(f"accuracy_delta={accuracy_diff:.1%}")
                elif accuracy_diff < 0:
                    comparison_factors.append(
                        f"higher_accuracy={alt_metrics.accuracy:.1%}"
                    )
            elif alt_metrics.total_cases == 0:
                why_not_reasons.append("No historical performance data")
                comparison_factors.append("no_history")

            # Compare speed
            speed_order = {"fast": 0, "medium": 1, "slow": 2}
            selected_speed = speed_order.get(selected.characteristics.speed, 1)
            alt_speed = speed_order.get(alt_strategy.characteristics.speed, 1)

            if alt_speed > selected_speed:
                why_not_reasons.append(
                    f"Slower execution ({alt_strategy.characteristics.speed} vs {selected.characteristics.speed})"
                )
                comparison_factors.append("slower")

            # Check domain applicability
            if not alt_strategy.is_applicable(domain):
                why_not_reasons.append(f"Not designed for '{domain}' domain")
                comparison_factors.append("domain_mismatch")

            # Check if it's a domain default
            if selected.name == self._domain_defaults.get(
                domain
            ) and alt_name != self._domain_defaults.get(domain):
                why_not_reasons.append("Not the default strategy for this domain")
                comparison_factors.append("not_default")

            # Default reason if none found
            if not why_not_reasons:
                why_not_reasons.append("Selected strategy had better overall fit")

            # Calculate confidence in the counterfactual
            cf_confidence = 0.5
            if alt_metrics.total_cases >= 10:
                cf_confidence += 0.2
            if len(why_not_reasons) >= 2:
                cf_confidence += 0.1

            counterfactuals.append(
                CounterfactualAnalysis(
                    alternative=alt_name,
                    why_not_selected="; ".join(why_not_reasons),
                    hypothetical_performance=(
                        alt_metrics.accuracy if alt_metrics.total_cases > 0 else 0.0
                    ),
                    confidence=min(cf_confidence, 1.0),
                    comparison_factors=comparison_factors,
                )
            )

        return counterfactuals


def get_default_strategies() -> Dict[str, ReasoningStrategy]:
    """Get a copy of the default strategy catalog.

    Returns:
        Dictionary of strategy name to strategy
    """
    return dict(DEFAULT_STRATEGIES)


def create_evaluator(
    strategies: Optional[Dict[str, ReasoningStrategy]] = None,
) -> StrategyEvaluator:
    """Create a strategy evaluator.

    Args:
        strategies: Optional custom strategies

    Returns:
        Configured StrategyEvaluator
    """
    return StrategyEvaluator(strategies)


def create_selector(
    evaluator: Optional[StrategyEvaluator] = None,
    policy: str = "best_accuracy",
) -> StrategySelector:
    """Create a strategy selector.

    Args:
        evaluator: Optional evaluator (creates new if None)
        policy: Selection policy

    Returns:
        Configured StrategySelector
    """
    if evaluator is None:
        evaluator = StrategyEvaluator()
    return StrategySelector(evaluator, policy)
