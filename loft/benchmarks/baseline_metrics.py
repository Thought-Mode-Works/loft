"""
Baseline metrics framework for LOFT infrastructure.

Issue #257: Baseline Validation Benchmarks
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class RuleGenerationMetrics:
    """Metrics for rule generation performance."""

    rules_per_hour: float
    avg_generation_time_ms: float
    avg_tokens_used: int
    success_rate: float  # % of attempts producing valid ASP
    retry_rate: float  # % requiring retry
    predicate_alignment_rate: float  # % using dataset predicates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ValidationMetrics:
    """Metrics for validation pipeline performance."""

    # Per-stage pass rates
    syntax_pass_rate: float
    semantic_pass_rate: float
    empirical_pass_rate: float
    consensus_pass_rate: float
    dialectical_pass_rate: float

    # Timing
    avg_validation_time_ms: float
    avg_time_per_stage: Dict[str, float] = field(default_factory=dict)

    # Overall
    overall_pass_rate: float = 0.0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MetaReasoningMetrics:
    """Metrics for meta-reasoning system."""

    # Strategy selection
    strategy_distribution: Dict[str, float]
    strategy_success_rates: Dict[str, float]
    adaptation_frequency: float  # Adaptations per 100 cases

    # Failure analysis
    failure_patterns_detected: int
    recommendations_generated: int
    recommendations_applied: int

    # Prompt optimization
    prompt_variants_tested: int
    avg_improvement_per_optimization: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TranslationMetrics:
    """Metrics for ontological bridge translation."""

    # ASP -> NL
    asp_to_nl_fidelity: float
    asp_to_nl_avg_time_ms: float

    # NL -> ASP
    nl_to_asp_fidelity: float
    nl_to_asp_avg_time_ms: float

    # Roundtrip
    roundtrip_fidelity: float
    semantic_preservation_rate: float

    # Grounding
    grounding_success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EndToEndMetrics:
    """End-to-end processing metrics."""

    cases_per_hour: float
    avg_case_time_ms: float

    # Breakdown
    avg_gap_identification_ms: float
    avg_rule_generation_ms: float
    avg_validation_ms: float
    avg_incorporation_ms: float
    avg_persistence_ms: float

    # Success rates
    cases_with_new_rules: float
    cases_with_improvement: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PersistenceMetrics:
    """Metrics for ASP persistence system."""

    avg_save_time_ms: float
    avg_load_time_ms: float
    avg_snapshot_time_ms: float

    # Storage
    avg_file_size_bytes: float
    storage_overhead_ratio: float  # Actual / theoretical minimum

    # Reliability
    save_success_rate: float
    load_success_rate: float
    corruption_recovery_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BaselineMetrics:
    """Complete baseline metrics for LOFT infrastructure."""

    # Metadata
    timestamp: str
    commit_hash: str
    configuration: Dict[str, Any]

    # Component metrics
    rule_generation: RuleGenerationMetrics
    validation: ValidationMetrics
    meta_reasoning: MetaReasoningMetrics
    translation: TranslationMetrics
    end_to_end: EndToEndMetrics
    persistence: PersistenceMetrics

    def save(self, path: Path):
        """
        Save baseline to JSON.

        Args:
            path: Path to save baseline metrics
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": self.timestamp,
            "commit_hash": self.commit_hash,
            "configuration": self.configuration,
            "rule_generation": self.rule_generation.to_dict(),
            "validation": self.validation.to_dict(),
            "meta_reasoning": self.meta_reasoning.to_dict(),
            "translation": self.translation.to_dict(),
            "end_to_end": self.end_to_end.to_dict(),
            "persistence": self.persistence.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved baseline metrics to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaselineMetrics":
        """
        Load baseline from JSON.

        Args:
            path: Path to baseline metrics file

        Returns:
            Loaded BaselineMetrics
        """
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            timestamp=data["timestamp"],
            commit_hash=data["commit_hash"],
            configuration=data["configuration"],
            rule_generation=RuleGenerationMetrics(**data["rule_generation"]),
            validation=ValidationMetrics(**data["validation"]),
            meta_reasoning=MetaReasoningMetrics(**data["meta_reasoning"]),
            translation=TranslationMetrics(**data["translation"]),
            end_to_end=EndToEndMetrics(**data["end_to_end"]),
            persistence=PersistenceMetrics(**data["persistence"]),
        )

    def to_markdown(self) -> str:
        """
        Generate markdown baseline report.

        Returns:
            Markdown-formatted report
        """
        report = f"""# LOFT Infrastructure Baseline Metrics

**Date**: {self.timestamp}
**Commit**: {self.commit_hash}
**Configuration**: {self.configuration.get('description', 'N/A')}

## Rule Generation

| Metric | Value |
|--------|-------|
| Rules per hour | {self.rule_generation.rules_per_hour:.1f} |
| Avg generation time | {self.rule_generation.avg_generation_time_ms:.1f} ms |
| Avg tokens used | {self.rule_generation.avg_tokens_used} |
| Success rate | {self.rule_generation.success_rate:.1%} |
| Retry rate | {self.rule_generation.retry_rate:.1%} |
| Predicate alignment | {self.rule_generation.predicate_alignment_rate:.1%} |

## Validation Pipeline

| Stage | Pass Rate | Avg Time |
|-------|-----------|----------|
| Syntax | {self.validation.syntax_pass_rate:.1%} | {self.validation.avg_time_per_stage.get('syntax', 0):.1f} ms |
| Semantic | {self.validation.semantic_pass_rate:.1%} | {self.validation.avg_time_per_stage.get('semantic', 0):.1f} ms |
| Empirical | {self.validation.empirical_pass_rate:.1%} | {self.validation.avg_time_per_stage.get('empirical', 0):.1f} ms |
| Consensus | {self.validation.consensus_pass_rate:.1%} | {self.validation.avg_time_per_stage.get('consensus', 0):.1f} ms |
| Dialectical | {self.validation.dialectical_pass_rate:.1%} | {self.validation.avg_time_per_stage.get('dialectical', 0):.1f} ms |
| **Overall** | **{self.validation.overall_pass_rate:.1%}** | **{self.validation.avg_validation_time_ms:.1f} ms** |

## Meta-Reasoning

| Metric | Value |
|--------|-------|
| Adaptation frequency | {self.meta_reasoning.adaptation_frequency:.2f} per 100 cases |
| Failure patterns detected | {self.meta_reasoning.failure_patterns_detected} |
| Recommendations generated | {self.meta_reasoning.recommendations_generated} |
| Recommendations applied | {self.meta_reasoning.recommendations_applied} |
| Prompt variants tested | {self.meta_reasoning.prompt_variants_tested} |
| Avg improvement | {self.meta_reasoning.avg_improvement_per_optimization:.1%} |

## Translation (Ontological Bridge)

| Metric | Value |
|--------|-------|
| ASP → NL fidelity | {self.translation.asp_to_nl_fidelity:.1%} |
| ASP → NL time | {self.translation.asp_to_nl_avg_time_ms:.1f} ms |
| NL → ASP fidelity | {self.translation.nl_to_asp_fidelity:.1%} |
| NL → ASP time | {self.translation.nl_to_asp_avg_time_ms:.1f} ms |
| Roundtrip fidelity | {self.translation.roundtrip_fidelity:.1%} |
| Semantic preservation | {self.translation.semantic_preservation_rate:.1%} |
| Grounding success | {self.translation.grounding_success_rate:.1%} |

## End-to-End

| Metric | Value |
|--------|-------|
| Cases per hour | {self.end_to_end.cases_per_hour:.1f} |
| Avg case time | {self.end_to_end.avg_case_time_ms:.1f} ms |
| Gap identification | {self.end_to_end.avg_gap_identification_ms:.1f} ms |
| Rule generation | {self.end_to_end.avg_rule_generation_ms:.1f} ms |
| Validation | {self.end_to_end.avg_validation_ms:.1f} ms |
| Incorporation | {self.end_to_end.avg_incorporation_ms:.1f} ms |
| Persistence | {self.end_to_end.avg_persistence_ms:.1f} ms |
| Cases with new rules | {self.end_to_end.cases_with_new_rules:.1%} |
| Cases with improvement | {self.end_to_end.cases_with_improvement:.1%} |

## Persistence

| Metric | Value |
|--------|-------|
| Avg save time | {self.persistence.avg_save_time_ms:.1f} ms |
| Avg load time | {self.persistence.avg_load_time_ms:.1f} ms |
| Avg snapshot time | {self.persistence.avg_snapshot_time_ms:.1f} ms |
| Avg file size | {self.persistence.avg_file_size_bytes / 1024:.1f} KB |
| Storage overhead | {self.persistence.storage_overhead_ratio:.2f}x |
| Save success rate | {self.persistence.save_success_rate:.1%} |
| Load success rate | {self.persistence.load_success_rate:.1%} |
| Corruption recovery | {self.persistence.corruption_recovery_rate:.1%} |
"""
        return report
