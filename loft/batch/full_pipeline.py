"""
Full pipeline processor connecting batch harness to the complete learning pipeline.

This module provides a production-ready processor that wires together:
- Gap identification from ASP reasoning failures
- Rule generation via LLM with predicate alignment
- Multi-stage validation pipeline
- Rule incorporation with stratification
- Persistence to disk

Issue #253: Phase 8 baseline validation infrastructure.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from loft.batch.schemas import BatchConfig, CaseResult, CaseStatus
from loft.core.incorporation import IncorporationResult, RuleIncorporationEngine
from loft.neural.rule_generator import (
    RuleGenerationError,
    RuleGenerator,
    extract_predicates_from_asp_facts,
)
from loft.neural.rule_schemas import GeneratedRule
from loft.persistence.asp_persistence import ASPPersistenceManager
from loft.symbolic.stratification import StratificationLevel
from loft.validation.validation_pipeline import ValidationPipeline
from loft.validation.validation_schemas import ValidationReport


@dataclass
class KnowledgeGap:
    """Represents an identified gap in the symbolic knowledge base."""

    gap_id: str
    description: str
    missing_predicate: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0  # Higher = more important to fill

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gap_id": self.gap_id,
            "description": self.description,
            "missing_predicate": self.missing_predicate,
            "context": self.context,
            "priority": self.priority,
        }


@dataclass
class ProcessingMetrics:
    """Metrics collected during pipeline processing."""

    gaps_identified: int = 0
    rules_generated: int = 0
    rules_validated: int = 0
    rules_incorporated: int = 0
    rules_persisted: int = 0
    generation_errors: int = 0
    validation_failures: int = 0
    incorporation_failures: int = 0
    total_processing_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    incorporation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gaps_identified": self.gaps_identified,
            "rules_generated": self.rules_generated,
            "rules_validated": self.rules_validated,
            "rules_incorporated": self.rules_incorporated,
            "rules_persisted": self.rules_persisted,
            "generation_errors": self.generation_errors,
            "validation_failures": self.validation_failures,
            "incorporation_failures": self.incorporation_failures,
            "total_processing_time_ms": self.total_processing_time_ms,
            "llm_time_ms": self.llm_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "incorporation_time_ms": self.incorporation_time_ms,
        }


class FullPipelineProcessor:
    """
    Production processor connecting all pipeline components.

    Processes cases through the complete learning pipeline:
    1. Identify knowledge gaps from case reasoning
    2. Generate candidate rules via LLM
    3. Validate through multi-stage pipeline
    4. Incorporate valid rules into stratified core
    5. Persist rules to disk

    Example:
        >>> processor = FullPipelineProcessor(
        ...     rule_generator=generator,
        ...     validation_pipeline=pipeline,
        ...     incorporation_engine=engine,
        ...     persistence_manager=persistence,
        ... )
        >>> result = processor.process_case(case_data, accumulated_rules=[])
        >>> print(f"Generated {result.rules_generated} rules")
    """

    def __init__(
        self,
        rule_generator: RuleGenerator,
        validation_pipeline: ValidationPipeline,
        incorporation_engine: RuleIncorporationEngine,
        persistence_manager: Optional[ASPPersistenceManager] = None,
        config: Optional[BatchConfig] = None,
        target_layer: StratificationLevel = StratificationLevel.TACTICAL,
        gap_identifier: Optional[Callable[[Dict[str, Any]], List[KnowledgeGap]]] = None,
    ):
        """
        Initialize full pipeline processor.

        Args:
            rule_generator: LLM-based rule generator
            validation_pipeline: Multi-stage validation pipeline
            incorporation_engine: Rule incorporation with stratification
            persistence_manager: ASP persistence manager (optional)
            config: Batch configuration
            target_layer: Default stratification layer for new rules
            gap_identifier: Custom gap identification function (optional)
        """
        self.rule_generator = rule_generator
        self.validation_pipeline = validation_pipeline
        self.incorporation_engine = incorporation_engine
        self.persistence_manager = persistence_manager
        self.config = config or BatchConfig()
        self.target_layer = target_layer
        self._gap_identifier = gap_identifier

        # Cumulative metrics
        self.metrics = ProcessingMetrics()

        logger.info(
            f"Initialized FullPipelineProcessor with target_layer={target_layer.value}"
        )

    def process_case(
        self,
        case: Dict[str, Any],
        accumulated_rules: List[str],
    ) -> CaseResult:
        """
        Process a single case through the full pipeline.

        Args:
            case: Test case dictionary with at least 'id' and 'asp_facts'
            accumulated_rules: List of rule IDs already generated

        Returns:
            CaseResult with processing outcome
        """
        start_time = time.time()
        case_id = case.get("id", str(uuid.uuid4())[:8])

        logger.info(f"Processing case {case_id} through full pipeline")

        try:
            # Extract predicates from case facts for alignment
            asp_facts = case.get("asp_facts", case.get("facts", ""))
            case_predicates = extract_predicates_from_asp_facts(asp_facts)

            # 1. Identify knowledge gaps
            gaps = self._identify_gaps(case)
            self.metrics.gaps_identified += len(gaps)

            # 2. Generate rules for gaps (limited by config)
            generated_rules: List[GeneratedRule] = []
            generation_start = time.time()

            for gap in gaps[: self.config.max_rules_per_case]:
                try:
                    rule = self._generate_rule_for_gap(gap, case, case_predicates)
                    if rule:
                        generated_rules.append(rule)
                        self.metrics.rules_generated += 1
                except RuleGenerationError as e:
                    logger.warning(f"Failed to generate rule for gap {gap.gap_id}: {e}")
                    self.metrics.generation_errors += 1

            self.metrics.llm_time_ms += (time.time() - generation_start) * 1000

            # 3. Validate generated rules
            validation_start = time.time()
            validated_rules: List[tuple[GeneratedRule, ValidationReport]] = []

            for rule in generated_rules:
                report = self._validate_rule(rule, case)
                self.metrics.rules_validated += 1

                if report.final_decision == "accept":
                    validated_rules.append((rule, report))
                else:
                    self.metrics.validation_failures += 1
                    logger.debug(
                        f"Rule rejected by validation: {report.rejection_reason}"
                    )

            self.metrics.validation_time_ms += (time.time() - validation_start) * 1000

            # 4. Incorporate valid rules
            incorporation_start = time.time()
            incorporated_count = 0
            generated_rule_ids: List[str] = []

            for rule, validation in validated_rules:
                inc_result = self._incorporate_rule(rule, validation)
                if inc_result.is_success():
                    incorporated_count += 1
                    self.metrics.rules_incorporated += 1
                    rule_id = f"rule_{case_id}_{incorporated_count}"
                    generated_rule_ids.append(rule_id)
                else:
                    self.metrics.incorporation_failures += 1
                    logger.debug(f"Incorporation failed: {inc_result.reason}")

            self.metrics.incorporation_time_ms += (
                time.time() - incorporation_start
            ) * 1000

            # 5. Persist if incorporation succeeded
            if incorporated_count > 0 and self.persistence_manager:
                try:
                    # Create snapshot for this batch cycle
                    cycle_number = len(accumulated_rules) + incorporated_count
                    self.persistence_manager.create_snapshot(
                        cycle_number=cycle_number,
                        description=f"After processing case {case_id}",
                    )
                    self.metrics.rules_persisted += incorporated_count
                except Exception as e:
                    logger.warning(f"Failed to persist rules: {e}")

            # Calculate prediction correctness
            ground_truth = case.get("ground_truth", case.get("expected_outcome", ""))
            prediction = case.get("prediction", "")
            prediction_correct = prediction == ground_truth if ground_truth else None

            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.total_processing_time_ms += processing_time_ms

            return CaseResult(
                case_id=case_id,
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=processing_time_ms,
                rules_generated=len(generated_rules),
                rules_accepted=incorporated_count,
                rules_rejected=len(generated_rules) - incorporated_count,
                prediction_correct=prediction_correct,
                confidence=self._calculate_confidence(validated_rules),
                generated_rule_ids=generated_rule_ids,
                metadata={
                    "gaps_identified": len(gaps),
                    "validation_failures": len(generated_rules) - len(validated_rules),
                    "case_predicates_count": len(case_predicates),
                },
            )

        except Exception as e:
            logger.error(f"Error processing case {case_id}: {e}")
            processing_time_ms = (time.time() - start_time) * 1000

            return CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                processed_at=datetime.now(),
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )

    def _identify_gaps(self, case: Dict[str, Any]) -> List[KnowledgeGap]:
        """
        Identify knowledge gaps from case analysis.

        Args:
            case: Test case dictionary

        Returns:
            List of identified knowledge gaps
        """
        # Use custom gap identifier if provided
        if self._gap_identifier:
            return self._gap_identifier(case)

        # Default gap identification based on case structure
        gaps = []
        case_id = case.get("id", "unknown")
        ground_truth = case.get("ground_truth", case.get("expected_outcome", ""))
        prediction = case.get("prediction", "unknown")

        # If prediction differs from ground truth, create gap
        if ground_truth and prediction != ground_truth:
            # Try to identify what predicate might be missing
            asp_facts = case.get("asp_facts", case.get("facts", ""))
            principle = case.get("legal_principle", case.get("principle", ""))

            missing_predicate = f"outcome_for_{case_id}"

            gap = KnowledgeGap(
                gap_id=f"gap_{case_id}_{uuid.uuid4().hex[:8]}",
                description=(
                    f"Case {case_id} predicted {prediction} but expected {ground_truth}. "
                    f"Need rule to determine correct outcome."
                ),
                missing_predicate=missing_predicate,
                context={
                    "case_id": case_id,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "asp_facts": asp_facts[:500] if asp_facts else "",
                    "legal_principle": principle,
                },
                priority=1.0,
            )
            gaps.append(gap)

        # Also check for explicit gaps in case data
        explicit_gaps = case.get("knowledge_gaps", [])
        for i, gap_data in enumerate(explicit_gaps):
            if isinstance(gap_data, str):
                gap = KnowledgeGap(
                    gap_id=f"explicit_gap_{case_id}_{i}",
                    description=gap_data,
                    missing_predicate=f"predicate_{i}",
                )
            elif isinstance(gap_data, dict):
                gap = KnowledgeGap(
                    gap_id=gap_data.get("id", f"explicit_gap_{case_id}_{i}"),
                    description=gap_data.get("description", "Unknown gap"),
                    missing_predicate=gap_data.get(
                        "missing_predicate", f"predicate_{i}"
                    ),
                    context=gap_data.get("context", {}),
                    priority=gap_data.get("priority", 1.0),
                )
            else:
                continue
            gaps.append(gap)

        return gaps

    def _generate_rule_for_gap(
        self,
        gap: KnowledgeGap,
        case: Dict[str, Any],
        case_predicates: List[str],
    ) -> Optional[GeneratedRule]:
        """
        Generate a rule to fill the knowledge gap.

        Args:
            gap: Knowledge gap to fill
            case: Test case context
            case_predicates: Predicates extracted from case facts

        Returns:
            Generated rule or None if generation fails
        """
        logger.debug(f"Generating rule for gap: {gap.gap_id}")

        # Get legal principle from case if available
        principle = gap.context.get(
            "legal_principle",
            case.get("legal_principle", case.get("principle", "")),
        )

        if principle:
            # Use principle-based generation with predicate alignment
            try:
                rule = self.rule_generator.generate_from_principle_aligned(
                    principle_text=principle,
                    dataset_predicates=case_predicates,
                )
                return rule
            except RuleGenerationError:
                # Fall through to gap filling
                pass

        # Fall back to gap filling
        gap_response = self.rule_generator.fill_knowledge_gap(
            gap_description=gap.description,
            missing_predicate=gap.missing_predicate,
            context=gap.context,
            dataset_predicates=case_predicates,
        )

        # Return the recommended candidate
        if gap_response.candidates:
            recommended_idx = min(
                gap_response.recommended_index, len(gap_response.candidates) - 1
            )
            candidate = gap_response.candidates[recommended_idx]
            return candidate.rule

        return None

    def _validate_rule(
        self,
        rule: GeneratedRule,
        case: Dict[str, Any],
    ) -> ValidationReport:
        """
        Validate a generated rule through the pipeline.

        Args:
            rule: Generated rule to validate
            case: Test case context

        Returns:
            Validation report with decision
        """
        return self.validation_pipeline.validate_rule(
            rule_asp=rule.asp_rule,
            rule_id=f"gen_{uuid.uuid4().hex[:8]}",
            target_layer=self.target_layer.value,
            proposer_reasoning=rule.reasoning,
            source_type=rule.source_type or "gap_fill",
            context={
                "case_id": case.get("id"),
                "domain": case.get("_domain", "unknown"),
            },
        )

    def _incorporate_rule(
        self,
        rule: GeneratedRule,
        validation_report: ValidationReport,
    ) -> IncorporationResult:
        """
        Incorporate a validated rule into the symbolic core.

        Args:
            rule: Validated rule to incorporate
            validation_report: Validation evidence

        Returns:
            Incorporation result
        """
        return self.incorporation_engine.incorporate(
            rule=rule,
            target_layer=self.target_layer,
            validation_report=validation_report,
            is_autonomous=True,
        )

    def _calculate_confidence(
        self,
        validated_rules: List[tuple[GeneratedRule, ValidationReport]],
    ) -> float:
        """
        Calculate overall confidence from validated rules.

        Args:
            validated_rules: List of (rule, validation) tuples

        Returns:
            Aggregate confidence score
        """
        if not validated_rules:
            return 0.0

        confidences = [
            rule.confidence * validation.aggregate_confidence
            for rule, validation in validated_rules
        ]
        return sum(confidences) / len(confidences)

    def get_metrics(self) -> ProcessingMetrics:
        """Get cumulative processing metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset cumulative metrics."""
        self.metrics = ProcessingMetrics()


def create_full_pipeline_processor(
    model: str = "claude-3-5-haiku-20241022",
    persistence_dir: str = "./asp_rules",
    enable_persistence: bool = True,
    target_layer: StratificationLevel = StratificationLevel.TACTICAL,
    validation_threshold: float = 0.6,
) -> FullPipelineProcessor:
    """
    Factory function to create a configured full pipeline processor.

    Creates all necessary components with sensible defaults for
    production batch processing.

    Args:
        model: LLM model identifier
        persistence_dir: Directory for ASP rule persistence
        enable_persistence: Whether to persist rules to disk
        target_layer: Default stratification layer for new rules
        validation_threshold: Minimum confidence for validation

    Returns:
        Configured FullPipelineProcessor

    Example:
        >>> processor = create_full_pipeline_processor(
        ...     model="claude-3-5-haiku-20241022",
        ...     enable_persistence=True,
        ... )
        >>> # Use with batch harness
        >>> harness.run_batch(cases, processor.process_case)
    """
    from loft.neural.llm_interface import LLMInterface
    from loft.neural.providers import create_anthropic_provider
    from loft.symbolic.asp_core import ASPCore

    # Create LLM interface
    provider = create_anthropic_provider(model=model)
    llm_interface = LLMInterface(provider=provider)

    # Create ASP core
    asp_core = ASPCore()

    # Create rule generator
    rule_generator = RuleGenerator(
        llm=llm_interface,
        asp_core=asp_core,
        domain="legal",
    )

    # Create validation pipeline
    validation_pipeline = ValidationPipeline(
        min_confidence=validation_threshold,
    )

    # Create incorporation engine
    incorporation_engine = RuleIncorporationEngine()

    # Create persistence manager if enabled
    persistence_manager = None
    if enable_persistence:
        persistence_manager = ASPPersistenceManager(
            base_dir=persistence_dir,
            enable_git=True,
        )

    return FullPipelineProcessor(
        rule_generator=rule_generator,
        validation_pipeline=validation_pipeline,
        incorporation_engine=incorporation_engine,
        persistence_manager=persistence_manager,
        target_layer=target_layer,
    )
