"""
Multi-stage validation pipeline for LLM-generated ASP rules.

This module orchestrates the complete validation process:
1. Syntactic validation
2. Semantic validation
3. Empirical validation
4. Consensus validation
5. Final decision with confidence gating
"""

from typing import List, Optional, Dict, Any
from loguru import logger

from loft.validation.validation_schemas import (
    ValidationReport,
    TestCase,
)
from loft.validation.asp_validators import ASPSyntaxValidator
from loft.validation.semantic_validator import SemanticValidator
from loft.validation.empirical_validator import EmpiricalValidator
from loft.validation.consensus_validator import ConsensusValidator

# Phase 4.1: Dialectical validation
try:
    from loft.dialectical.critic import CriticSystem
    from loft.neural.rule_schemas import GeneratedRule

    DIALECTICAL_AVAILABLE = True
except ImportError:
    DIALECTICAL_AVAILABLE = False
    logger.debug("Dialectical validation not available (Phase 4.1 components missing)")


class ValidationPipeline:
    """
    Multi-stage validation pipeline for generated rules.

    Runs syntactic, semantic, empirical, and consensus validation
    stages, aggregating results into a final decision.
    """

    def __init__(
        self,
        syntax_validator: Optional[ASPSyntaxValidator] = None,
        semantic_validator: Optional[SemanticValidator] = None,
        empirical_validator: Optional[EmpiricalValidator] = None,
        consensus_validator: Optional[ConsensusValidator] = None,
        critic_system: Optional["CriticSystem"] = None,
        min_confidence: float = 0.6,
        enable_dialectical: bool = False,
    ):
        """
        Initialize validation pipeline.

        Args:
            syntax_validator: ASP syntax validator (creates default if None)
            semantic_validator: Semantic validator (creates default if None)
            empirical_validator: Empirical validator (optional)
            consensus_validator: Consensus validator (optional)
            critic_system: Dialectical critic system (Phase 4.1, optional)
            min_confidence: Minimum confidence for acceptance (0.0-1.0)
            enable_dialectical: Enable dialectical validation stage
        """
        self.syntax_validator = syntax_validator or ASPSyntaxValidator()
        self.semantic_validator = semantic_validator or SemanticValidator()
        self.empirical_validator = empirical_validator
        self.consensus_validator = consensus_validator
        self.critic_system = critic_system
        self.min_confidence = min_confidence
        self.enable_dialectical = enable_dialectical and DIALECTICAL_AVAILABLE

        if enable_dialectical and not DIALECTICAL_AVAILABLE:
            logger.warning(
                "Dialectical validation requested but Phase 4.1 components not available"
            )

        logger.info(
            f"Initialized ValidationPipeline with min_confidence={min_confidence}, "
            f"dialectical={self.enable_dialectical}"
        )

    def validate_rule(
        self,
        rule_asp: str,
        rule_id: Optional[str] = None,
        target_layer: str = "tactical",
        existing_rules: Optional[str] = None,
        existing_predicates: Optional[List[str]] = None,
        test_cases: Optional[List[TestCase]] = None,
        proposer_reasoning: Optional[str] = None,
        source_type: str = "generated",
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """
        Run full validation pipeline on a generated rule.

        Args:
            rule_asp: The ASP rule to validate
            rule_id: Optional identifier for the rule
            target_layer: Target stratification layer (operational/tactical/strategic)
            existing_rules: Existing ASP rules in knowledge base
            existing_predicates: List of existing predicates
            test_cases: Optional test cases for empirical validation
            proposer_reasoning: Original reasoning for the rule
            source_type: Source of the rule (principle/case/gap_fill/etc)
            context: Additional context

        Returns:
            ValidationReport with results from all stages

        Example:
            >>> pipeline = ValidationPipeline()
            >>> report = pipeline.validate_rule(
            ...     "enforceable(C) :- contract(C), not void(C).",
            ...     target_layer="tactical"
            ... )
            >>> assert report.final_decision in ["accept", "reject", "revise", "flag_for_review"]
        """
        report = ValidationReport(
            rule_asp=rule_asp,
            rule_id=rule_id,
            target_layer=target_layer,
        )

        # Stage 1: Syntactic Validation
        logger.info(f"Running syntactic validation for rule: {rule_id or 'unknown'}")
        syntax_result = self.syntax_validator.validate_generated_rule(rule_asp, existing_predicates)
        report.add_stage("syntactic", syntax_result)

        if not syntax_result.is_valid:
            # Early termination on syntax error
            report.final_decision = "reject"
            report.rejection_reason = "Syntax validation failed"
            report.aggregate_confidence = 0.0
            logger.warning(f"Rule {rule_id} rejected due to syntax errors")
            return report

        # Stage 2: Semantic Validation
        logger.info(f"Running semantic validation for rule: {rule_id or 'unknown'}")
        semantic_result = self.semantic_validator.validate_rule(
            rule_asp,
            existing_rules=existing_rules,
            target_layer=target_layer,
            context=context or {},
        )
        report.add_stage("semantic", semantic_result)

        if not semantic_result.is_valid:
            report.final_decision = "reject"
            report.rejection_reason = "Semantic validation failed"
            report.aggregate_confidence = 0.2  # Low confidence
            logger.warning(f"Rule {rule_id} rejected due to semantic errors")
            return report

        # Stage 3: Empirical Validation (if test cases provided)
        if self.empirical_validator and test_cases:
            logger.info(f"Running empirical validation for rule: {rule_id or 'unknown'}")
            empirical_result = self.empirical_validator.validate_rule(
                rule_asp, test_cases, existing_rules
            )
            report.add_stage("empirical", empirical_result)

            if not empirical_result.is_valid:
                report.final_decision = "revise"
                report.suggested_revisions.extend(
                    [
                        f"Failed test case: {f.test_case.description}"
                        for f in empirical_result.failures[:3]
                    ]
                )
                report.aggregate_confidence = 0.4
                logger.info(f"Rule {rule_id} needs revision due to empirical failures")
                return report

        # Stage 4: Consensus Validation (if validator configured)
        if self.consensus_validator and proposer_reasoning:
            logger.info(f"Running consensus validation for rule: {rule_id or 'unknown'}")
            consensus_result = self.consensus_validator.validate_rule(
                rule_asp,
                proposer_reasoning,
                source_type=source_type,
                existing_predicates=existing_predicates,
            )
            report.add_stage("consensus", consensus_result)

            if consensus_result.decision == "reject":
                report.final_decision = "reject"
                report.rejection_reason = "Consensus rejected rule"
                report.aggregate_confidence = 0.3
                logger.warning(f"Rule {rule_id} rejected by consensus")
                return report

            if consensus_result.decision == "revise":
                report.final_decision = "revise"
                report.suggested_revisions.extend(consensus_result.suggested_revisions[:5])
                report.aggregate_confidence = 0.5
                logger.info(f"Rule {rule_id} needs revision per consensus")
                return report

        # Stage 5: Dialectical Validation (Phase 4.1, if enabled)
        if self.enable_dialectical and self.critic_system:
            logger.info(f"Running dialectical validation for rule: {rule_id or 'unknown'}")

            # Create GeneratedRule object for critic
            generated_rule = GeneratedRule(
                rule_id=rule_id or "unknown",
                asp_rule=rule_asp,
                confidence=report.aggregate_confidence,
                strategy="validation",
                metadata={"validation_stage": "dialectical"},
            )

            # Get existing rules for context
            existing_rules_list = []
            if existing_rules:
                existing_rules_list = [r.strip() for r in existing_rules.split("\n") if r.strip()]

            # Run critique
            critique_report = self.critic_system.critique_rule(
                generated_rule, existing_rules_list, context=context
            )

            report.add_stage("dialectical", critique_report)

            # Handle critique results
            if critique_report.should_reject():
                report.final_decision = "reject"
                report.rejection_reason = f"Critical issues identified: {critique_report.issues[0].description if critique_report.issues else 'Multiple issues'}"
                report.aggregate_confidence = 0.2
                logger.warning(f"Rule {rule_id} rejected by dialectical critique")
                return report

            if critique_report.should_revise():
                report.final_decision = "revise"
                if critique_report.suggested_revision:
                    report.suggested_revisions.append(
                        f"Suggested revision: {critique_report.suggested_revision}"
                    )
                for issue in critique_report.issues[:3]:
                    report.suggested_revisions.append(issue.description)
                report.aggregate_confidence = 0.6
                report.metadata["critique_confidence"] = critique_report.confidence
                logger.info(f"Rule {rule_id} needs revision per dialectical critique")
                return report

            # If critique passed, factor into confidence
            logger.info(
                f"Dialectical critique passed with {len(critique_report.issues)} minor issues"
            )

        # Calculate aggregate confidence
        confidence_scores = []

        # Syntax: if valid, high confidence
        if syntax_result.is_valid:
            confidence_scores.append(0.9)

        # Semantic: use warnings as signal
        semantic_conf = 1.0 - (len(semantic_result.warnings) * 0.1)
        confidence_scores.append(max(0.5, semantic_conf))

        # Empirical: use accuracy if available
        if "empirical" in report.stage_results:
            empirical = report.stage_results["empirical"]
            confidence_scores.append(empirical.accuracy)

        # Consensus: use consensus strength if available
        if "consensus" in report.stage_results:
            consensus = report.stage_results["consensus"]
            confidence_scores.append(consensus.consensus_strength)

        # Dialectical: use critique confidence if available
        if "dialectical" in report.stage_results:
            critique = report.stage_results["dialectical"]
            # If critique passed without major issues, add confidence
            if not critique.should_reject() and not critique.should_revise():
                confidence_scores.append(critique.confidence)

        report.aggregate_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        )

        # Final Decision
        if report.aggregate_confidence >= self.min_confidence:
            report.final_decision = "accept"
            logger.info(
                f"Rule {rule_id} accepted with confidence {report.aggregate_confidence:.2f}"
            )
        else:
            report.final_decision = "flag_for_review"
            report.flag_reason = f"Confidence {report.aggregate_confidence:.2f} below threshold {self.min_confidence}"
            logger.info(f"Rule {rule_id} flagged for review")

        return report

    def validate_batch(
        self,
        rules: List[str],
        target_layer: str = "tactical",
        existing_rules: Optional[str] = None,
        existing_predicates: Optional[List[str]] = None,
    ) -> List[ValidationReport]:
        """
        Validate multiple rules through the pipeline.

        Args:
            rules: List of ASP rules to validate
            target_layer: Target stratification layer
            existing_rules: Existing knowledge base
            existing_predicates: Existing predicates

        Returns:
            List of ValidationReport objects
        """
        reports = []

        for i, rule in enumerate(rules):
            report = self.validate_rule(
                rule_asp=rule,
                rule_id=f"rule_{i}",
                target_layer=target_layer,
                existing_rules=existing_rules,
                existing_predicates=existing_predicates,
            )
            reports.append(report)

        return reports

    def get_pipeline_stats(self, reports: List[ValidationReport]) -> Dict[str, Any]:
        """
        Get aggregate statistics from validation reports.

        Args:
            reports: List of validation reports

        Returns:
            Dictionary with pipeline statistics
        """
        if not reports:
            return {
                "total_rules": 0,
                "accepted": 0,
                "rejected": 0,
                "needs_revision": 0,
                "flagged": 0,
                "mean_confidence": 0.0,
            }

        accepted = sum(1 for r in reports if r.final_decision == "accept")
        rejected = sum(1 for r in reports if r.final_decision == "reject")
        needs_revision = sum(1 for r in reports if r.final_decision == "revise")
        flagged = sum(1 for r in reports if r.final_decision == "flag_for_review")

        mean_confidence = sum(r.aggregate_confidence for r in reports) / len(reports)

        return {
            "total_rules": len(reports),
            "accepted": accepted,
            "rejected": rejected,
            "needs_revision": needs_revision,
            "flagged": flagged,
            "mean_confidence": mean_confidence,
            "acceptance_rate": accepted / len(reports) if reports else 0.0,
        }
