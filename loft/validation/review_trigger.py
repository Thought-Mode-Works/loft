"""
Review trigger system for detecting when human review is needed.

Analyzes rules and validation reports to determine if human oversight
is required based on confidence, impact, novelty, and other factors.
"""

from typing import List, Optional, Set

from loguru import logger

from loft.neural.rule_schemas import GeneratedRule
from loft.validation.review_schemas import (
    ReviewConfig,
    ReviewTriggerResult,
    RuleImpact,
)
from loft.validation.validation_schemas import ValidationReport


class ReviewTrigger:
    """
    Determine when human review is required.

    Checks multiple conditions to identify rules that need human oversight.
    """

    def __init__(self, config: Optional[ReviewConfig] = None):
        """
        Initialize review trigger.

        Args:
            config: Review configuration with thresholds
        """
        self.config = config or ReviewConfig()
        logger.debug("Initialized ReviewTrigger")

    def should_review(
        self,
        rule: GeneratedRule,
        validation_report: ValidationReport,
        existing_predicates: Optional[Set[str]] = None,
        total_rules_count: int = 0,
    ) -> Optional[ReviewTriggerResult]:
        """
        Determine if rule needs human review.

        Args:
            rule: Generated rule to check
            validation_report: Validation results
            existing_predicates: Set of known predicates (for novelty detection)
            total_rules_count: Total number of rules in knowledge base

        Returns:
            ReviewTriggerResult if review needed, None otherwise

        Example:
            >>> trigger = ReviewTrigger()
            >>> result = trigger.should_review(rule, report)
            >>> if result:
            ...     print(f"Review needed: {result.reason}")
        """
        triggers = []

        # 1. Check if validation pipeline flagged for review
        if validation_report.final_decision == "flag_for_review":
            triggers.append(
                (
                    "confidence_borderline",
                    "medium",
                    validation_report.metadata.get(
                        "flag_reason", "Confidence borderline for target layer"
                    ),
                )
            )

        # 2. Check confidence variance (validation sources disagree)
        if (
            hasattr(validation_report, "aggregate_confidence")
            and validation_report.aggregate_confidence
        ):
            if (
                validation_report.aggregate_confidence.variance
                > self.config.confidence_variance_threshold
            ):
                triggers.append(
                    (
                        "validation_disagreement",
                        "medium",
                        f"High variance in validation: {validation_report.aggregate_confidence.variance:.2f}",
                    )
                )

        # 3. Check constitutional layer
        if self._affects_constitutional_layer(rule):
            triggers.append(
                (
                    "constitutional_layer",
                    "critical",
                    "Constitutional layer requires human approval",
                )
            )

        # 4. Check for novel predicates
        if existing_predicates and self.config.enable_novelty_detection:
            if rule.new_predicates:
                novel = [p for p in rule.new_predicates if p not in existing_predicates]
                if novel:
                    triggers.append(
                        (
                            "novel_predicate",
                            "medium",
                            f"New predicates: {', '.join(novel)}",
                        )
                    )

        # 5. Check consensus (if available)
        if "consensus" in validation_report.stage_results:
            consensus_result = validation_report.stage_results["consensus"]
            if (
                hasattr(consensus_result, "consensus_strength")
                and consensus_result.consensus_strength < self.config.consensus_strength_threshold
            ):
                triggers.append(
                    (
                        "divided_consensus",
                        "high",
                        f"Consensus strength only {consensus_result.consensus_strength:.2f}",
                    )
                )

        # 6. Check empirical failures
        if "empirical" in validation_report.stage_results:
            empirical_result = validation_report.stage_results["empirical"]
            if (
                hasattr(empirical_result, "failures")
                and empirical_result.failures
                and len(empirical_result.failures) > 0
            ):
                triggers.append(
                    (
                        "empirical_failures",
                        "medium",
                        f"{len(empirical_result.failures)} test failures",
                    )
                )

        # 7. Check high impact (if total_rules_count provided)
        if total_rules_count > 0:
            impact = self._assess_impact(rule, total_rules_count)
            if impact.is_high_impact(self.config.impact_threshold):
                triggers.append(
                    (
                        "high_impact",
                        "high",
                        f"High impact: affects {impact.affects_rules} rules, novelty {impact.novelty_score:.2f}",
                    )
                )

        if not triggers:
            return None

        # Use highest priority trigger
        priority_order = ["critical", "high", "medium", "low"]
        triggers.sort(key=lambda x: priority_order.index(x[1]))
        trigger_type, priority, reason = triggers[0]

        result = ReviewTriggerResult(
            trigger_type=trigger_type,
            priority=priority,
            reason=reason,
            all_triggers=[t[0] for t in triggers],
        )

        logger.info(f"Review trigger fired: {result.summary()}")
        return result

    def _affects_constitutional_layer(self, rule: GeneratedRule) -> bool:
        """
        Check if rule modifies constitutional layer predicates.

        Args:
            rule: Generated rule to check

        Returns:
            True if constitutional predicates are used
        """
        constitutional_predicates = set(self.config.constitutional_predicates)

        # Check predicates used in the rule
        if rule.predicates_used:
            if any(pred in constitutional_predicates for pred in rule.predicates_used):
                return True

        # Check new predicates
        if rule.new_predicates:
            if any(pred in constitutional_predicates for pred in rule.new_predicates):
                return True

        # Also check if rule text contains constitutional keywords
        rule_text_lower = rule.asp_rule.lower()
        constitutional_keywords = ["constitutional", "fundamental", "core_principle"]
        if any(keyword in rule_text_lower for keyword in constitutional_keywords):
            return True

        return False

    def _assess_impact(self, rule: GeneratedRule, total_rules_count: int) -> RuleImpact:
        """
        Assess how many existing rules/cases this rule affects.

        Args:
            rule: Generated rule
            total_rules_count: Total number of rules in knowledge base

        Returns:
            RuleImpact assessment

        Note:
            This is a simplified implementation. A full implementation would:
            - Analyze predicate dependencies
            - Check how many existing rules use the same predicates
            - Estimate test case impact
            - Calculate true novelty score
        """
        # Simplified impact assessment
        # In a full implementation, this would analyze the actual ASP knowledge base

        # Estimate based on predicates used
        predicates = rule.predicates_used or []
        new_predicates = rule.new_predicates or []

        # Rough heuristic: more predicates = potentially higher impact
        # New predicates = potentially novel/high impact
        estimated_affects = len(predicates) * 2  # Rough estimate
        novelty = len(new_predicates) / max(len(predicates) + len(new_predicates), 1)

        return RuleImpact(
            affects_rules=estimated_affects,
            affects_test_cases=0,  # Would need actual test suite analysis
            predicate_usage_frequency={},
            novelty_score=novelty,
        )

    def get_trigger_types(self) -> List[str]:
        """
        Get list of all trigger types.

        Returns:
            List of trigger type names
        """
        return [
            "confidence_borderline",
            "validation_disagreement",
            "constitutional_layer",
            "novel_predicate",
            "divided_consensus",
            "empirical_failures",
            "high_impact",
        ]

    def explain_triggers(self) -> str:
        """
        Generate human-readable explanation of all trigger types.

        Returns:
            Formatted string explaining triggers
        """
        explanations = [
            "Review Trigger Types:",
            "",
            "1. confidence_borderline: Confidence score is near threshold",
            "2. validation_disagreement: High variance among validation sources",
            "3. constitutional_layer: Rule affects constitutional predicates",
            "4. novel_predicate: Rule introduces new predicates",
            "5. divided_consensus: Low agreement among voter LLMs",
            "6. empirical_failures: Rule fails test cases",
            "7. high_impact: Rule affects many existing rules (>10%)",
        ]
        return "\n".join(explanations)
