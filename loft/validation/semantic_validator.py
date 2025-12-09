"""
Semantic validator for LLM-generated ASP rules.

This module validates the semantic properties of generated rules, including:
- Logical consistency
- Predicate usage
- Stratification layer compatibility
- Integration with existing knowledge base
"""

from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import sys
import os
from loguru import logger
import clingo

from loft.validation.validation_schemas import ValidationResult
from loft.validation.asp_validators import ASPSemanticValidator
from loft.symbolic.asp_core import ASPCore


@contextmanager
def suppress_clingo_warnings():
    """
    Context manager to suppress Clingo informational warnings.

    Clingo outputs warnings like "atom does not occur in any rule head"
    when validating individual rules without full context. These are
    expected and not meaningful for our validation purposes.
    """
    # Save original stderr
    old_stderr = sys.stderr
    try:
        # Redirect stderr to devnull
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = old_stderr


class SemanticValidator:
    """
    Validates semantic properties of LLM-generated rules.

    Checks logical consistency, predicate coherence, and integration
    with existing knowledge base.
    """

    def __init__(self, asp_core: Optional[ASPCore] = None):
        """
        Initialize semantic validator.

        Args:
            asp_core: Optional ASP core with existing knowledge base
        """
        self.asp_core = asp_core
        self.asp_semantic_validator = ASPSemanticValidator()
        logger.debug("Initialized SemanticValidator")

    def validate_rule(
        self,
        rule_text: str,
        existing_rules: Optional[str] = None,
        target_layer: str = "tactical",
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate semantic properties of a generated rule.

        Args:
            rule_text: The ASP rule to validate
            existing_rules: Existing ASP rules in the knowledge base
            target_layer: Target stratification layer (operational/tactical/strategic)
            context: Additional context for validation

        Returns:
            ValidationResult with semantic validation outcome

        Example:
            >>> validator = SemanticValidator()
            >>> result = validator.validate_rule(
            ...     "enforceable(C) :- contract(C), not void(C).",
            ...     existing_rules="contract(c1). void(c2).",
            ...     target_layer="tactical"
            ... )
            >>> assert result.is_valid
        """
        errors = []
        warnings = []
        details = {}
        context = context or {}

        # 1. Check consistency with existing rules
        if existing_rules:
            consistency_check = self._check_consistency_with_base(
                rule_text, existing_rules
            )
            if not consistency_check["is_consistent"]:
                errors.append(
                    f"Rule creates inconsistency with existing knowledge base: "
                    f"{consistency_check['message']}"
                )
            details["consistency"] = consistency_check

        # 2. Check self-consistency (rule doesn't contradict itself)
        self_consistency = self._check_self_consistency(rule_text)
        if not self_consistency["is_consistent"]:
            errors.append(f"Rule is self-contradictory: {self_consistency['message']}")
        details["self_consistency"] = self_consistency

        # 3. Check stratification compatibility
        stratification_check = self._check_stratification_layer(rule_text, target_layer)
        if not stratification_check["is_compatible"]:
            warnings.append(
                f"Rule may not fit {target_layer} layer: {stratification_check['reason']}"
            )
        details["stratification"] = stratification_check

        # 4. Check for circular dependencies
        circularity_check = self._check_circularity(rule_text, existing_rules)
        if circularity_check["has_cycle"]:
            errors.append(
                f"Rule creates circular dependency: {circularity_check['description']}"
            )
        details["circularity"] = circularity_check

        # 5. Check predicate coherence (predicates make sense together)
        coherence_check = self._check_predicate_coherence(rule_text, context)
        if coherence_check["issues"]:
            for issue in coherence_check["issues"]:
                warnings.append(f"Predicate coherence: {issue}")
        details["coherence"] = coherence_check

        # 6. Check for redundancy with existing rules
        if existing_rules:
            redundancy_check = self._check_redundancy(rule_text, existing_rules)
            if redundancy_check["is_redundant"]:
                warnings.append(f"Rule may be redundant: {redundancy_check['reason']}")
            details["redundancy"] = redundancy_check

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            error_messages=errors,
            warnings=warnings,
            details=details,
            stage_name="semantic",
        )

    def _check_consistency_with_base(
        self, rule_text: str, existing_rules: str
    ) -> Dict[str, Any]:
        """
        Check if rule is consistent with existing knowledge base.

        Args:
            rule_text: The new rule to check
            existing_rules: Existing ASP rules

        Returns:
            Dict with consistency result
        """
        combined_program = f"{existing_rules}\n{rule_text}"

        is_consistent, msg = self.asp_semantic_validator.check_consistency(
            combined_program
        )

        return {
            "is_consistent": is_consistent,
            "message": msg,
        }

    def _check_self_consistency(self, rule_text: str) -> Dict[str, Any]:
        """
        Check if rule is internally consistent.

        Args:
            rule_text: The rule to check

        Returns:
            Dict with self-consistency result
        """
        is_consistent, msg = self.asp_semantic_validator.check_consistency(rule_text)

        return {
            "is_consistent": is_consistent,
            "message": msg,
        }

    def _check_stratification_layer(
        self, rule_text: str, target_layer: str
    ) -> Dict[str, Any]:
        """
        Check if rule is compatible with target stratification layer.

        Layers:
        - operational: Ground facts and simple derivations
        - tactical: Business logic and rules
        - strategic: High-level reasoning and policies

        Args:
            rule_text: The rule to check
            target_layer: Target layer (operational/tactical/strategic)

        Returns:
            Dict with compatibility result
        """
        # Heuristics for layer classification
        has_variables = any(c.isupper() for c in rule_text if c.isalpha())
        has_negation = "not " in rule_text
        has_aggregation = any(
            agg in rule_text for agg in ["#count", "#sum", "#min", "#max"]
        )
        is_constraint = rule_text.strip().startswith(":-")

        # Operational layer: mostly ground facts
        if target_layer == "operational":
            if has_variables or has_negation or has_aggregation:
                return {
                    "is_compatible": False,
                    "reason": "Operational layer should contain ground facts, not complex rules",
                }

        # Tactical layer: business logic with rules
        elif target_layer == "tactical":
            if has_aggregation:
                return {
                    "is_compatible": False,
                    "reason": "Aggregations typically belong in strategic layer",
                }

        # Strategic layer: high-level reasoning
        elif target_layer == "strategic":
            if not (has_negation or has_aggregation or is_constraint):
                return {
                    "is_compatible": True,
                    "reason": "Simple rule, but acceptable in strategic layer",
                }

        return {"is_compatible": True, "reason": "Rule fits target layer"}

    def _check_circularity(
        self, rule_text: str, existing_rules: Optional[str]
    ) -> Dict[str, Any]:
        """
        Check for circular dependencies through negation.

        Args:
            rule_text: The new rule
            existing_rules: Existing rules to check against

        Returns:
            Dict with circularity check result
        """
        # Basic implementation: check if adding rule creates unstratified program
        if existing_rules:
            combined = f"{existing_rules}\n{rule_text}"
        else:
            combined = rule_text

        # Use Clingo's grounding to detect issues (suppress informational warnings)
        try:
            with suppress_clingo_warnings():
                ctl = clingo.Control()
                ctl.add("base", [], combined)
                ctl.ground([("base", [])])

            # If grounding succeeds without issues, likely no problematic cycles
            return {
                "has_cycle": False,
                "description": "No circular dependencies detected",
            }

        except Exception as e:
            error_msg = str(e)
            if "cycle" in error_msg.lower() or "stratif" in error_msg.lower():
                return {
                    "has_cycle": True,
                    "description": f"Circular dependency detected: {error_msg}",
                }
            else:
                # Other error, not necessarily circularity
                return {
                    "has_cycle": False,
                    "description": f"Grounding issue (not circularity): {error_msg}",
                }

    def _check_predicate_coherence(
        self, rule_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if predicates used together make semantic sense.

        This is a heuristic check based on naming patterns and context.

        Args:
            rule_text: The rule text
            context: Context including domain information

        Returns:
            Dict with coherence issues
        """
        issues = []

        # Extract predicates (simplified heuristic)
        import re

        predicates = set(re.findall(r"([a-z][a-z0-9_]*)\s*\(", rule_text))

        # Check for mixed domains (heuristic based on predicate names)
        legal_predicates = {
            p
            for p in predicates
            if any(
                keyword in p
                for keyword in [
                    "contract",
                    "party",
                    "enforce",
                    "legal",
                    "statute",
                    "court",
                ]
            )
        }
        tech_predicates = {
            p
            for p in predicates
            if any(
                keyword in p
                for keyword in ["server", "client", "network", "database", "api"]
            )
        }

        if legal_predicates and tech_predicates:
            issues.append(
                f"Rule mixes legal and technical predicates: "
                f"legal={legal_predicates}, tech={tech_predicates}"
            )

        # Check for temporal predicates mixed with static ones
        temporal_predicates = {
            p
            for p in predicates
            if any(
                keyword in p
                for keyword in ["before", "after", "during", "when", "time"]
            )
        }
        if temporal_predicates and len(predicates) > len(temporal_predicates):
            # Mixed temporal and non-temporal - might be intentional
            pass

        return {"issues": issues, "predicates_found": list(predicates)}

    def _check_redundancy(self, rule_text: str, existing_rules: str) -> Dict[str, Any]:
        """
        Check if rule is redundant with existing rules.

        Args:
            rule_text: New rule to check
            existing_rules: Existing rules

        Returns:
            Dict with redundancy result
        """
        # Test if rule changes answer sets
        answer_sets_before = self.asp_semantic_validator.get_answer_sets(
            existing_rules, max_sets=5
        )

        combined = f"{existing_rules}\n{rule_text}"
        answer_sets_after = self.asp_semantic_validator.get_answer_sets(
            combined, max_sets=5
        )

        # Compare both count and contents of answer sets
        if len(answer_sets_before) != len(answer_sets_after):
            return {
                "is_redundant": False,
                "reason": f"Rule changes answer set count: {len(answer_sets_before)} -> {len(answer_sets_after)}",
            }

        # Same count - check if contents changed
        # Convert answer sets to comparable format (sorted string representations)
        def answer_set_to_str(symbols):
            return sorted([str(s) for s in symbols])

        before_strs = [answer_set_to_str(ans_set) for ans_set in answer_sets_before]
        after_strs = [answer_set_to_str(ans_set) for ans_set in answer_sets_after]

        # Sort to compare sets of answer sets
        before_strs.sort()
        after_strs.sort()

        if before_strs != after_strs:
            return {
                "is_redundant": False,
                "reason": "Rule changes answer set contents (adds new derivations)",
            }

        return {
            "is_redundant": True,
            "reason": "Rule does not change answer sets (may be redundant or already derivable)",
        }

    def validate_batch(
        self,
        rules: List[str],
        existing_rules: Optional[str] = None,
        target_layer: str = "tactical",
    ) -> List[ValidationResult]:
        """
        Validate multiple rules in batch.

        Args:
            rules: List of rules to validate
            existing_rules: Existing knowledge base
            target_layer: Target stratification layer

        Returns:
            List of ValidationResult objects
        """
        results = []

        for rule in rules:
            result = self.validate_rule(
                rule_text=rule,
                existing_rules=existing_rules,
                target_layer=target_layer,
            )
            results.append(result)

        return results
