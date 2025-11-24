"""
Critic LLM System for dialectical rule analysis.

This system analyzes proposed rules to find flaws, edge cases, and contradictions,
implementing dialectical reasoning where rules are refined through criticism.
"""

import json
from typing import Any, List, Optional

from loguru import logger

from loft.dialectical.critique_prompts import (
    get_contradiction_check_prompt,
    get_critique_prompt,
    get_edge_case_prompt,
    get_synthesis_prompt,
)
from loft.dialectical.critique_schemas import (
    Contradiction,
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    EdgeCase,
)
from loft.neural.rule_schemas import GeneratedRule


class CriticSystem:
    """
    Critic LLM specialized in finding flaws and contradictions.

    Workflow:
    1. Receive proposed rule + context
    2. Generate criticisms from multiple angles
    3. Identify edge cases
    4. Check for contradictions
    5. Score severity of issues
    6. Optionally synthesize improved version
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        enable_synthesis: bool = True,
        mock_mode: bool = False,
    ):
        """
        Initialize the critic system.

        Args:
            llm_client: LLM client for generating critiques (if None, mock mode)
            enable_synthesis: Whether to generate improved rule versions
            mock_mode: If True, use mock critiques for testing
        """
        self.llm_client = llm_client
        self.enable_synthesis = enable_synthesis
        self.mock_mode = mock_mode or llm_client is None

        if self.mock_mode:
            logger.warning("CriticSystem running in mock mode (no real LLM)")

        logger.info(
            f"Initialized CriticSystem (synthesis: {enable_synthesis}, mock: {self.mock_mode})"
        )

    def critique_rule(
        self,
        proposed_rule: GeneratedRule,
        existing_rules: List[str],
        context: Optional[str] = None,
    ) -> CritiqueReport:
        """
        Generate comprehensive critique of proposed rule.

        Args:
            proposed_rule: The rule to critique
            existing_rules: List of existing rules in the system
            context: Additional context about the domain

        Returns:
            CritiqueReport with identified issues, edge cases, and contradictions
        """
        logger.info(f"Critiquing rule: {proposed_rule.asp_rule[:60]}...")

        if self.mock_mode:
            return self._mock_critique(proposed_rule, existing_rules)

        # Generate critique using LLM
        prompt = get_critique_prompt(
            proposed_rule.asp_rule,
            existing_rules,
            context or "Contract law domain",
        )

        try:
            response = self._call_llm(prompt)
            critique_data = json.loads(response)

            # Parse critique data into structured objects
            issues = [
                CritiqueIssue(
                    category=i["category"],
                    description=i["description"],
                    severity=CritiqueSeverity(i.get("severity", "medium")),
                    suggestion=i.get("suggestion"),
                )
                for i in critique_data.get("issues", [])
            ]

            edge_cases = [
                EdgeCase(
                    description=ec["description"],
                    scenario=ec.get("scenario", {}),
                    expected_outcome=ec.get("expected_outcome"),
                    current_outcome=ec.get("current_outcome"),
                    severity=CritiqueSeverity(ec.get("severity", "medium")),
                )
                for ec in critique_data.get("edge_cases", [])
            ]

            contradictions = [
                Contradiction(
                    description=c["description"],
                    new_rule=proposed_rule.asp_rule,
                    conflicting_rule=c["conflicting_rule"],
                    example_scenario=c.get("example_scenario", {}),
                    severity=CritiqueSeverity(c.get("severity", "high")),
                    resolution_suggestion=c.get("resolution_suggestion"),
                )
                for c in critique_data.get("contradictions", [])
            ]

            # Build critique report
            report = CritiqueReport(
                rule=proposed_rule.asp_rule,
                issues=issues,
                edge_cases=edge_cases,
                contradictions=contradictions,
                overall_severity=CritiqueSeverity(critique_data.get("overall_severity", "low")),
                recommendation=critique_data.get("recommendation"),
                suggested_revision=critique_data.get("suggested_revision"),
                confidence=critique_data.get("confidence", 0.5),
            )

            logger.info(
                f"Critique complete: {len(issues)} issues, {len(edge_cases)} edge cases, "
                f"{len(contradictions)} contradictions"
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate critique: {e}")
            return CritiqueReport(
                rule=proposed_rule.asp_rule,
                recommendation="error",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def find_edge_cases(self, rule: GeneratedRule) -> List[EdgeCase]:
        """
        Identify edge cases not covered by rule.

        Args:
            rule: The rule to analyze

        Returns:
            List of identified edge cases
        """
        logger.info(f"Finding edge cases for: {rule.asp_rule[:60]}...")

        if self.mock_mode:
            return self._mock_edge_cases(rule)

        prompt = get_edge_case_prompt(rule.asp_rule, "Contract law domain")

        try:
            response = self._call_llm(prompt)
            data = json.loads(response)

            edge_cases = [
                EdgeCase(
                    description=ec["description"],
                    scenario=ec.get("scenario", {}),
                    expected_outcome=ec.get("expected_outcome"),
                    current_outcome=ec.get("current_outcome"),
                    severity=CritiqueSeverity(ec.get("severity", "medium")),
                )
                for ec in data.get("edge_cases", [])
            ]

            logger.info(f"Found {len(edge_cases)} edge cases")
            return edge_cases

        except Exception as e:
            logger.error(f"Failed to find edge cases: {e}")
            return []

    def check_contradictions(
        self, new_rule: GeneratedRule, existing_rules: List[str]
    ) -> List[Contradiction]:
        """
        Find contradictions with existing rules.

        Args:
            new_rule: The proposed new rule
            existing_rules: List of existing rules to check against

        Returns:
            List of detected contradictions
        """
        logger.info(f"Checking contradictions for: {new_rule.asp_rule[:60]}...")

        if not existing_rules:
            logger.debug("No existing rules to check against")
            return []

        if self.mock_mode:
            return self._mock_contradictions(new_rule, existing_rules)

        prompt = get_contradiction_check_prompt(new_rule.asp_rule, existing_rules)

        if not prompt:
            return []

        try:
            response = self._call_llm(prompt)
            data = json.loads(response)

            contradictions = [
                Contradiction(
                    description=c["description"],
                    new_rule=new_rule.asp_rule,
                    conflicting_rule=c["conflicting_rule"],
                    example_scenario=c.get("example_scenario", {}),
                    severity=CritiqueSeverity(c.get("severity", "high")),
                    resolution_suggestion=c.get("resolution_suggestion"),
                )
                for c in data.get("contradictions", [])
            ]

            logger.info(f"Found {len(contradictions)} contradictions")
            return contradictions

        except Exception as e:
            logger.error(f"Failed to check contradictions: {e}")
            return []

    def synthesize_improvement(
        self, rule: GeneratedRule, critique: CritiqueReport, existing_rules: List[str]
    ) -> Optional[GeneratedRule]:
        """
        Synthesize an improved version of the rule based on critique.

        Args:
            rule: The original rule
            critique: The critique report
            existing_rules: Existing rules for context

        Returns:
            Improved version of the rule, or None if synthesis fails
        """
        if not self.enable_synthesis:
            logger.debug("Synthesis disabled")
            return None

        logger.info("Synthesizing improved rule...")

        if self.mock_mode:
            return self._mock_synthesis(rule, critique)

        # Create structured critique summary
        critique_summary = {
            "issues": [
                {"category": i.category, "description": i.description} for i in critique.issues
            ],
            "edge_cases": [ec.description for ec in critique.edge_cases],
            "contradictions": [c.description for c in critique.contradictions],
        }

        prompt = get_synthesis_prompt(
            rule.asp_rule, json.dumps(critique_summary, indent=2), existing_rules
        )

        try:
            response = self._call_llm(prompt)
            data = json.loads(response)

            improved_rule = GeneratedRule(
                rule_id=f"{rule.rule_id}_improved",
                asp_rule=data["improved_rule"],
                confidence=data.get("confidence", 0.7),
                strategy=rule.strategy,
                metadata={
                    "original_rule": rule.asp_rule,
                    "changes": data.get("changes"),
                    "addresses_issues": data.get("addresses_issues", []),
                    "synthesis_confidence": data.get("confidence"),
                },
            )

            logger.info("Successfully synthesized improved rule")
            return improved_rule

        except Exception as e:
            logger.error(f"Failed to synthesize improvement: {e}")
            return None

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The prompt to send

        Returns:
            LLM response as string
        """
        if not self.llm_client:
            raise ValueError("No LLM client configured")

        # This will be implemented based on the specific LLM client
        # For now, assume it has a .generate() method
        return self.llm_client.generate(prompt)

    # Mock implementations for testing without LLM

    def _mock_critique(self, rule: GeneratedRule, existing_rules: List[str]) -> CritiqueReport:
        """Generate mock critique for testing."""
        # Analyze rule structure for common issues
        issues = []

        if ":-" in rule.asp_rule:
            # Check for missing common conditions
            if "contract(C)" in rule.asp_rule and "consideration" not in rule.asp_rule:
                issues.append(
                    CritiqueIssue(
                        category="missing_condition",
                        description="Missing consideration requirement for contract enforceability",
                        severity=CritiqueSeverity.MEDIUM,
                        suggestion="Add consideration(C) to the rule body",
                    )
                )

            if "contract(C)" in rule.asp_rule and "capacity" not in rule.asp_rule:
                issues.append(
                    CritiqueIssue(
                        category="missing_condition",
                        description="Missing capacity check for parties",
                        severity=CritiqueSeverity.MEDIUM,
                        suggestion="Add capacity checks for all parties",
                    )
                )

        # Determine overall severity
        severity = CritiqueSeverity.LOW
        if issues:
            severity = max(i.severity for i in issues)

        # Simple recommendation
        if len(issues) >= 3:
            recommendation = "revise"
        elif len(issues) > 0:
            recommendation = "revise"
        else:
            recommendation = "accept"

        return CritiqueReport(
            rule=rule.asp_rule,
            issues=issues,
            edge_cases=[],
            contradictions=[],
            overall_severity=severity,
            recommendation=recommendation,
            confidence=0.6,
        )

    def _mock_edge_cases(self, rule: GeneratedRule) -> List[EdgeCase]:
        """Generate mock edge cases for testing."""
        return [
            EdgeCase(
                description="Contract with minor party",
                scenario={"party_age": 17, "contract_type": "employment"},
                expected_outcome="Contract voidable by minor",
                severity=CritiqueSeverity.HIGH,
            ),
            EdgeCase(
                description="Oral contract exceeding statute of frauds threshold",
                scenario={"contract_value": 500, "contract_form": "oral"},
                expected_outcome="Contract unenforceable under statute of frauds",
                severity=CritiqueSeverity.MEDIUM,
            ),
        ]

    def _mock_contradictions(
        self, rule: GeneratedRule, existing_rules: List[str]
    ) -> List[Contradiction]:
        """Generate mock contradictions for testing."""
        # Simple heuristic: check if rule head appears in existing rules
        contradictions = []

        if ":-" in rule.asp_rule:
            head = rule.asp_rule.split(":-")[0].strip()

            for existing in existing_rules:
                if ":-" in existing and head in existing:
                    # Potential contradiction
                    contradictions.append(
                        Contradiction(
                            description=f"Rule may conflict with existing rule on {head}",
                            new_rule=rule.asp_rule,
                            conflicting_rule=existing,
                            example_scenario={},
                            severity=CritiqueSeverity.MEDIUM,
                            resolution_suggestion="Review conditions to ensure rules are compatible",
                        )
                    )

        return contradictions

    def _mock_synthesis(
        self, rule: GeneratedRule, critique: CritiqueReport
    ) -> Optional[GeneratedRule]:
        """Generate mock improved rule for testing."""
        # Simple synthesis: add suggested conditions
        improved = rule.asp_rule

        for issue in critique.issues:
            if issue.category == "missing_condition" and issue.suggestion:
                # Try to add the suggestion to the rule body
                if ":-" in improved and "consideration" in issue.suggestion.lower():
                    improved = improved.replace(".", ", consideration(C).")

        if improved != rule.asp_rule:
            return GeneratedRule(
                rule_id=f"{rule.rule_id}_improved",
                asp_rule=improved,
                confidence=0.75,
                strategy=rule.strategy,
                metadata={"original": rule.asp_rule, "mock_synthesis": True},
            )

        return None
