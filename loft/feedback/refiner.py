"""
Rule refinement engine using LLM feedback.

Proposes rule refinements based on performance issues,
using LLM to generate improved versions of underperforming rules.

Issue #278: Rule Refinement and Feedback Loop
"""

from typing import List, Optional

from loft.feedback.schemas import (
    PerformanceIssue,
    RefinementProposal,
    RulePerformanceMetrics,
)
from loft.neural.llm_interface import LLMInterface


class RuleRefiner:
    """
    Generates rule refinement proposals based on performance feedback.

    Uses LLM to analyze underperforming rules and propose improved versions,
    guided by specific performance issues and failure examples.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initialize rule refiner.

        Args:
            llm: LLM interface for generating refinements
        """
        self.llm = llm

    def propose_refinement(
        self,
        rule_text: str,
        rule_id: str,
        metrics: RulePerformanceMetrics,
        issues: List[PerformanceIssue],
    ) -> Optional[RefinementProposal]:
        """
        Generate a refinement proposal for an underperforming rule.

        Args:
            rule_text: The ASP rule to refine
            rule_id: Rule identifier
            metrics: Performance metrics for the rule
            issues: Specific issues identified with the rule

        Returns:
            RefinementProposal or None if refinement not possible
        """
        # Build prompt for LLM
        prompt = self._build_refinement_prompt(rule_text, metrics, issues)

        # Get LLM response
        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more focused refinements
                max_tokens=1000,
            )

            # Parse response into refinement proposal
            proposal = self._parse_refinement_response(
                response, rule_id, rule_text, issues
            )

            return proposal

        except Exception as e:
            # Log error and return None
            print(f"Error generating refinement: {e}")
            return None

    def _build_refinement_prompt(
        self,
        rule_text: str,
        metrics: RulePerformanceMetrics,
        issues: List[PerformanceIssue],
    ) -> str:
        """Build prompt for LLM to generate refinement."""
        prompt_parts = [
            "# Rule Refinement Task",
            "",
            "You are a legal reasoning expert analyzing an underperforming ASP rule.",
            "Your task is to propose a refined version that addresses identified issues.",
            "",
            "## Current Rule",
            "```asp",
            rule_text,
            "```",
            "",
            "## Performance Metrics",
            f"- Accuracy when used: {metrics.accuracy_when_used:.1%} ({metrics.correct_when_used}/{metrics.times_used})",
            f"- Usage rate: {metrics.usage_rate:.1%} ({metrics.times_used}/{metrics.total_questions} questions)",
            f"- Average confidence: {metrics.avg_confidence:.1%}",
            "",
        ]

        # Add domain breakdown if available
        if metrics.by_domain:
            prompt_parts.append("## Performance by Domain")
            for domain, stats in metrics.by_domain.items():
                accuracy = stats["correct"] / stats["used"] if stats["used"] > 0 else 0
                prompt_parts.append(
                    f"- {domain}: {accuracy:.1%} ({stats['correct']}/{stats['used']})"
                )
            prompt_parts.append("")

        # Add identified issues
        if issues:
            prompt_parts.append("## Identified Issues")
            for i, issue in enumerate(issues, 1):
                prompt_parts.append(f"{i}. [{issue.issue_type}] {issue.description}")
                if issue.example_failures:
                    prompt_parts.append("   Example failures:")
                    for example in issue.example_failures[:2]:
                        prompt_parts.append(f"   - {example}")
            prompt_parts.append("")

        # Add refinement instructions
        prompt_parts.extend(
            [
                "## Refinement Task",
                "",
                "Analyze the rule and propose a refined version that:",
                "1. Addresses the identified issues",
                "2. Maintains the core legal principle",
                "3. Uses valid ASP syntax",
                "4. Is more accurate on the problem cases",
                "",
                "Provide your response in the following format:",
                "",
                "REFINEMENT_TYPE: [strengthen|weaken|generalize|specialize|add_exception]",
                "REFINED_RULE:",
                "```asp",
                "[your refined ASP rule here]",
                "```",
                "",
                "RATIONALE:",
                "[Explain why this refinement addresses the issues]",
                "",
                "EXPECTED_IMPACT:",
                "[Describe how this should improve performance]",
                "",
                "TEST_CASES:",
                "1. [Test case to validate the refinement]",
                "2. [Another test case]",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_refinement_response(
        self,
        response: str,
        rule_id: str,
        original_rule: str,
        issues: List[PerformanceIssue],
    ) -> RefinementProposal:
        """
        Parse LLM response into RefinementProposal.

        Args:
            response: Raw LLM response
            rule_id: Original rule ID
            original_rule: Original ASP rule
            issues: Issues being addressed

        Returns:
            RefinementProposal
        """
        # Extract refinement type
        refinement_type = self._extract_field(response, "REFINEMENT_TYPE:")
        if not refinement_type:
            refinement_type = "general_refinement"

        # Extract refined rule
        refined_rule = self._extract_code_block(response, "REFINED_RULE:")
        if not refined_rule:
            # Fallback: use original rule if extraction failed
            refined_rule = original_rule

        # Extract rationale
        rationale = self._extract_field(response, "RATIONALE:")
        if not rationale:
            rationale = "LLM-proposed refinement based on performance feedback"

        # Extract expected impact
        expected_impact = self._extract_field(response, "EXPECTED_IMPACT:")
        if not expected_impact:
            expected_impact = "Improved accuracy on identified failure cases"

        # Extract test cases
        test_cases = self._extract_test_cases(response)

        # Estimate confidence based on response quality
        confidence = self._estimate_confidence(response, refined_rule, original_rule)

        return RefinementProposal(
            original_rule_id=rule_id,
            proposed_asp_rule=refined_rule.strip(),
            refinement_type=refinement_type.strip(),
            rationale=rationale.strip(),
            expected_impact=expected_impact.strip(),
            confidence=confidence,
            test_cases=test_cases,
            issues_addressed=issues,
        )

    def _extract_field(self, text: str, field_marker: str) -> str:
        """Extract a field from the response text."""
        if field_marker not in text:
            return ""

        start = text.find(field_marker) + len(field_marker)
        # Find next field marker or end of text
        markers = [
            "REFINEMENT_TYPE:",
            "REFINED_RULE:",
            "RATIONALE:",
            "EXPECTED_IMPACT:",
            "TEST_CASES:",
        ]
        end_positions = [
            text.find(m, start) for m in markers if text.find(m, start) > start
        ]
        end = min(end_positions) if end_positions else len(text)

        return text[start:end].strip()

    def _extract_code_block(self, text: str, field_marker: str) -> str:
        """Extract ASP code from markdown code block."""
        if field_marker not in text:
            return ""

        start = text.find(field_marker)
        section = text[start:]

        # Find ```asp or ``` after the marker
        code_start = section.find("```")
        if code_start == -1:
            return ""

        # Skip past the opening ```
        code_start = section.find("\n", code_start) + 1
        code_end = section.find("```", code_start)

        if code_end == -1:
            return ""

        return section[code_start:code_end].strip()

    def _extract_test_cases(self, text: str) -> List[str]:
        """Extract test cases from the response."""
        if "TEST_CASES:" not in text:
            return []

        start = text.find("TEST_CASES:") + len("TEST_CASES:")
        section = text[start:].strip()

        test_cases = []
        for line in section.split("\n"):
            line = line.strip()
            # Look for numbered test cases
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove number/bullet prefix
                test_case = line.lstrip("0123456789.-) ").strip()
                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _estimate_confidence(
        self, response: str, refined_rule: str, original_rule: str
    ) -> float:
        """
        Estimate confidence in the refinement proposal.

        Higher confidence if:
        - Response is well-structured
        - Refined rule is different from original
        - Rationale is provided
        - Test cases are included
        """
        confidence = 0.5  # Base confidence

        # Check for well-structured response
        if all(
            marker in response
            for marker in ["REFINEMENT_TYPE:", "REFINED_RULE:", "RATIONALE:"]
        ):
            confidence += 0.2

        # Check if rule was actually changed
        if refined_rule and refined_rule != original_rule:
            confidence += 0.15

        # Check for rationale
        if (
            "RATIONALE:" in response
            and len(self._extract_field(response, "RATIONALE:")) > 50
        ):
            confidence += 0.1

        # Check for test cases
        if "TEST_CASES:" in response:
            confidence += 0.05

        return min(confidence, 1.0)
