"""
Synthesizer LLM agent for dialectical debate (Phase 4.2).

Combines insights from generator (thesis) and critic (antithesis) to create
improved rules (synthesis).
"""

import json
from typing import Any, List, Optional

from loguru import logger

from loft.dialectical.critique_schemas import CritiqueReport
from loft.dialectical.debate_schemas import DebateArgument
from loft.neural.rule_schemas import GeneratedRule


class Synthesizer:
    """
    Synthesizer agent that combines thesis and antithesis into synthesis.

    In dialectical reasoning:
    - Thesis: Generator's proposed rule
    - Antithesis: Critic's identified flaws
    - Synthesis: Improved rule that addresses flaws while preserving intent
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        mock_mode: bool = False,
    ):
        """
        Initialize synthesizer.

        Args:
            llm_client: LLM client for synthesis (if None, mock mode)
            mock_mode: If True, use mock synthesis for testing
        """
        self.llm_client = llm_client
        self.mock_mode = mock_mode or llm_client is None

        if self.mock_mode:
            logger.warning("Synthesizer running in mock mode (no real LLM)")

        logger.info(f"Initialized Synthesizer (mock: {self.mock_mode})")

    def synthesize(
        self,
        thesis: GeneratedRule,
        antithesis: CritiqueReport,
        existing_rules: List[str],
        context: Optional[str] = None,
    ) -> tuple[GeneratedRule, DebateArgument]:
        """
        Synthesize improved rule from thesis and antithesis.

        Args:
            thesis: Original proposed rule
            antithesis: Critique identifying flaws
            existing_rules: Existing rules for context
            context: Additional domain context

        Returns:
            Tuple of (synthesized_rule, argument explaining synthesis)
        """
        logger.info(f"Synthesizing from thesis: {thesis.asp_rule[:60]}...")

        if self.mock_mode:
            return self._mock_synthesis(thesis, antithesis, existing_rules)

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(thesis, antithesis, existing_rules, context)

        try:
            logger.debug("Calling LLM for synthesis")
            response = self._call_llm(prompt)
            logger.debug("Parsing LLM response as JSON")

            # Extract JSON from response (may be wrapped in markdown)
            json_str = self._extract_json(response)
            data = json.loads(json_str, strict=False)

            # Normalize ASP rule (remove extra whitespace, newlines)
            asp_rule = data["synthesized_rule"].replace('\n', ' ').replace('\r', '')
            asp_rule = ' '.join(asp_rule.split())  # Collapse multiple spaces

            # Create synthesized rule
            synthesis_rule = GeneratedRule(
                asp_rule=asp_rule,
                confidence=data.get("confidence", 0.8),
                reasoning=data.get("reasoning", "Synthesized from debate"),
                predicates_used=thesis.predicates_used,
                source_type="refinement",
                source_text=f"Synthesis of: {thesis.asp_rule}",
            )

            # Create synthesis argument
            argument = DebateArgument(
                speaker="synthesizer",
                content=data.get("argument", "Combined thesis and antithesis"),
                references=[thesis.asp_rule, str(len(antithesis.issues))],
                confidence=data.get("confidence", 0.8),
            )

            logger.info("Successfully synthesized improved rule")
            return synthesis_rule, argument

        except Exception as e:
            logger.error(f"Failed to synthesize: {e}")
            # Fall back to thesis
            return thesis, DebateArgument(
                speaker="synthesizer",
                content=f"Synthesis failed: {e}. Keeping original thesis.",
                confidence=0.3,
            )

    def _build_synthesis_prompt(
        self,
        thesis: GeneratedRule,
        antithesis: CritiqueReport,
        existing_rules: List[str],
        context: Optional[str] = None,
    ) -> str:
        """Build prompt for synthesis."""
        issues_summary = "\n".join(
            [
                f"- {issue.description} (severity: {issue.severity.value})"
                for issue in antithesis.issues
            ]
        )

        edge_cases_summary = "\n".join([f"- {ec.description}" for ec in antithesis.edge_cases])

        contradictions_summary = "\n".join(
            [f"- {c.description}" for c in antithesis.contradictions]
        )

        return f"""
You are a Synthesizer in a dialectical reasoning system. Your role is to combine
insights from both the Generator (thesis) and Critic (antithesis) to create an
improved rule (synthesis).

## Thesis (Proposed Rule)
{thesis.asp_rule}

**Reasoning:** {thesis.reasoning}
**Confidence:** {thesis.confidence}

## Antithesis (Critique)

**Issues Identified:**
{issues_summary or "None"}

**Edge Cases:**
{edge_cases_summary or "None"}

**Contradictions:**
{contradictions_summary or "None"}

**Recommendation:** {antithesis.recommendation}
{f"**Suggested Revision:** {antithesis.suggested_revision}" if antithesis.suggested_revision else ""}

## Existing Rules Context
{chr(10).join(existing_rules) if existing_rules else "No existing rules"}

## Your Task
Synthesize an improved ASP rule that:
1. Preserves the intent and purpose of the thesis
2. Addresses the issues identified in the antithesis
3. Handles the edge cases
4. Resolves any contradictions
5. Maintains compatibility with existing rules

Return a JSON object with:
- synthesized_rule: The improved ASP rule (single line, no newlines)
- reasoning: Explanation of how you combined thesis and antithesis
- argument: Your dialectical argument for this synthesis
- confidence: Your confidence in the synthesis (0.0-1.0)
- changes_made: List of specific changes from thesis

IMPORTANT: Return valid JSON with no newlines inside string values.

Example:
{{
  "synthesized_rule": "enforceable(C) :- contract(C), consideration(C), capacity(P1), capacity(P2), signed(C).",
  "reasoning": "Added consideration and capacity checks from critique while preserving contract basis",
  "argument": "The thesis correctly identified the need for a contract, but the critique rightly pointed out missing consideration and capacity requirements. This synthesis incorporates both perspectives.",
  "confidence": 0.85,
  "changes_made": ["Added consideration(C)", "Added capacity checks for both parties"]
}}
        """.strip()

    def _extract_json(self, response: str) -> str:
        """
        Extract JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        import re

        # Try to extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find JSON object in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # Return as-is and let JSON parser handle it
        return response

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if not self.llm_client:
            raise ValueError("No LLM client configured")

        logger.debug(
            f"LLM Request:\n{prompt[:500]}..." if len(prompt) > 500 else f"LLM Request:\n{prompt}"
        )

        # Use LLMInterface.query() method
        llm_response = self.llm_client.query(question=prompt)
        response = llm_response.raw_text

        logger.debug(
            f"LLM Response:\n{response[:500]}..."
            if len(response) > 500
            else f"LLM Response:\n{response}"
        )

        return response

    def _mock_synthesis(
        self,
        thesis: GeneratedRule,
        antithesis: CritiqueReport,
        existing_rules: List[str],
    ) -> tuple[GeneratedRule, DebateArgument]:
        """Mock synthesis for testing."""
        # Apply suggested revisions from critique
        synthesized = thesis.asp_rule

        changes = []
        if antithesis.suggested_revision:
            synthesized = antithesis.suggested_revision
            changes.append("Applied suggested revision from critique")
        else:
            # Try to add missing conditions
            for issue in antithesis.issues:
                if (
                    "consideration" in issue.description.lower()
                    and "consideration" not in synthesized
                ):
                    synthesized = synthesized.replace(".", ", consideration(C).")
                    changes.append("Added consideration requirement")
                if "capacity" in issue.description.lower() and "capacity" not in synthesized:
                    synthesized = synthesized.replace(".", ", capacity(P1), capacity(P2).")
                    changes.append("Added capacity checks")

        # If no changes, return thesis
        if synthesized == thesis.asp_rule:
            return thesis, DebateArgument(
                speaker="synthesizer",
                content="Thesis is already satisfactory. No synthesis needed.",
                confidence=thesis.confidence,
            )

        # Create synthesized rule
        synthesis_rule = GeneratedRule(
            asp_rule=synthesized,
            confidence=min(thesis.confidence + 0.1, 0.95),
            reasoning=f"Mock synthesis: {thesis.reasoning}. Changes: {', '.join(changes)}",
            predicates_used=thesis.predicates_used + ["consideration", "capacity"],
            source_type="refinement",
            source_text=f"Synthesized from: {thesis.source_text}",
        )

        argument = DebateArgument(
            speaker="synthesizer",
            content=f"Synthesized rule by addressing critique issues. Changes: {', '.join(changes)}",
            references=[thesis.asp_rule],
            confidence=synthesis_rule.confidence,
        )

        logger.info(f"Mock synthesis complete. Made {len(changes)} changes")
        return synthesis_rule, argument
