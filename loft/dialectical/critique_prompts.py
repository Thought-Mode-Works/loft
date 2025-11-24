"""
Specialized prompts for the critic LLM system.

These prompts guide the LLM to perform dialectical analysis of rules,
finding flaws, edge cases, and contradictions.
"""

from typing import List


def get_critique_prompt(rule: str, existing_rules: List[str], context: str = "") -> str:
    """
    Generate a prompt for comprehensive rule critique.

    Args:
        rule: The ASP rule to critique
        existing_rules: List of existing rules in the system
        context: Additional context about the domain or purpose

    Returns:
        Formatted prompt for LLM critique
    """
    existing_rules_str = "\n".join(f"  {r}" for r in existing_rules)
    if not existing_rules_str:
        existing_rules_str = "  (no existing rules)"

    return f"""You are a critical analyst specializing in logic and formal reasoning. Your role is to find flaws, edge cases, and contradictions in proposed rules.

PROPOSED RULE:
{rule}

EXISTING RULES:
{existing_rules_str}

CONTEXT:
{context if context else "Contract law domain - rules must be sound and comprehensive"}

TASK:
Provide a thorough critique of the proposed rule. Identify:

1. LOGICAL FLAWS: Any errors in reasoning, circular logic, or invalid inferences
2. MISSING CONDITIONS: Important conditions or constraints that should be included
3. EDGE CASES: Scenarios where the rule might fail or produce unexpected results
4. CONTRADICTIONS: Conflicts with existing rules
5. SCOPE ISSUES: Whether the rule is too broad, too narrow, or ill-defined

For each issue found, provide:
- Category (logical_flaw, missing_condition, too_broad, too_narrow, contradiction)
- Detailed description
- Severity (low, medium, high, critical)
- Suggested fix

Format your response as JSON with this structure:
{{
  "issues": [
    {{
      "category": "string",
      "description": "string",
      "severity": "low|medium|high|critical",
      "suggestion": "string"
    }}
  ],
  "edge_cases": [
    {{
      "description": "string",
      "scenario": {{}},
      "expected_outcome": "string",
      "severity": "low|medium|high|critical"
    }}
  ],
  "contradictions": [
    {{
      "description": "string",
      "conflicting_rule": "string",
      "example_scenario": {{}},
      "severity": "low|medium|high|critical",
      "resolution_suggestion": "string"
    }}
  ],
  "overall_severity": "low|medium|high|critical",
  "recommendation": "accept|revise|reject",
  "suggested_revision": "string or null",
  "confidence": 0.0-1.0
}}

Be thorough but fair. Not every rule is fatally flawed. Focus on substantive issues."""


def get_edge_case_prompt(rule: str, context: str = "") -> str:
    """
    Generate a prompt specifically for edge case identification.

    Args:
        rule: The ASP rule to analyze
        context: Additional context about the domain

    Returns:
        Formatted prompt for edge case generation
    """
    return f"""You are an expert at identifying edge cases and boundary conditions in formal rules.

RULE TO ANALYZE:
{rule}

CONTEXT:
{context if context else "Contract law domain"}

TASK:
Generate a comprehensive list of edge cases that might break or challenge this rule.
Consider:

1. BOUNDARY CONDITIONS: What happens at the extremes?
2. NULL/MISSING DATA: What if required information is absent?
3. UNUSUAL COMBINATIONS: Rare but valid scenarios
4. TEMPORAL ISSUES: Time-dependent edge cases
5. CONTRADICTORY FACTS: What if inputs conflict?
6. LEGAL EDGE CASES: Special circumstances in contract law

For each edge case, provide:
- Clear description of the scenario
- Structured representation of the facts
- Expected outcome
- Why this is challenging for the rule

Format as JSON:
{{
  "edge_cases": [
    {{
      "description": "string",
      "scenario": {{"fact1": "value", "fact2": "value"}},
      "expected_outcome": "string",
      "current_outcome": "string",
      "severity": "low|medium|high|critical"
    }}
  ]
}}

Aim for 5-10 diverse, realistic edge cases."""


def get_contradiction_check_prompt(new_rule: str, existing_rules: List[str]) -> str:
    """
    Generate a prompt for contradiction detection.

    Args:
        new_rule: The new rule being proposed
        existing_rules: List of existing rules to check against

    Returns:
        Formatted prompt for contradiction detection
    """
    existing_rules_str = "\n".join(f"  {r}" for r in existing_rules)
    if not existing_rules_str:
        return None  # No contradictions possible with no existing rules

    return f"""You are a logic expert specializing in contradiction detection in formal rule systems.

NEW RULE:
{new_rule}

EXISTING RULES:
{existing_rules_str}

TASK:
Identify any contradictions between the new rule and existing rules.

A contradiction exists when:
- The rules produce conflicting conclusions for the same input
- One rule's conditions negate another's
- Rules create logical inconsistency in the system

For each contradiction found, provide:
- Description of the contradiction
- The conflicting existing rule
- A concrete example scenario demonstrating the conflict
- Severity of the contradiction
- Suggested resolution

Format as JSON:
{{
  "contradictions": [
    {{
      "description": "string",
      "conflicting_rule": "string",
      "example_scenario": {{}},
      "severity": "low|medium|high|critical",
      "resolution_suggestion": "string"
    }}
  ]
}}

If no contradictions exist, return an empty list.
Be precise - only report actual logical contradictions, not just related rules."""


def get_synthesis_prompt(rule: str, critique: str, existing_rules: List[str]) -> str:
    """
    Generate a prompt for synthesizing an improved rule based on critique.

    Args:
        rule: The original rule
        critique: The critique report
        existing_rules: Existing rules for context

    Returns:
        Formatted prompt for rule synthesis
    """
    existing_rules_str = "\n".join(f"  {r}" for r in existing_rules)

    return f"""You are a rule synthesis expert. Your role is to improve rules based on critique.

ORIGINAL RULE:
{rule}

CRITIQUE:
{critique}

EXISTING RULES (for context):
{existing_rules_str}

TASK:
Synthesize an improved version of the rule that addresses the critique while:
1. Maintaining the original intent
2. Fixing identified flaws
3. Handling edge cases
4. Avoiding contradictions
5. Using proper ASP syntax

Provide:
- The improved rule in ASP syntax
- Explanation of what changed and why
- Confidence in the improvement (0.0-1.0)

Format as JSON:
{{
  "improved_rule": "string (ASP syntax)",
  "changes": "string (explanation)",
  "confidence": 0.0-1.0,
  "addresses_issues": ["issue1", "issue2", ...]
}}

The improved rule must be valid ASP syntax and logically sound."""
