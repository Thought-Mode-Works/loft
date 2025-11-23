"""
Prompt template system for LLM queries.

Provides versioned, reusable prompt templates with variable injection
and few-shot example management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from string import Template


@dataclass
class PromptTemplate:
    """
    Reusable prompt template with variable substitution.

    Templates use $variable syntax for substitution.
    """

    name: str
    version: str
    template: str
    description: str = ""
    required_variables: List[str] = field(default_factory=list)
    few_shot_examples: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def render(self, variables: Dict[str, Any]) -> str:
        """
        Render template with provided variables.

        Args:
            variables: Dictionary of variable values

        Returns:
            Rendered template string

        Raises:
            KeyError: If required variables are missing
        """
        # Check required variables
        missing = [v for v in self.required_variables if v not in variables]
        if missing:
            raise KeyError(f"Missing required variables: {missing}")

        # Substitute variables
        template = Template(self.template)
        return template.safe_substitute(variables)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "description": self.description,
            "required_variables": self.required_variables,
            "few_shot_examples": self.few_shot_examples,
            "system_prompt": self.system_prompt,
        }


# ============================================================================
# COMMON PROMPT TEMPLATES
# ============================================================================


GAP_IDENTIFICATION = PromptTemplate(
    name="gap_identification",
    version="1.0",
    description="Identify missing information needed to answer a question",
    template="""Given the following question and context, identify what information is missing to provide a complete answer.

Question: $question

Current Context:
$context

What additional information, facts, or rules are needed to answer this question completely?
List the gaps in knowledge.
""",
    required_variables=["question", "context"],
    system_prompt="You are an expert at identifying knowledge gaps in legal reasoning.",
)


ELEMENT_EXTRACTION = PromptTemplate(
    name="element_extraction",
    version="1.0",
    description="Extract legal elements from a fact pattern",
    template="""Analyze the following fact pattern and extract the relevant legal elements.

Fact Pattern:
$fact_pattern

Legal Doctrine:
$doctrine

Extract all elements present in the fact pattern that are relevant to the legal doctrine.
For each element, indicate whether it is:
- Clearly present
- Partially present
- Absent

Format your response as a structured analysis.
""",
    required_variables=["fact_pattern", "doctrine"],
    few_shot_examples=[
        """Example:
Fact Pattern: Alice and Bob signed a written contract for the sale of land.
Doctrine: Statute of Frauds

Elements:
- Contract exists: Clearly present (mentioned explicitly)
- Within statute (land sale): Clearly present
- Signed writing: Clearly present
- Essential terms: Partially present (need more detail)
- Part performance: Absent
"""
    ],
    system_prompt="You are an expert at legal element extraction and analysis.",
)


RULE_PROPOSAL = PromptTemplate(
    name="rule_proposal",
    version="1.0",
    description="Propose a rule to handle an edge case",
    template="""Given the following scenario and existing rules, propose a new rule to handle this edge case.

Scenario:
$scenario

Existing Rules:
$existing_rules

Propose a rule in Answer Set Programming (ASP) format that would handle this case.
Include:
1. The ASP rule syntax
2. Explanation of when it applies
3. Confidence level (0.0-1.0)
4. Suggested stratification level (constitutional, strategic, tactical, operational)

Your rule should be consistent with existing rules and legal principles.
""",
    required_variables=["scenario", "existing_rules"],
    system_prompt=(
        "You are an expert in legal reasoning and Answer Set Programming. "
        "Propose rules that are logically sound and legally defensible."
    ),
)


EXPLANATION_GENERATION = PromptTemplate(
    name="explanation",
    version="1.0",
    description="Explain why a conclusion follows from premises",
    template="""Explain why the following conclusion follows (or doesn't follow) from the given premises.

Premises:
$premises

Conclusion:
$conclusion

Provide a clear, step-by-step explanation of the reasoning. If the conclusion does not follow,
explain what is missing or incorrect.
""",
    required_variables=["premises", "conclusion"],
    system_prompt="You are an expert at legal reasoning and logical inference.",
)


CONFIDENCE_ASSESSMENT = PromptTemplate(
    name="confidence_assessment",
    version="1.0",
    description="Assess confidence in a legal conclusion",
    template="""Assess the confidence level for the following legal conclusion.

Fact Pattern:
$fact_pattern

Legal Rules:
$rules

Conclusion:
$conclusion

Provide:
1. Confidence score (0.0-1.0)
2. Reasoning for the confidence level
3. Key uncertainties or assumptions
4. What additional information would increase confidence
""",
    required_variables=["fact_pattern", "rules", "conclusion"],
    system_prompt=(
        "You are an expert at assessing legal reasoning confidence. "
        "Be calibrated and realistic about uncertainty."
    ),
)


CONSISTENCY_CHECK = PromptTemplate(
    name="consistency_check",
    version="1.0",
    description="Check if rules are mutually consistent",
    template="""Check whether the following rules are mutually consistent.

Rules:
$rules

Context:
$context

Determine if these rules can all be true simultaneously, or if there are contradictions.
If inconsistent, identify the conflicting rules and explain the contradiction.
""",
    required_variables=["rules", "context"],
    system_prompt="You are an expert at detecting logical inconsistencies in rule systems.",
)


COT_LEGAL_REASONING = PromptTemplate(
    name="cot_legal_reasoning",
    version="1.0",
    description="Chain-of-thought template for complex legal reasoning",
    template="""Let's analyze this legal question step-by-step.

Question: $question

Facts:
$facts

Relevant Law:
$law

Step-by-step analysis:
1. Identify the legal issue
2. State the applicable rules
3. Apply rules to facts
4. Address any exceptions or defenses
5. Reach a conclusion

Let's work through each step carefully.
""",
    required_variables=["question", "facts", "law"],
    system_prompt=(
        "You are an expert legal analyst. Use systematic, step-by-step reasoning "
        "to analyze complex legal questions."
    ),
)


ANALOGICAL_REASONING = PromptTemplate(
    name="analogical_reasoning",
    version="1.0",
    description="Reason by analogy from precedent cases",
    template="""Use analogical reasoning to analyze the current case based on precedent.

Current Case:
$current_case

Precedent Cases:
$precedents

For each precedent:
1. Identify key similarities to current case
2. Identify key differences
3. Determine whether precedent is controlling, persuasive, or distinguishable
4. Apply reasoning to current case

Conclude with how the precedents inform the analysis of the current case.
""",
    required_variables=["current_case", "precedents"],
    system_prompt=(
        "You are an expert at legal analogical reasoning and case analysis. "
        "Carefully distinguish material from immaterial differences."
    ),
)


# Template registry for easy access
TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    "gap_identification": GAP_IDENTIFICATION,
    "element_extraction": ELEMENT_EXTRACTION,
    "rule_proposal": RULE_PROPOSAL,
    "explanation": EXPLANATION_GENERATION,
    "confidence_assessment": CONFIDENCE_ASSESSMENT,
    "consistency_check": CONSISTENCY_CHECK,
    "cot_legal_reasoning": COT_LEGAL_REASONING,
    "analogical_reasoning": ANALOGICAL_REASONING,
}


def get_template(name: str) -> PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: Template name

    Returns:
        PromptTemplate instance

    Raises:
        KeyError: If template not found
    """
    if name not in TEMPLATE_REGISTRY:
        available = ", ".join(TEMPLATE_REGISTRY.keys())
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return TEMPLATE_REGISTRY[name]


def list_templates() -> List[str]:
    """List all available template names."""
    return list(TEMPLATE_REGISTRY.keys())
