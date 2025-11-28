"""
Versioned prompt templates for LLM rule generation.

This module contains all prompt templates used to generate ASP rules from
natural language, organized by scenario and version for A/B testing and
performance tracking.
"""

# Version format: TEMPLATE_NAME_V{major}_{minor}
# Increment minor for prompt refinements, major for structural changes

# =============================================================================
# PRINCIPLE TO RULE PROMPTS
# =============================================================================

PRINCIPLE_TO_RULE_V1_0 = """You are an expert in converting legal principles into Answer Set Programming (ASP) rules.

Your task is to convert the following legal principle into one or more ASP rules in Clingo syntax.

**Legal Principle:**
{principle_text}

**Domain:** {domain}

**Existing Predicates:**
{existing_predicates}

**Guidelines:**
1. Use only the existing predicates listed above, or define new predicates if absolutely necessary
2. Follow ASP/Clingo syntax strictly (rules use `:-`, facts end with `.`)
3. Use descriptive predicate and variable names
4. Handle exceptions and edge cases appropriately
5. Use stratified negation (`not`) for default reasoning
6. Provide clear variable naming (e.g., `C` for contract, `P` for party)

**Output Requirements:**
- Generate the ASP rule(s)
- Explain your reasoning
- List all predicates used
- List any new predicates introduced
- Provide alternative formulations if the principle is ambiguous
- Include a confidence score (0.0-1.0) indicating your certainty
- If from a specific jurisdiction or source, include citation

**Example:**
Principle: "A contract is enforceable if it is not void and satisfies all legal requirements."
ASP Rule: `enforceable(C) :- contract(C), not void(C), satisfies_requirements(C).`

Now convert the given principle above into ASP rule(s).
"""

PRINCIPLE_TO_RULE_V1_1 = """You are an expert legal knowledge engineer specializing in Answer Set Programming (ASP).

**Task:** Convert a legal principle into formal ASP rules in Clingo syntax.

**Input:**
- **Principle:** {principle_text}
- **Domain:** {domain}
- **Available Predicates:** {existing_predicates}
{constraints}

**ASP Syntax Requirements:**
- Rules: `head :- body1, body2, ..., bodyN.`
- Facts: `predicate(args).`
- Negation as failure: `not predicate(args)`
- Variables: Uppercase (C, P, W)
- Constants: Lowercase (alice, bob, c1)

**Generation Strategy:**
1. Identify the main legal condition and conclusion
2. Map condition to body predicates, conclusion to head predicate
3. Handle exceptions using negation (`not exception(C)`)
4. Consider edge cases and corner scenarios
5. Stratify rules to avoid circular dependencies

**Quality Criteria:**
- Correctness: Rule accurately captures the principle
- Completeness: All aspects of the principle are covered
- Consistency: Rule doesn't contradict existing knowledge
- Clarity: Predicate names are self-documenting

**Output Format:**
Respond with a structured JSON object containing:
- `asp_rule`: The main ASP rule(s)
- `confidence`: Your confidence (0.0-1.0)
- `reasoning`: Step-by-step explanation
- `predicates_used`: List of existing predicates referenced
- `new_predicates`: List of new predicates defined
- `alternative_formulations`: Other valid formulations
- `source_type`: "principle"
- `source_text`: The original principle
- `citation`: Legal citation if applicable
- `jurisdiction`: Jurisdiction if applicable

Convert the principle now.
"""

# =============================================================================
# CASE LAW TO RULE PROMPTS
# =============================================================================

CASE_TO_RULE_V1_0 = """You are an expert in extracting legal rules from judicial opinions.

Your task is to read the case excerpt below and extract the holding as an ASP rule.

**Case Excerpt:**
{case_text}

**Citation:** {citation}
**Jurisdiction:** {jurisdiction}
**Domain:** {domain}

**Available Predicates:**
{existing_predicates}

{focus}

**Instructions:**
1. Identify the key holding or rule of law from this case
2. Express it as one or more ASP rules in Clingo syntax
3. Focus on the legal principle, not the specific facts
4. Generalize appropriately while preserving legal accuracy
5. Use existing predicates where possible
6. Define new predicates only if necessary for clarity

**Output Requirements:**
- ASP rule(s) capturing the holding
- Explanation of how you extracted the rule from the case
- List predicates used (existing + new)
- Alternative formulations if the holding is ambiguous
- Confidence score (0.0-1.0)
- Full citation and jurisdiction

**Example:**
Case: "The court held that part performance of an oral contract for land sale can satisfy the statute of frauds if the buyer took possession, made improvements, and paid consideration."

ASP Rule:
```
satisfies_statute_of_frauds(C) :-
    land_sale_contract(C),
    part_performance(C),
    substantial_actions_taken(C),
    detrimental_reliance(C).
```

Now extract the rule from the case above.
"""

# =============================================================================
# GAP FILLING PROMPTS
# =============================================================================

GAP_FILLING_V1_0 = """You are an expert in automated reasoning and knowledge representation.

The symbolic reasoning core has identified a knowledge gap that prevents it from making a determination.

**Gap Description:**
{gap_description}

**Missing Predicate:**
{missing_predicate}

**Existing Predicates:**
{existing_predicates}

**Context:**
{context}

**Task:**
Generate one or more candidate ASP rules that define the missing predicate to fill this knowledge gap.

**Requirements:**
1. Each candidate rule should be a complete, valid ASP rule
2. Rules should be consistent with the existing knowledge base
3. Consider multiple approaches (strict vs. permissive, different conditions)
4. Provide test cases that would validate each candidate

**Output Format:**
Return a JSON object with:
- `gap_description`: The gap being addressed
- `missing_predicate`: The predicate being defined
- `candidates`: List of candidate rules (each with rule, confidence, applicability)
- `recommended_index`: Index of your recommended candidate (0-based)
- `requires_validation`: Boolean - does this need human review?
- `test_cases_needed`: List of test scenarios to validate the rules
- `confidence`: Overall confidence in the solution

**Evaluation Criteria:**
- Coverage: Does the rule handle all relevant cases?
- Precision: Does it avoid over-generalization?
- Consistency: Does it align with existing rules?
- Testability: Can we validate it empirically?

Generate the candidate rules now.
"""

GAP_FILLING_V1_1 = """You are a knowledge engineer specializing in filling gaps in formal knowledge bases.

**Knowledge Gap Detected:**
{gap_description}

**Missing:** {missing_predicate}
**Context:** {context}
**Available Predicates:** {existing_predicates}

**Your Task:**
Design ASP rules to define `{missing_predicate}` such that the reasoning system can continue.

**Strategy:**
1. Analyze what `{missing_predicate}` should mean in this domain
2. Identify necessary and sufficient conditions
3. Generate 2-4 candidate formulations with different trade-offs:
   - Conservative (high precision, may miss edge cases)
   - Permissive (high recall, may over-trigger)
   - Balanced (middle ground)
   - Context-specific (optimized for this gap)

4. For each candidate:
   - Estimate applicability (how well it addresses this specific gap)
   - Estimate complexity (simpler is better if equally accurate)
   - Provide test cases that would validate/invalidate it

**Output:**
Structure your response as a JSON object matching the GapFillingResponse schema:
- Multiple candidates with different approaches
- Recommend the best one (index)
- Flag if human validation is needed
- Provide specific test cases

Be thorough but practical - we need rules that work, not perfect formalisms.
"""

# =============================================================================
# CONSENSUS VOTING PROMPTS
# =============================================================================

CONSENSUS_VOTE_V1_0 = """You are a critical reviewer evaluating a proposed ASP rule for correctness and quality.

**Proposed Rule:**
```asp
{proposed_rule}
```

**Proposer's Reasoning:**
{proposer_reasoning}

**Context:**
- Domain: {domain}
- Source: {source_type}
- Existing predicates: {existing_predicates}

**Your Task:**
Vote on whether to ACCEPT, REJECT, or REVISE this rule.

**Evaluation Criteria:**
1. **Syntactic Validity:** Is the ASP syntax correct?
2. **Semantic Correctness:** Does the rule capture the intended meaning?
3. **Consistency:** Could this contradict existing knowledge?
4. **Completeness:** Are there missing cases or exceptions?
5. **Clarity:** Are predicate and variable names clear?
6. **Stratification:** Does it avoid problematic recursion?

**Vote Options:**
- **ACCEPT:** Rule is correct and ready for use
- **REJECT:** Rule has fundamental flaws, should be discarded
- **REVISE:** Rule is mostly correct but needs modifications

**Output Format:**
Respond with a JSON object containing:
- `vote`: "accept", "reject", or "revise"
- `confidence`: Your confidence in this vote (0.0-1.0)
- `issues_found`: List of specific issues (can be empty if accepting)
- `suggested_revision`: If vote="revise", provide corrected ASP rule
- `test_cases_to_validate`: Test cases that should be run
- `reasoning`: Detailed explanation of your vote

Be rigorous but fair. The goal is correct, usable rules, not perfection.

Cast your vote now.
"""

CONSENSUS_VOTE_V1_1 = """You are a formal methods expert reviewing a proposed knowledge base rule.

**Rule Under Review:**
```asp
{proposed_rule}
```

**Justification:**
{proposer_reasoning}

**Metadata:**
- Domain: {domain}
- Source: {source_type}
- Available predicates: {existing_predicates}

**Review Process:**

**Step 1: Syntax Check**
- Is the ASP/Clingo syntax valid?
- Are variables properly capitalized?
- Are operators correct (`:-, not, .`)?

**Step 2: Semantic Analysis**
- Does the rule express a coherent logical statement?
- Are the predicates used appropriately?
- Are there type errors (e.g., using a contract predicate with a person variable)?

**Step 3: Consistency Check**
- Could this rule create contradictions with existing knowledge?
- Does it introduce unsafe negation or unstratified recursion?
- Are there edge cases that could cause issues?

**Step 4: Completeness Check**
- Does the rule handle all relevant scenarios?
- Are there missing conditions or exceptions?
- Should there be multiple rules instead of one?

**Step 5: Quality Assessment**
- Is the rule as simple as possible while remaining correct?
- Could it be more readable or maintainable?
- Are there alternative formulations that would be better?

**Decision:**
Based on your analysis, vote:
- **ACCEPT** if the rule passes all checks
- **REVISE** if it needs modifications but the core is sound
- **REJECT** if it has fundamental flaws

**Output:**
Provide a structured JSON response with your vote, confidence, identified issues, suggested revision (if any), test cases for validation, and detailed reasoning.

Conduct your review now.
"""

# =============================================================================
# RULE REPAIR PROMPTS
# =============================================================================

RULE_REPAIR_V1_0 = """You are an expert ASP (Answer Set Programming) debugger.

**Problem:** The following ASP rule was generated but has syntax or completeness issues.

**Malformed Rule:**
```asp
{malformed_rule}
```

**Error Message:**
{error_message}

**Original Principle/Context:**
{original_context}

**Available Predicates from Dataset:**
{available_predicates}

**Your Task:**
1. Identify what's wrong with the rule (truncation, syntax error, etc.)
2. Reconstruct the COMPLETE, VALID ASP rule
3. Use predicates that match the available predicates from the dataset

**CRITICAL Requirements:**
- The rule MUST end with a period '.'
- All parentheses MUST be balanced
- All predicates MUST be complete (no truncated names)
- Use predicates that exist in the dataset facts
- Follow proper Clingo ASP syntax

**Output Format:**
Return a GeneratedRule JSON object with:
- `asp_rule`: The corrected, complete ASP rule
- `confidence`: Your confidence in the fix (0.0-1.0)
- `reasoning`: Explanation of what was wrong and how you fixed it
- `predicates_used`: List of predicates in the rule
- `new_predicates`: Any new predicates introduced
- `alternative_formulations`: Empty list (not needed for repair)
- `source_type`: "refinement"
- `source_text`: The original context

Fix the rule now.
"""

RULE_REPAIR_V1_1 = """You are an expert ASP rule repair specialist.

**Broken Rule:**
```
{malformed_rule}
```

**Error:** {error_message}

**Context:** {original_context}

**Dataset Predicates (USE THESE):**
{available_predicates}

**Fix Requirements:**
1. Complete any truncated predicates
2. Ensure rule ends with '.'
3. Balance all parentheses
4. Match predicate names to dataset predicates above
5. Use proper variable naming (uppercase: C, P, X)

**Common Fixes Needed:**
- Truncated: `occupation_conti` → `occupation_continuous(X, yes)`
- Missing period: `pred(X)` → `pred(X).`
- Unbalanced: `pred(X, Y` → `pred(X, Y)`

Respond with a complete GeneratedRule JSON.
"""

# =============================================================================
# PREDICATE-ALIGNED GENERATION PROMPTS
# =============================================================================

ALIGNED_PRINCIPLE_TO_RULE_V1_0 = """You are an expert in converting legal principles into Answer Set Programming (ASP) rules.

**CRITICAL:** You must generate rules that use predicates matching the dataset format below.

**Legal Principle:**
{principle_text}

**Domain:** {domain}

**Dataset Predicate Examples (USE THESE EXACT FORMATS):**
{dataset_predicates}

**Guidelines:**
1. Use EXACTLY the predicate formats shown in the dataset examples
2. Match the arity (number of arguments) of predicates exactly
3. Use the same argument patterns (e.g., `claim(X)` not `claim(X, Y)`)
4. The head predicate should be `enforceable(X)` or `unenforceable(X)` to match expected outcomes
5. Variables should be uppercase (X, C, P)

**Example Mapping:**
- If dataset has `occupation_continuous(claim1, yes)`, use `occupation_continuous(X, yes)` not `continuous_occupation(X)`
- If dataset has `statutory_period(claim1, 20)`, use `statutory_period(X, P)` not `period(X)`

**Output:**
Generate a single, complete ASP rule that:
1. Uses predicates from the dataset
2. Ends with a period
3. Has balanced parentheses
4. Derives `enforceable(X)` or `unenforceable(X)`

Respond with a GeneratedRule JSON object.
"""

# =============================================================================
# MULTI-LLM REFINEMENT PROMPTS
# =============================================================================

REFINEMENT_V1_0 = """You are reviewing a rule that has received mixed votes from other reviewers.

**Original Rule:**
```asp
{original_rule}
```

**Votes Received:**
{votes_summary}

**Common Issues Identified:**
{common_issues}

**Task:**
Synthesize the feedback and produce an improved version of the rule.

**Instructions:**
1. Address the common issues identified
2. Preserve the correct aspects of the original rule
3. Balance different reviewer perspectives
4. Ensure the refined rule is syntactically and semantically valid

**Output:**
Provide a GeneratedRule object with:
- Refined ASP rule
- Explanation of what changed and why
- Confidence in the refinement
- Source type: "refinement"

Create the refined rule now.
"""

# =============================================================================
# PROMPT TEMPLATE REGISTRY
# =============================================================================

PROMPT_VERSIONS = {
    "principle_to_rule": {
        "v1.0": PRINCIPLE_TO_RULE_V1_0,
        "v1.1": PRINCIPLE_TO_RULE_V1_1,
        "latest": "v1.1",
    },
    "aligned_principle_to_rule": {
        "v1.0": ALIGNED_PRINCIPLE_TO_RULE_V1_0,
        "latest": "v1.0",
    },
    "case_to_rule": {
        "v1.0": CASE_TO_RULE_V1_0,
        "latest": "v1.0",
    },
    "gap_filling": {
        "v1.0": GAP_FILLING_V1_0,
        "v1.1": GAP_FILLING_V1_1,
        "latest": "v1.1",
    },
    "consensus_vote": {
        "v1.0": CONSENSUS_VOTE_V1_0,
        "v1.1": CONSENSUS_VOTE_V1_1,
        "latest": "v1.1",
    },
    "rule_repair": {
        "v1.0": RULE_REPAIR_V1_0,
        "v1.1": RULE_REPAIR_V1_1,
        "latest": "v1.1",
    },
    "refinement": {
        "v1.0": REFINEMENT_V1_0,
        "latest": "v1.0",
    },
}


def get_prompt(template_name: str, version: str = "latest") -> str:
    """
    Get a prompt template by name and version.

    Args:
        template_name: Name of the template (e.g., "principle_to_rule")
        version: Version string (e.g., "v1.0") or "latest"

    Returns:
        The prompt template string

    Raises:
        KeyError: If template or version doesn't exist
    """
    if template_name not in PROMPT_VERSIONS:
        raise KeyError(
            f"Unknown template: {template_name}. Available: {list(PROMPT_VERSIONS.keys())}"
        )

    versions = PROMPT_VERSIONS[template_name]

    if version == "latest":
        version = versions["latest"]

    if version not in versions:
        raise KeyError(
            f"Unknown version {version} for {template_name}. Available: {list(versions.keys())}"
        )

    return versions[version]


def list_templates() -> dict:
    """
    List all available templates and their versions.

    Returns:
        Dictionary mapping template names to version info
    """
    return {
        name: {"versions": list(v.keys()), "latest": v["latest"]}
        for name, v in PROMPT_VERSIONS.items()
    }
