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

GAP_FILLING_V1_2 = """You are a knowledge engineer specializing in filling gaps in formal knowledge bases.

**Knowledge Gap Detected:**
{gap_description}

**Missing:** {missing_predicate}
**Context:** {context}
**Available Predicates:** {existing_predicates}

**Your Task:**
Design ASP rules to define `{missing_predicate}` such that the reasoning system can continue.

**CRITICAL: Clingo ASP Syntax Requirements**

Clingo uses DIFFERENT arithmetic syntax than Python. Follow these rules STRICTLY:

**VALID Clingo Arithmetic:**
```asp
% Comparisons must be in constraints or rule bodies with ground arithmetic
amount_exceeds_threshold(X) :- contract(X), amount(X, A), threshold(T), A > T.
insufficient_payment(X) :- payment(X, P), required(X, R), P < R.
total_value(X, V) :- item_a(X, A), item_b(X, B), V = A + B.
percentage_met(X) :- actual(X, A), target(X, T), A * 100 >= T * 90.
```

**INVALID Syntax (DO NOT USE):**
```asp
% WRONG: Python-style infix multiplication with floats
amount_check(X) :- amount(X, A), A < 0.9 * 50000.

% WRONG: abs() function (not built-in in standard Clingo)
difference_check(X) :- actual(X, A), expected(X, E), abs(A - E) > 100.

% WRONG: Floating point numbers
threshold_check(X) :- value(X, V), V > 0.75 * max_value.

% WRONG: Unbound arithmetic operations
value_calc(X) :- X = 5 * Y.  % Y must be bound first
```

**CORRECT Patterns for Common Operations:**

1. **Percentage comparisons** - Use integer arithmetic:
   ```asp
   % "amount is at least 90% of target" becomes:
   meets_threshold(X) :- amount(X, A), target(X, T), A * 100 >= T * 90.
   ```

2. **Difference calculations** - Split into positive/negative cases:
   ```asp
   % Instead of abs(A - B) > Threshold:
   significant_difference(X) :- val_a(X, A), val_b(X, B), A > B, A - B > Threshold.
   significant_difference(X) :- val_a(X, A), val_b(X, B), B >= A, B - A > Threshold.
   ```

3. **Threshold comparisons** - Use predicates:
   ```asp
   above_threshold(X) :- amount(X, A), min_threshold(T), A >= T.
   below_threshold(X) :- amount(X, A), max_threshold(T), A <= T.
   ```

4. **Aggregates for counting/summing:**
   ```asp
   total_items(X, N) :- case(X), N = #count {{ I : item(X, I) }}.
   total_value(X, S) :- case(X), S = #sum {{ V, I : item(X, I), value(I, V) }}.
   ```

**Strategy:**
1. Analyze what `{missing_predicate}` should mean in this domain
2. Identify necessary and sufficient conditions
3. Generate 2-4 candidate formulations with different trade-offs:
   - Conservative (high precision, may miss edge cases)
   - Permissive (high recall, may over-trigger)
   - Balanced (middle ground)
   - Context-specific (optimized for this gap)

4. For each candidate:
   - Ensure ALL arithmetic uses valid Clingo syntax
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

# Gap-filling with dataset predicate alignment (issue #166)
GAP_FILLING_V1_3 = """You are a knowledge engineer specializing in filling gaps in formal knowledge bases.

**Knowledge Gap Detected:**
{gap_description}

**Missing:** {missing_predicate}
**Context:** {context}
**Available Predicates from Knowledge Base:** {existing_predicates}
{dataset_predicates_section}
**Your Task:**
Design ASP rules to define `{missing_predicate}` such that the reasoning system can continue.

**CRITICAL: Predicate Alignment Requirements**

Your generated rules MUST use the exact predicate patterns from the dataset. The dataset predicates
listed above are the ONLY predicates that will match actual case facts. Using different predicates
will result in rules that cannot be validated or applied.

**CRITICAL: Clingo ASP Syntax Requirements**

Clingo uses DIFFERENT arithmetic syntax than Python. Follow these rules STRICTLY:

**VALID Clingo Arithmetic:**
```asp
% Comparisons must be in constraints or rule bodies with ground arithmetic
amount_exceeds_threshold(X) :- contract(X), amount(X, A), threshold(T), A > T.
insufficient_payment(X) :- payment(X, P), required(X, R), P < R.
total_value(X, V) :- item_a(X, A), item_b(X, B), V = A + B.
percentage_met(X) :- actual(X, A), target(X, T), A * 100 >= T * 90.
```

**INVALID Syntax (DO NOT USE):**
```asp
% WRONG: Python-style infix multiplication with floats
amount_check(X) :- amount(X, A), A < 0.9 * 50000.

% WRONG: abs() function (not built-in in standard Clingo)
difference_check(X) :- actual(X, A), expected(X, E), abs(A - E) > 100.

% WRONG: Floating point numbers
threshold_check(X) :- value(X, V), V > 0.75 * max_value.

% WRONG: Unbound arithmetic operations
value_calc(X) :- X = 5 * Y.  % Y must be bound first
```

**CORRECT Patterns for Common Operations:**

1. **Percentage comparisons** - Use integer arithmetic:
   ```asp
   % "amount is at least 90% of target" becomes:
   meets_threshold(X) :- amount(X, A), target(X, T), A * 100 >= T * 90.
   ```

2. **Difference calculations** - Split into positive/negative cases:
   ```asp
   % Instead of abs(A - B) > Threshold:
   significant_difference(X) :- val_a(X, A), val_b(X, B), A > B, A - B > Threshold.
   significant_difference(X) :- val_a(X, A), val_b(X, B), B >= A, B - A > Threshold.
   ```

3. **Threshold comparisons** - Use predicates:
   ```asp
   above_threshold(X) :- amount(X, A), min_threshold(T), A >= T.
   below_threshold(X) :- amount(X, A), max_threshold(T), A <= T.
   ```

4. **Aggregates for counting/summing:**
   ```asp
   total_items(X, N) :- case(X), N = #count {{ I : item(X, I) }}.
   total_value(X, S) :- case(X), S = #sum {{ V, I : item(X, I), value(I, V) }}.
   ```

**Strategy:**
1. Analyze what `{missing_predicate}` should mean in this domain
2. Map required concepts to AVAILABLE DATASET PREDICATES
3. Generate 2-4 candidate formulations with different trade-offs:
   - Conservative (high precision, may miss edge cases)
   - Permissive (high recall, may over-trigger)
   - Balanced (middle ground)
   - Context-specific (optimized for this gap)

4. For each candidate:
   - Use ONLY predicates from the dataset predicate list
   - Ensure ALL arithmetic uses valid Clingo syntax
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

# Gap-filling with variable safety requirements (issue #167)
GAP_FILLING_V1_4 = """You are a knowledge engineer specializing in filling gaps in formal knowledge bases.

**Knowledge Gap Detected:**
{gap_description}

**Missing:** {missing_predicate}
**Context:** {context}
**Available Predicates from Knowledge Base:** {existing_predicates}
{dataset_predicates_section}
**Your Task:**
Design ASP rules to define `{missing_predicate}` such that the reasoning system can continue.

**CRITICAL: Predicate Alignment Requirements**

Your generated rules MUST use the exact predicate patterns from the dataset. The dataset predicates
listed above are the ONLY predicates that will match actual case facts. Using different predicates
will result in rules that cannot be validated or applied.

**CRITICAL: Clingo Variable Safety Requirements**

Clingo requires ALL variables in the rule HEAD to appear in at least one POSITIVE body literal.
This is called "variable safety" - variables must be "grounded" (bound to actual values) before
they can appear in the head.

**SAFE vs UNSAFE Variable Examples:**

```asp
% WRONG - Unsafe variable (Fall not bound in body):
cause_of_harm(X, Fall) :- dangerous_condition(X).
% ERROR: Variable 'Fall' appears in head but not in any positive body literal

% CORRECT - All head variables bound in body:
cause_of_harm(X, Fall) :- dangerous_condition(X), type_of_harm(X, Fall).
% 'Fall' is now bound by type_of_harm(X, Fall)

% CORRECT - Use constant if value is specific:
cause_of_harm(X, fall) :- dangerous_condition(X).
% 'fall' is lowercase (constant), not a variable

% WRONG - Unsafe variable in aggregation context:
total_harm(X, Type, N) :- N = #count {{ Y : harm(Y) }}.
% ERROR: X and Type are not bound in body

% CORRECT - Bind all variables:
total_harm(X, Type, N) :- case(X), harm_type(Type), N = #count {{ Y : harm(X, Y, Type) }}.
```

**Variable Safety Rules:**
1. Every UPPERCASE variable in the head MUST appear in at least one positive body atom
2. Variables in negative literals (`not pred(X)`) do NOT count as grounding
3. Variables in comparisons (`X > Y`) do NOT count as grounding on their own
4. Variables in aggregates must still be grounded elsewhere in the body
5. Use lowercase for constants (specific values), uppercase for variables

**CRITICAL: Clingo ASP Syntax Requirements**

Clingo uses DIFFERENT arithmetic syntax than Python. Follow these rules STRICTLY:

**VALID Clingo Arithmetic:**
```asp
% Comparisons must be in constraints or rule bodies with ground arithmetic
amount_exceeds_threshold(X) :- contract(X), amount(X, A), threshold(T), A > T.
insufficient_payment(X) :- payment(X, P), required(X, R), P < R.
total_value(X, V) :- item_a(X, A), item_b(X, B), V = A + B.
percentage_met(X) :- actual(X, A), target(X, T), A * 100 >= T * 90.
```

**INVALID Syntax (DO NOT USE):**
```asp
% WRONG: Python-style infix multiplication with floats
amount_check(X) :- amount(X, A), A < 0.9 * 50000.

% WRONG: abs() function (not built-in in standard Clingo)
difference_check(X) :- actual(X, A), expected(X, E), abs(A - E) > 100.

% WRONG: Floating point numbers
threshold_check(X) :- value(X, V), V > 0.75 * max_value.

% WRONG: Unbound arithmetic operations
value_calc(X) :- X = 5 * Y.  % Y must be bound first
```

**CORRECT Patterns for Common Operations:**

1. **Percentage comparisons** - Use integer arithmetic:
   ```asp
   % "amount is at least 90% of target" becomes:
   meets_threshold(X) :- amount(X, A), target(X, T), A * 100 >= T * 90.
   ```

2. **Difference calculations** - Split into positive/negative cases:
   ```asp
   % Instead of abs(A - B) > Threshold:
   significant_difference(X) :- val_a(X, A), val_b(X, B), A > B, A - B > Threshold.
   significant_difference(X) :- val_a(X, A), val_b(X, B), B >= A, B - A > Threshold.
   ```

3. **Threshold comparisons** - Use predicates:
   ```asp
   above_threshold(X) :- amount(X, A), min_threshold(T), A >= T.
   below_threshold(X) :- amount(X, A), max_threshold(T), A <= T.
   ```

4. **Aggregates for counting/summing:**
   ```asp
   total_items(X, N) :- case(X), N = #count {{ I : item(X, I) }}.
   total_value(X, S) :- case(X), S = #sum {{ V, I : item(X, I), value(I, V) }}.
   ```

**Strategy:**
1. Analyze what `{missing_predicate}` should mean in this domain
2. Map required concepts to AVAILABLE DATASET PREDICATES
3. Generate 2-4 candidate formulations with different trade-offs:
   - Conservative (high precision, may miss edge cases)
   - Permissive (high recall, may over-trigger)
   - Balanced (middle ground)
   - Context-specific (optimized for this gap)

4. For each candidate:
   - VERIFY all head variables appear in positive body literals
   - Use ONLY predicates from the dataset predicate list
   - Ensure ALL arithmetic uses valid Clingo syntax
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

# Gap-filling with punctuation rules (issue #168)
GAP_FILLING_V1_5 = """You are a knowledge engineer specializing in filling gaps in formal knowledge bases.

**Knowledge Gap Detected:**
{gap_description}

**Missing:** {missing_predicate}
**Context:** {context}
**Available Predicates from Knowledge Base:** {existing_predicates}
{dataset_predicates_section}
**Your Task:**
Design ASP rules to define `{missing_predicate}` such that the reasoning system can continue.

**CRITICAL: Predicate Alignment Requirements**

Your generated rules MUST use the exact predicate patterns from the dataset. The dataset predicates
listed above are the ONLY predicates that will match actual case facts. Using different predicates
will result in rules that cannot be validated or applied.

**CRITICAL: ASP Punctuation Rules (issue #168)**

ASP uses VERY SPECIFIC punctuation. Common LLM errors include using periods (.) where commas
should be used. Follow these rules STRICTLY:

**Punctuation in ASP:**
- `.` (period): ONLY used at the END of a complete rule or fact to terminate it
- `,` (comma): Separates predicates in rule bodies (conjunction - "and")
- `;` (semicolon): Separates alternatives in choice rules (disjunction - "or")
- `:-` (colon-hyphen): Separates the head from the body of a rule

**WRONG - Period used as separator (causes parse errors):**
```asp
% WRONG: Period between predicates (OOP-style dot notation)
physical_harm(Spectator.FoulBall) :- at_game(Spectator).
% ERROR: Clingo sees "Spectator.FoulBall" as invalid syntax

% WRONG: Period instead of comma
injured_by(X, Y) :- at_game(X). hit_by(X, Y).
% ERROR: This is TWO separate statements, not one rule
```

**CORRECT - Comma separates predicates:**
```asp
% CORRECT: Comma separates predicates in body
cause_of_harm(X, Type) :- dangerous_condition(X), type_of_harm(X, Type).

% CORRECT: Comma between all body literals
injured_by(Spectator, FoulBall) :- at_baseball_game(Spectator), foul_ball_strike(FoulBall), not_in_screened_section(Spectator), physical_harm(Spectator, FoulBall).
```

**Multi-argument predicates use commas, NOT periods:**
```asp
% WRONG: physical_harm(Spectator.FoulBall)
% CORRECT: physical_harm(Spectator, FoulBall)
```

**CRITICAL: Clingo Variable Safety Requirements**

Clingo requires ALL variables in the rule HEAD to appear in at least one POSITIVE body literal.
This is called "variable safety" - variables must be "grounded" (bound to actual values) before
they can appear in the head.

**SAFE vs UNSAFE Variable Examples:**

```asp
% WRONG - Unsafe variable (Fall not bound in body):
cause_of_harm(X, Fall) :- dangerous_condition(X).
% ERROR: Variable 'Fall' appears in head but not in any positive body literal

% CORRECT - All head variables bound in body:
cause_of_harm(X, Fall) :- dangerous_condition(X), type_of_harm(X, Fall).
% 'Fall' is now bound by type_of_harm(X, Fall)

% CORRECT - Use constant if value is specific:
cause_of_harm(X, fall) :- dangerous_condition(X).
% 'fall' is lowercase (constant), not a variable
```

**Variable Safety Rules:**
1. Every UPPERCASE variable in the head MUST appear in at least one positive body atom
2. Variables in negative literals (`not pred(X)`) do NOT count as grounding
3. Variables in comparisons (`X > Y`) do NOT count as grounding on their own
4. Variables in aggregates must still be grounded elsewhere in the body
5. Use lowercase for constants (specific values), uppercase for variables

**CRITICAL: Clingo ASP Syntax Requirements**

Clingo uses DIFFERENT arithmetic syntax than Python. Follow these rules STRICTLY:

**VALID Clingo Arithmetic:**
```asp
% Comparisons must be in constraints or rule bodies with ground arithmetic
amount_exceeds_threshold(X) :- contract(X), amount(X, A), threshold(T), A > T.
insufficient_payment(X) :- payment(X, P), required(X, R), P < R.
total_value(X, V) :- item_a(X, A), item_b(X, B), V = A + B.
percentage_met(X) :- actual(X, A), target(X, T), A * 100 >= T * 90.
```

**INVALID Syntax (DO NOT USE):**
```asp
% WRONG: Python-style infix multiplication with floats
amount_check(X) :- amount(X, A), A < 0.9 * 50000.

% WRONG: abs() function (not built-in in standard Clingo)
difference_check(X) :- actual(X, A), expected(X, E), abs(A - E) > 100.

% WRONG: Floating point numbers
threshold_check(X) :- value(X, V), V > 0.75 * max_value.

% WRONG: Unbound arithmetic operations
value_calc(X) :- X = 5 * Y.  % Y must be bound first
```

**CORRECT Patterns for Common Operations:**

1. **Percentage comparisons** - Use integer arithmetic:
   ```asp
   % "amount is at least 90% of target" becomes:
   meets_threshold(X) :- amount(X, A), target(X, T), A * 100 >= T * 90.
   ```

2. **Difference calculations** - Split into positive/negative cases:
   ```asp
   % Instead of abs(A - B) > Threshold:
   significant_difference(X) :- val_a(X, A), val_b(X, B), A > B, A - B > Threshold.
   significant_difference(X) :- val_a(X, A), val_b(X, B), B >= A, B - A > Threshold.
   ```

3. **Threshold comparisons** - Use predicates:
   ```asp
   above_threshold(X) :- amount(X, A), min_threshold(T), A >= T.
   below_threshold(X) :- amount(X, A), max_threshold(T), A <= T.
   ```

4. **Aggregates for counting/summing:**
   ```asp
   total_items(X, N) :- case(X), N = #count {{ I : item(X, I) }}.
   total_value(X, S) :- case(X), S = #sum {{ V, I : item(X, I), value(I, V) }}.
   ```

**Strategy:**
1. Analyze what `{missing_predicate}` should mean in this domain
2. Map required concepts to AVAILABLE DATASET PREDICATES
3. Generate 2-4 candidate formulations with different trade-offs:
   - Conservative (high precision, may miss edge cases)
   - Permissive (high recall, may over-trigger)
   - Balanced (middle ground)
   - Context-specific (optimized for this gap)

4. For each candidate:
   - VERIFY no embedded periods (use commas to separate predicates)
   - VERIFY all head variables appear in positive body literals
   - Use ONLY predicates from the dataset predicate list
   - Ensure ALL arithmetic uses valid Clingo syntax
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

**Legal Principle:**
{principle_text}

**Domain:** {domain}

**AVAILABLE PREDICATES FROM DATASET:**
{dataset_predicates}

**RULES FOR PREDICATE SELECTION:**
1. PREFER predicates from the list above when they match the concept
2. Match predicate names EXACTLY as shown (e.g., use `attachment_method(X, Y)` not `annexation(X, Y)`)
3. For yes/no predicates, use the exact format shown (e.g., `custom_built(X, yes)`)
4. If NO suitable predicate exists in the list for a required concept, you MAY create a new predicate
5. The head MUST be `enforceable(X)` or `unenforceable(X)`
6. Use uppercase variables: X, Y, N, P

**PREDICATE MATCHING EXAMPLES:**
- Dataset has `attachment_method(X, bolted)` → use this, NOT `annexation(X, yes)`
- Dataset has `custom_built(X, yes)` → use this, NOT `adaptation(X, custom)`
- Dataset has `occupation_continuous(X, yes)` → use this, NOT `continuous_occupation(X)`

**WHEN TO CREATE NEW PREDICATES:**
Only create a new predicate if the legal concept has NO logical match in the available predicates.
List any new predicates in the `new_predicates` field.

**OUTPUT:**
Generate ONE complete ASP rule that:
1. Prefers dataset predicates over invented ones
2. Ends with a period
3. Has balanced parentheses
4. Derives `enforceable(X)` or `unenforceable(X)`

Respond with a GeneratedRule JSON object.
"""

ALIGNED_PRINCIPLE_TO_RULE_V1_1 = """You are an expert in converting legal principles into GENERAL Answer Set Programming (ASP) rules.

**CRITICAL REQUIREMENT: USE ONLY THESE DATASET PREDICATES**

The ONLY predicates you may use are listed below. DO NOT invent new predicate names.
If a legal concept doesn't have a matching predicate, find the CLOSEST match from this list.

**AVAILABLE PREDICATES (USE THESE EXACTLY):**
{dataset_predicates}

**Legal Principle:**
{principle_text}

**Domain:** {domain}

**STRICT RULES:**

1. **USE ONLY DATASET PREDICATES**: Every predicate in your rule body MUST appear in the list above.
   - If the principle mentions "annexation", look for predicates like `built_in(X, yes)` or `attachment_method(X, Y)`
   - If the principle mentions "intent", look for predicates like `custom_built(X, yes)` or `built_in(X, yes)`
   - DO NOT create `annexed(X, yes)` or `intent_permanent(X, yes)` if they're not in the list

2. **MAP LEGAL CONCEPTS TO DATASET PREDICATES:**
   - "annexed/annexation" → use `attachment_method(X, Y)` or `built_in(X, yes)`
   - "adaptation/adapted" → use `custom_built(X, yes)`
   - "intent/intention" → use `built_in(X, yes)` or infer from other predicates
   - "continuous" → use `occupation_continuous(X, yes)`
   - "hostile/adverse" → use `occupation_hostile(X, yes)` or `use_adverse(X, yes)`

3. **GENERALIZE (no specific names):**
   - NEVER use specific names (Frank, Nancy, Alice) - use VARIABLES (X, Y, P)
   - Use uppercase for variables: X, Y, Z, N, Owner, Buyer

4. **SYNTAX:**
   - Head MUST be `enforceable(X)` or `unenforceable(X)`
   - Rule MUST end with a period
   - All parentheses must be balanced

**EXAMPLE:**

Legal principle: "The fixture test requires annexation, adaptation, and intent"
Available predicates: `attachment_method(X, Y)`, `custom_built(X, yes/no)`, `built_in(X, yes/no)`

WRONG (invents predicates):
```asp
enforceable(X) :- annexed(X, yes), adapted(X, yes), intent_permanent(X, yes).
```

CORRECT (uses available predicates):
```asp
enforceable(X) :- attachment_method(X, bolted), custom_built(X, yes), built_in(X, yes).
```

**OUTPUT:**
Generate ONE complete ASP rule using ONLY predicates from the list above.
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
        "v1.1": ALIGNED_PRINCIPLE_TO_RULE_V1_1,
        "latest": "v1.1",
    },
    "case_to_rule": {
        "v1.0": CASE_TO_RULE_V1_0,
        "latest": "v1.0",
    },
    "gap_filling": {
        "v1.0": GAP_FILLING_V1_0,
        "v1.1": GAP_FILLING_V1_1,
        "v1.2": GAP_FILLING_V1_2,
        "v1.3": GAP_FILLING_V1_3,
        "v1.4": GAP_FILLING_V1_4,
        "v1.5": GAP_FILLING_V1_5,
        "latest": "v1.5",
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
