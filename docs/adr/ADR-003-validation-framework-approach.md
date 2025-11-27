# ADR-003: Multi-Stage Validation Framework Approach

**Status**: Accepted

**Date**: 2024-11-23 (Retrospective)

**Decision Makers**: Core Development Team

**Note**: This document was created on 2025-11-27 as retrospective documentation. The date above is an estimate of when this architectural decision was originally made, based on code history and design patterns in the codebase.

## Context

LOFT is a self-modifying system that incorporates new rules autonomously. Without robust validation, the system could:
- Accept syntactically invalid ASP (crashes solver)
- Accept semantically inconsistent rules (no answer sets)
- Accept empirically incorrect rules (fails test cases)
- Degrade performance over time (accuracy regression)

We needed a validation framework that:
1. Catches errors before they enter the rule base
2. Balances thoroughness with performance
3. Supports multiple validation strategies
4. Enables automatic rollback on regression

## Decision

**We implement a multi-stage validation pipeline with four sequential stages:**

1. **Syntactic Validation** - ASP parser (Clingo)
2. **Semantic Validation** - Consistency checking (SAT solver)
3. **Empirical Validation** - Test case execution
4. **Consensus Validation** - Multi-agent agreement (Phase 2+)

Each stage acts as a gate: rules must pass to proceed to the next stage.

Additionally: **Performance Monitoring** continuously tracks incorporated rules and triggers automatic rollback on regression.

## Validation Stages

### Stage 1: Syntactic Validation

**Purpose**: Ensure rule is valid ASP syntax

**Method**: Parse with Clingo, check for syntax errors

**Example Pass**:
```asp
enforceable(C) :- contract(C), has_writing(C).
```

**Example Fail**:
```asp
enforceable(C) :- contract(C) has_writing(C).  % Missing comma - SYNTAX ERROR
```

**Performance**: <10ms (fast)

**Implementation**:
```python
def validate_syntax(asp_text: str) -> bool:
    try:
        ctl = clingo.Control()
        ctl.add("base", [], asp_text)
        ctl.ground([("base", [])])
        return True
    except clingo.SyntaxError:
        return False
```

### Stage 2: Semantic Validation

**Purpose**: Ensure rule doesn't create logical inconsistencies

**Method**: Combine rule with existing rule base, check for:
- Unsatisfiable programs (no answer sets)
- Contradictions (both P and -P derived)
- Dangerous cycles through negation

**Example Pass**:
```asp
% Existing: contract(C) :- offer(C), acceptance(C).
% New rule: enforceable(C) :- contract(C).  ✓ Consistent
```

**Example Fail**:
```asp
% Existing: enforceable(C) :- contract(C), not void(C).
% New rule: void(C) :- contract(C).  ✗ Makes all contracts void!
```

**Performance**: 50-200ms (depends on rule base size)

**Implementation**:
```python
def validate_semantic(new_rule: str, existing_rules: List[str]) -> bool:
    combined = "\n".join(existing_rules + [new_rule])
    ctl = clingo.Control()
    ctl.add("base", [], combined)
    ctl.ground([("base", [])])
    result = ctl.solve()
    return result.satisfiable  # Has at least one answer set
```

### Stage 3: Empirical Validation

**Purpose**: Ensure rule produces correct results on test cases

**Method**: Run rule against curated test cases (ground truth)

**Test Case Format**:
```python
@dataclass
class ValidationCase:
    case_id: str
    description: str
    asp_facts: str  # Facts for this test
    expected_results: Dict[str, bool]  # predicate → expected truth value
```

**Example Test**:
```python
test = ValidationCase(
    case_id="sof_001",
    description="Contract with writing satisfies statute of frauds",
    asp_facts="contract(c1). has_writing(c1). statute_of_frauds_applies(c1).",
    expected_results={"enforceable": True}
)
```

**Validation**:
```python
def validate_empirical(rule: str, test_cases: List[ValidationCase]) -> float:
    passed = 0
    for test in test_cases:
        result = run_asp(rule + "\n" + test.asp_facts)
        if matches_expected(result, test.expected_results):
            passed += 1
    return passed / len(test_cases)  # Accuracy
```

**Acceptance Threshold**:
- Strategic layer: ≥ 90% accuracy
- Tactical layer: ≥ 75% accuracy
- Operational layer: ≥ 50% accuracy

**Performance**: 100-500ms (depends on test suite size)

### Stage 4: Consensus Validation (Phase 2+)

**Purpose**: Ensure multiple reasoning strategies agree

**Method**: Multi-agent debate
- Proposer: Argues for rule
- Critic: Finds flaws
- Synthesizer: Resolves disagreement

**Example**:
```
Proposer: "liable(X) :- negligent(X), caused_harm(X)."
Critic: "But what about contributory negligence defenses?"
Synthesizer: "liable(X) :- negligent(X), caused_harm(X), not has_defense(X)."
```

**Acceptance**: ≥ 2/3 agents agree

**Performance**: 5-10 seconds (LLM queries)

**Note**: Implemented in Phase 2.2, not in Phase 0

### Stage 5: Performance Monitoring (Continuous)

**Purpose**: Detect accuracy regression after incorporation

**Method**:
- Track accuracy on held-out test set
- Compare before/after incorporation
- Trigger rollback if regression > threshold

**Regression Threshold**: Accuracy drop > 5%

**Example**:
```python
baseline_accuracy = 0.92
system.incorporate_rule(new_rule)
new_accuracy = 0.85  # Dropped 7%!
# Regression detected: 7% > 5% threshold
system.rollback()  # Revert to previous state
```

**Performance**: Runs asynchronously, doesn't block

## Design Rationale

### Why Multi-Stage?

**Sequential filtering reduces waste**:
- 90% of bad rules fail syntactic validation (<10ms)
- 8% fail semantic validation (50-200ms)
- 2% fail empirical validation (100-500ms)
- <1% fail consensus validation (5-10s)

**Example**: 1000 proposed rules
- After syntax: 100 remain (900 rejected, 9 seconds total)
- After semantic: 20 remain (80 rejected, 4 seconds total)
- After empirical: 10 remain (10 rejected, 1 second total)
- After consensus: 5 accepted (5 rejected, 25 seconds total)

Total: ~39 seconds for 1000 rules vs. 2.5 hours if all went to consensus!

### Why These Specific Stages?

**Syntactic**: Necessary (invalid ASP crashes solver)
**Semantic**: Critical (inconsistent rules break reasoning)
**Empirical**: High confidence (test cases are ground truth)
**Consensus**: Safety net (catches subtle issues)

### Why Not Just Test Cases?

Empirical validation alone is insufficient:

1. **Test coverage**: Cannot test all cases
2. **Semantic bugs**: Test cases won't catch all inconsistencies
3. **Syntax errors**: Would crash on test execution
4. **Edge cases**: Rare scenarios not in test suite

Multi-stage catches different error classes.

## Alternatives Considered

### 1. Single-Stage Validation (Test Cases Only)
**Pros**: Simpler, one metric (accuracy)
**Cons**: Misses syntax/semantic errors, slow (runs all tests)
**Decision**: Rejected - too coarse

### 2. Static Analysis Only (No Test Cases)
**Pros**: Fast, deterministic
**Cons**: Cannot verify correctness, only consistency
**Decision**: Rejected - insufficient confidence

### 3. LLM-Based Validation Only
**Pros**: Flexible, natural language reasoning
**Cons**: Non-deterministic, hallucinates, expensive
**Decision**: Rejected - cannot be sole validator (used in consensus stage)

### 4. Proof-Based Validation (Formal Verification)
**Pros**: Mathematical certainty
**Cons**: Expensive, requires formal specs, not always decidable
**Decision**: Deferred to future research

## Implementation Details

### Validation Pipeline

```python
class ValidationPipeline:
    def validate(self, rule: ASPRule) -> ValidationReport:
        # Stage 1: Syntax
        syntax_result = self.syntax_validator.validate(rule)
        if not syntax_result.passed:
            return ValidationReport(stage="syntax", passed=False)

        # Stage 2: Semantic
        semantic_result = self.semantic_validator.validate(rule)
        if not semantic_result.passed:
            return ValidationReport(stage="semantic", passed=False)

        # Stage 3: Empirical
        empirical_result = self.empirical_validator.validate(rule)
        if empirical_result.accuracy < self.threshold:
            return ValidationReport(stage="empirical", passed=False)

        # Stage 4: Consensus (Phase 2+)
        if self.use_consensus:
            consensus_result = self.consensus_validator.validate(rule)
            if not consensus_result.passed:
                return ValidationReport(stage="consensus", passed=False)

        return ValidationReport(stage="all", passed=True)
```

### Test Case Management

Test cases are organized by legal domain:
```
tests/test_cases/
├── contracts/
│   ├── statute_of_frauds.json
│   ├── consideration.json
│   └── capacity.json
├── torts/
│   ├── negligence.json
│   └── strict_liability.json
└── property/
    └── adverse_possession.json
```

### Confidence Scoring

Each validation stage contributes to overall confidence:

```python
confidence = (
    0.10 * syntax_score +    # 10%: Must pass
    0.20 * semantic_score +  # 20%: Consistency
    0.50 * empirical_score + # 50%: Ground truth
    0.20 * consensus_score   # 20%: Expert agreement
)
```

## Consequences

### Positive

1. **Safety**: Multi-stage filtering prevents bad rules
2. **Efficiency**: Early stages are fast, late stages are expensive but rare
3. **Flexibility**: Can adjust thresholds per stratification layer
4. **Debuggability**: Know exactly which stage failed
5. **Confidence**: Multiple independent checks increase trust

### Negative

1. **Complexity**: Four stages vs. single validation
2. **Latency**: Sequential stages add delay (but minimal due to filtering)
3. **Maintenance**: Test cases must be curated and kept current
4. **False negatives**: Overly strict validation may reject good rules

### Risks and Mitigations

**Risk**: Test cases become outdated (legal changes)
**Mitigation**: Continuous test case review, automated test generation (Phase 3)

**Risk**: Semantic validation too expensive for large rule bases
**Mitigation**: Stratification reduces consistency check scope, incremental solving

**Risk**: Consensus validation too slow
**Mitigation**: Only for strategic layer, async processing, parallel agents

## Validation Metrics

We track:
- **Pass rate** per stage (diagnostic: where do rules fail?)
- **False positive rate** (accepted bad rules - detected via monitoring)
- **False negative rate** (rejected good rules - harder to measure)
- **Validation latency** (p50, p95, p99)

**Target Metrics** (Phase 0):
- Syntax pass rate: ~90% (many LLM outputs have minor syntax errors)
- Semantic pass rate: ~80% (of those passing syntax)
- Empirical pass rate: ~60% (of those passing semantic)
- Overall acceptance rate: ~40-50%

## Related Decisions

- **ADR-002**: Stratified core (validation requirements vary by stratum)
- **ADR-005**: Version control (enables rollback on validation failure)
- **Phase 2.2**: Consensus validation implementation

## Future Evolution

**Phase 2**:
- Add consensus validation stage
- Expand test case coverage
- Implement dialectical refinement (critic ↔ proposer)

**Phase 3**:
- Automated test case generation
- Adversarial testing
- Self-improving validation (learn from mistakes)

**Phase 4**:
- Formal verification for constitutional layer
- Distributed consensus (multiple nodes)
- Real-time performance monitoring dashboard

## References

- [loft/validation/](../../loft/validation/) - Implementation
- [tests/unit/test_validation.py](../../tests/unit/test_validation.py) - Tests
- [ROADMAP.md](../../ROADMAP.md) - Phase planning

---

**Last Updated**: 2025-11-27
**Status**: Accepted
