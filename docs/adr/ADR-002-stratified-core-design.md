# ADR-002: Stratified Core Design

**Status**: Accepted

**Date**: 2024-11-22 (Retrospective)

**Decision Makers**: Core Development Team

**Note**: This document was created on 2025-11-27 as retrospective documentation. The date above is an estimate of when this architectural decision was originally made, based on code history and design patterns in the codebase.

## Context

LOFT needs to manage hundreds to thousands of ASP rules representing legal knowledge. These rules have varying levels of:
- **Confidence**: Some rules are well-established law, others are experimental
- **Modifiability**: Some rules should never change, others should adapt
- **Authority**: Some rules override others in conflicts

We needed a way to organize rules that:
1. Prevents unstable or experimental rules from corrupting foundational knowledge
2. Enables self-modification without risking system integrity
3. Provides clear modification authority and validation requirements
4. Supports gradual learning and improvement

## Decision

**We organize the ASP rule base into four stratification levels:**

1. **Constitutional (Stratum 0)** - Immutable axioms
2. **Strategic (Stratum 1)** - High-confidence, rarely modified
3. **Tactical (Stratum 2)** - Medium-confidence, regularly updated
4. **Operational (Stratum 3)** - Low-confidence, frequently modified

Each level has distinct confidence thresholds, modification policies, and validation requirements.

## Stratification Levels

### Constitutional Layer (Stratum 0)
**Purpose**: Immutable legal axioms and foundational definitions

**Characteristics**:
- Confidence threshold: 1.0 (certainty)
- Modification: ❌ Forbidden (requires manual intervention)
- Validation: N/A (cannot be changed)
- Examples:
  - Type definitions: `legal_entity(X) :- person(X) ; corporation(X).`
  - Fundamental axioms: `contract(X) :- offer(X), acceptance(X), consideration(X).`
  - Core logic: Transitivity, reflexivity, basic set operations

**Rationale**: Some legal concepts are definitional. Allowing modification would break the entire system.

### Strategic Layer (Stratum 1)
**Purpose**: High-confidence rules for well-established legal doctrines

**Characteristics**:
- Confidence threshold: ≥ 0.90
- Modification: ⚠️ Restricted (requires rigorous validation)
- Validation: All stages (Syntax + Semantic + Empirical + Consensus)
- Examples:
  - Major legal doctrines: Statute of Frauds, reasonable person standard
  - Precedential rules from Supreme Court cases
  - Widely accepted interpretations

**Rationale**: These rules are highly reliable but may need updates for new precedents or legal changes.

### Tactical Layer (Stratum 2)
**Purpose**: Medium-confidence rules for case-specific reasoning

**Characteristics**:
- Confidence threshold: ≥ 0.75
- Modification: ✅ Allowed (requires standard validation)
- Validation: Syntax + Semantic + Empirical
- Examples:
  - Circuit-level precedents
  - Context-specific rules
  - Domain-specific heuristics

**Rationale**: These rules handle most day-to-day legal reasoning and should be updatable as the system learns.

### Operational Layer (Stratum 3)
**Purpose**: Low-confidence, experimental rules

**Characteristics**:
- Confidence threshold: ≥ 0.50
- Modification: ✅ Encouraged (lightweight validation)
- Validation: Syntax only
- Examples:
  - Newly proposed rules
  - Experimental hypotheses
  - Temporary overrides for testing

**Rationale**: Rapid experimentation enables learning. Failures are acceptable and easily rolled back.

## Design Rationale

### Why Stratification?

#### 1. **Stability Guarantees**
Without stratification, a bad rule can corrupt the entire knowledge base:

```asp
% Bad operational rule shouldn't affect constitutional axioms
% BAD: Without stratification
legal_entity(X) :- experimental_heuristic(X).  % Overwrites definition!

% GOOD: With stratification
legal_entity(X) :- person(X) ; corporation(X).  % Constitutional (stratum 0)
possibly_entity(X) :- experimental_heuristic(X). % Operational (stratum 3)
```

#### 2. **Safe Self-Modification**
The system can modify operational/tactical rules without risking foundational knowledge:

```python
# Safe: System can experiment with operational rules
system.propose_rule("liable(X) :- new_doctrine(X).", layer=Operational)

# Unsafe: System cannot break constitutional definitions
system.propose_rule("contract(X).", layer=Constitutional)  # REJECTED!
```

#### 3. **Validation Efficiency**
Higher strata require more rigorous validation (expensive). Lower strata validate quickly:

| Layer | Validation Time | Rejection Rate |
|-------|----------------|----------------|
| Constitutional | N/A | N/A (immutable) |
| Strategic | 5-10 seconds | ~40% |
| Tactical | 1-2 seconds | ~25% |
| Operational | <100ms | ~10% |

This enables fast iteration at operational level while maintaining quality at strategic level.

#### 4. **Conflict Resolution**
When rules conflict, higher strata win:

```asp
% Constitutional (stratum 0): General rule
enforceable(C) :- contract(C), not void(C).

% Tactical (stratum 2): Specific exception
void(C) :- contract(C), fraud(C).

% Operational (stratum 3): Temporary override (testing)
-void(c123).  % Classical negation

% Resolution: Constitutional + Tactical override Operational
% Result: If fraud(c123), then void(c123) despite operational override
```

### Why Four Levels?

**Why not two (immutable + mutable)?**
- Too coarse - no distinction between high-confidence and experimental rules
- Cannot express "modifiable but carefully" (strategic layer)

**Why not three?**
- Initial design had three (foundational, standard, experimental)
- Found need for distinction between "high-confidence modifiable" (strategic) and "medium-confidence modifiable" (tactical)

**Why not five+?**
- Diminishing returns - four levels provide sufficient granularity
- More levels increase cognitive load and complexity
- Can add more levels later if needed (YAGNI)

## Alternatives Considered

### 1. Flat Rule Base (No Stratification)
**Pros**: Simpler, no stratification overhead
**Cons**: No stability guarantees, dangerous self-modification
**Decision**: Rejected - too risky for autonomous system

### 2. Priority-Based System (Weights)
**Pros**: Continuous confidence values, fine-grained control
**Cons**: No clear modification boundaries, hard to reason about
**Decision**: Rejected - discrete strata easier to understand and enforce

### 3. Two-Level (Immutable + Mutable)
**Pros**: Simple dichotomy
**Cons**: Too coarse, no distinction between confidence levels
**Decision**: Rejected - insufficient granularity

### 4. Dynamic Stratification (Rules Move Between Strata)
**Pros**: Rules can "graduate" from operational → tactical → strategic
**Cons**: Complex bookkeeping, unclear promotion criteria
**Decision**: Deferred to Phase 3 (self-modification with learning)

## Implementation Details

### Rule Representation
```python
@dataclass
class ASPRule:
    rule_id: str
    asp_text: str
    stratification_level: StratificationLevel  # Enum
    confidence: float
    metadata: RuleMetadata
```

### Modification Guards
```python
def propose_rule(rule: ASPRule, layer: StratificationLevel):
    if layer == StratificationLevel.CONSTITUTIONAL:
        raise ImmutableLayerError("Cannot modify constitutional layer")

    if layer == StratificationLevel.STRATEGIC:
        if rule.confidence < 0.90:
            raise ConfidenceError("Strategic rules require ≥0.90 confidence")
        validation_result = run_full_validation(rule)
    elif layer == StratificationLevel.TACTICAL:
        validation_result = run_standard_validation(rule)
    else:  # Operational
        validation_result = run_syntax_validation(rule)

    if validation_result.passed:
        incorporate_rule(rule, layer)
```

### Persistence
Rules are stored in separate files by stratum:
```
asp_rules/
├── constitutional.lp  # Immutable
├── strategic.lp       # High-confidence
├── tactical.lp        # Medium-confidence
└── operational.lp     # Low-confidence
```

This enables easy rollback: `rm operational.lp` removes all experimental rules.

## Consequences

### Positive

1. **Safety**: Constitutional layer prevents catastrophic failures
2. **Flexibility**: Operational layer enables rapid experimentation
3. **Clarity**: Clear modification policies for each layer
4. **Performance**: Lighter validation for lower strata (faster iteration)
5. **Auditing**: Easy to track which layer changed (for debugging)

### Negative

1. **Complexity**: Four layers vs. flat structure
2. **Ambiguity**: Sometimes unclear which stratum a rule belongs to
3. **Overhead**: Need to classify every rule by stratum
4. **Inflexibility**: Cannot easily move rules between strata (by design)

### Risks and Mitigations

**Risk**: Operational rules have low validation, could be harmful
**Mitigation**: Performance monitoring + automatic rollback on regression

**Risk**: Strategic rules are hard to update (rigorous validation)
**Mitigation**: Accept slower updates for critical rules (feature, not bug)

**Risk**: Ambiguous classification (is this tactical or operational?)
**Mitigation**: Document classification guidelines, use confidence thresholds

## Validation Criteria

A rule's stratum is determined by:
1. **Confidence score** (from empirical validation)
2. **Source provenance** (manually curated = higher stratum)
3. **Modification history** (stable rules graduate upward)

**Classification Rules**:
- Confidence ≥ 0.90 + manual curation → Strategic
- Confidence ≥ 0.75 → Tactical
- Confidence ≥ 0.50 → Operational
- Confidence < 0.50 → Rejected

## Related Decisions

- **ADR-001**: ASP choice (enables stratification with ASP's expressiveness)
- **ADR-003**: Validation framework (different validation per stratum)
- **ADR-005**: Version control (tracks changes per stratum)

## Future Evolution

**Phase 3**: Dynamic promotion/demotion between strata based on performance:
- Operational rule with consistently high accuracy → promotes to Tactical
- Tactical rule with regression → demotes to Operational or deprecates

**Phase 4**: Fine-grained permissions (user roles × strata):
- Admin: Can modify strategic
- Expert: Can modify tactical
- System: Can modify operational only

## References

- [Stratified Logic Programs](https://en.wikipedia.org/wiki/Stratified_logic_program)
- [ROADMAP.md](../../ROADMAP.md) - Phase planning
- [MAINTAINABILITY.md](../MAINTAINABILITY.md) - Future enhancements

---

**Last Updated**: 2025-11-27
**Status**: Accepted
