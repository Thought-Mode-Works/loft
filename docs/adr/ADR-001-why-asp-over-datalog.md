# ADR-001: Why ASP Over Datalog

**Status**: Accepted

**Date**: 2024-11-20 (Retrospective)

**Decision Makers**: Core Development Team

**Note**: This document was created on 2025-11-27 as retrospective documentation. The date above is an estimate of when this architectural decision was originally made, based on code history and design patterns in the codebase.

## Context

LOFT requires a logic programming paradigm for representing legal rules. We needed to choose between:
1. **Answer Set Programming (ASP)** - Stable model semantics, nonmonotonic reasoning
2. **Datalog** - Stratified negation, bottom-up evaluation
3. **Prolog** - Backward chaining, procedural semantics
4. **Description Logics** - Ontological reasoning, decidable fragments

The system must handle:
- Defeasible reasoning (rules with exceptions)
- Nonmonotonic logic (new information can retract conclusions)
- Legal reasoning patterns (unless, except, defaults)
- Open-world assumption (absence of information ≠ false)
- Complex negation (negation as failure, classical negation)

## Decision

**We chose Answer Set Programming (ASP) as the core symbolic representation.**

Implementation: Clingo (ASP solver) via Python bindings

## Rationale

### ASP Advantages for Legal Reasoning

#### 1. Nonmonotonic Reasoning
ASP natively supports negation as failure (`not`), essential for legal defaults:

```asp
% Default rule: contracts are enforceable unless proven otherwise
enforceable(C) :- contract(C), not unenforceable(C).

% Exception: contracts without writing are unenforceable for statute of frauds
unenforceable(C) :- contract(C), statute_of_frauds_applies(C), not has_writing(C).
```

Adding `statute_of_frauds_applies(c1)` can retract `enforceable(c1)` - impossible in monotonic Datalog.

#### 2. Default Logic and Exceptions
Legal rules have defaults and exceptions. ASP's stable model semantics handles this elegantly:

```asp
% General rule
liable(X) :- negligent(X), caused_harm(X), not has_defense(X).

% Exceptions are expressed as additional facts/rules
has_defense(X) :- comparative_negligence(X).
has_defense(X) :- assumption_of_risk(X).
```

#### 3. Choice Rules and Disjunctive Logic
ASP supports **choice rules** for representing alternatives:

```asp
% A contract is either valid or void, but not both
{ valid(C) ; void(C) } = 1 :- contract(C).

% Preference: contracts are valid unless proven void
:- void(C), not proven_void(C).
```

#### 4. Constraints and Integrity Rules
ASP's **constraints** express prohibitions directly:

```asp
% Integrity constraint: cannot be both liable and immune
:- liable(X), immune(X).

% Constitutional constraint: fundamental rights cannot be violated
:- rule_violates_rights(R), active(R).
```

### Why Not Datalog?

**Datalog limitations:**
1. **Monotonic**: Adding facts cannot retract conclusions (no true defaults)
2. **Stratified negation only**: Cannot express cyclic dependencies through negation
3. **No choice/disjunction**: Cannot represent "one of these must be true"
4. **No constraints**: Cannot express prohibitions as first-class citizens

**Example Datalog cannot express:**
```asp
% ASP: "Guilty unless proven innocent" OR "innocent until proven guilty"
guilty(X) :- accused(X), not innocent(X).  % ASP can express both!
innocent(X) :- accused(X), not guilty(X).  % Needs stable model semantics
```

Datalog would require explicit stratification and cannot express symmetric defaults.

### Why Not Prolog?

**Prolog limitations:**
1. **Procedural semantics**: Order matters, hard to reason about declaratively
2. **No stable models**: Uses SLD resolution (depth-first search with backtracking)
3. **Closed-world assumption**: `not(X)` iff `X` fails (not true nonmonotonic reasoning)
4. **Cut operator**: Procedural control flow breaks declarative semantics

**Example:**
```prolog
% Prolog: Order matters!
innocent(X) :- accused(X), \+ guilty(X).
guilty(X) :- accused(X), \+ innocent(X).
% ^ This loops infinitely in Prolog! Works fine in ASP.
```

### Why Not Description Logics (OWL/DL)?

**Description Logic limitations:**
1. **Open-world assumption**: Too weak for legal reasoning (cannot express defaults)
2. **No negation as failure**: Only classical negation
3. **Complexity**: Reasoning in expressive DLs is expensive
4. **Not designed for rules**: Better for ontologies than rule-based reasoning

DL is excellent for **ontological modeling** (class hierarchies, taxonomies) but poor for **rule-based inference** with exceptions.

## Alternatives Considered

### 1. Hybrid ASP + Description Logics
- **Pros**: Best of both (ontologies + rules)
- **Cons**: Complexity, integration overhead, semantic impedance mismatch
- **Decision**: Deferred to Phase 1.5 (LinkedASP) if needed

### 2. Stratified Datalog with Extensions
- **Pros**: Simpler, faster for stratifiable programs
- **Cons**: Cannot express core legal patterns (symmetric defaults, disjunctions)
- **Decision**: Rejected - too restrictive

### 3. Pure Neural (LLM-based) Reasoning
- **Pros**: Flexible, handles natural language directly
- **Cons**: Not verifiable, no logical guarantees, hallucinates
- **Decision**: Used as **complement** (neural + symbolic hybrid), not replacement

## Consequences

### Positive

1. **Expressive Power**: Can represent complex legal reasoning patterns
2. **Declarative**: Rules are order-independent, easy to reason about
3. **Verifiable**: ASP has formal semantics, provable properties
4. **Mature Tooling**: Clingo is fast, well-maintained, feature-complete
5. **Active Community**: ASP research community, regular competitions, benchmarks

### Negative

1. **Learning Curve**: ASP is less known than Prolog/Datalog
2. **Performance**: Grounding phase can be expensive for large programs (mitigated by stratification)
3. **Debugging**: Answer set semantics can be unintuitive for newcomers
4. **Integration**: Requires Python bindings (clingo-py) rather than pure Python

### Mitigations

1. **Documentation**: Comprehensive docs and examples (this project!)
2. **Stratification**: Organize rules into layers to reduce grounding complexity
3. **Logging**: Detailed logging of solver operations for debugging
4. **Abstractions**: Python wrappers (loft.symbolic) hide Clingo complexity

## Related Decisions

- **ADR-002**: Stratified core design (leverages ASP expressiveness)
- **ADR-003**: Validation framework (relies on ASP consistency checks)
- **Phase 1.5**: LinkedASP (may add DL-style metadata later)

## References

- [Clingo Documentation](https://potassco.org/clingo/)
- [Answer Set Programming Book](https://www.springer.com/gp/book/9783642203084)
- [ASP vs Datalog Comparison](https://www.cs.uni-potsdam.de/~torsten/Potassco/Slides/datalog.pdf)
- [Legal Reasoning with ASP](https://arxiv.org/abs/1805.09925) (Research paper)

---

**Last Updated**: 2025-11-27
**Status**: Accepted
