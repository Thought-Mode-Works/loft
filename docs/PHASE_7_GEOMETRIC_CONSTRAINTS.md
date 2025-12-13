# Phase 7: Geometric Constraints & Invariance

## Overview

Phase 7 implements formal mathematical constraints ensuring legal and logical principles are preserved throughout the neuro-symbolic system. These constraints provide provable guarantees that the system maintains content-neutrality, party symmetry, temporal consistency, and other critical properties.

## Components

### 1. O(d)-Equivariance (`loft/constraints/equivariance.py`)

**Purpose**: Content-neutrality - outcomes depend on legal structure, not arbitrary content like specific names or monetary amounts.

**Key Classes**:
- `TransformationType`: Enum of transformation types (PARTY_PERMUTATION, AMOUNT_SCALING, CONTENT_SUBSTITUTION)
- `EquivarianceConstraint`: Abstract base class for equivariance properties
- `PartyPermutationEquivariance`: Verifies party swap doesn't change analysis
- `AmountScalingEquivariance`: Verifies proportional amount changes preserve relative outcomes
- `ContentSubstitutionEquivariance`: Verifies arbitrary content substitution doesn't affect outcomes
- `EquivarianceVerifier`: Verifies rules satisfy equivariance constraints
- `EquivarianceReport`: Report on verification results

**Example**:
```python
from loft.constraints.equivariance import (
    EquivarianceVerifier,
    PartyPermutationEquivariance,
)

# Create verifier with party permutation constraint
verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

# Define a rule to verify
def my_rule(case):
    return {"valid": case.get("has_offer") and case.get("has_acceptance")}

# Verify rule satisfies content-neutrality
test_cases = [
    {"plaintiff": "alice", "defendant": "bob", "has_offer": True, "has_acceptance": True},
    {"plaintiff": "charlie", "defendant": "diana", "has_offer": False, "has_acceptance": True},
]

report = verifier.verify_rule(my_rule, test_cases)
print(f"Rule is equivariant: {report.is_equivariant}")
```

### 2. Party Symmetry (`loft/constraints/symmetry.py`)

**Purpose**: Ensure rules treat parties symmetrically where legally appropriate.

**Key Classes**:
- `SymmetryType`: Enum of symmetry types (FULL, PARTIAL, ASYMMETRIC)
- `PartySymmetryConstraint`: Constraint for party symmetry
- `PartySymmetryTester`: Tests rules for party symmetry
- `SymmetryTestReport`: Report on symmetry test results
- `SymmetryViolation`: Details of a symmetry violation

**Example**:
```python
from loft.constraints.symmetry import PartySymmetryTester, SymmetryType

# Create tester with rules
tester = PartySymmetryTester(rules)

# Test symmetry
report = tester.test_symmetry(rule, test_cases, SymmetryType.FULL)
print(f"Rule is symmetric: {report.is_symmetric}")
```

### 3. Temporal Consistency (`loft/constraints/temporal.py`)

**Purpose**: Ensure similar cases with the same temporal relationships produce similar outcomes.

**Key Classes**:
- `TemporalTransformType`: Enum of temporal transformations (UNIFORM_SHIFT, DURATION_SCALE)
- `TemporalField`: Specification of a temporal field in cases
- `TemporalConsistencyTester`: Tests rules for temporal consistency
- `TemporalConsistencyReport`: Report on temporal consistency
- `TemporalViolation`: Details of a temporal violation

**Example**:
```python
from loft.constraints.temporal import TemporalConsistencyTester, TemporalField

# Define temporal fields
fields = [
    TemporalField(name="formation_date", field_type="date"),
    TemporalField(name="performance_date", field_type="date"),
]

tester = TemporalConsistencyTester(fields)

# Test shift invariance
report = tester.test_shift_invariance(rule, test_cases)
print(f"Temporally consistent: {report.is_consistent}")
```

### 4. Measure Theory (`loft/constraints/measure_theory.py`)

**Purpose**: Provide measure-theoretic framework for confidence and probability reasoning.

**Key Classes**:
- `MeasurableOutcome`: Enum of possible rule outcomes (TRUE, FALSE, UNDEFINED)
- `CaseDimension`: Specification of a dimension in case space
- `CaseSpace`: The measurable space of legal cases
- `MonomialPotential`: Represents legal element combinations as monomial potentials
- `CaseDistribution`: Probability distribution over case space
- `MeasurableSet`: A measurable subset of case space
- `MeasurableLegalRule`: A legal rule as a measurable function
- `RuleConfidenceCalculator`: Calculates confidence scores

**Example**:
```python
from loft.constraints.measure_theory import (
    CaseSpace, CaseDimension, CaseDistribution, MonomialPotential
)

# Define case space
space = CaseSpace(dimensions=[
    CaseDimension("has_offer", "boolean"),
    CaseDimension("has_acceptance", "boolean"),
    CaseDimension("amount", "numeric", bounds=(0, 10000)),
])

# Create distribution from samples
samples = [{"has_offer": True, "has_acceptance": True, "amount": 500}, ...]
dist = CaseDistribution.from_samples(space, samples)

# Create monomial potential
potential = MonomialPotential(
    elements=["offer", "acceptance"],
    weights=[1.0, 1.0],
)
```

### 5. Ring Structure (`loft/constraints/ring_structure.py`)

**Purpose**: Algebraic structure for principled rule composition.

**Key Classes**:
- `RingElement`: Abstract base for ring elements
- `BooleanRule`: Rules in the boolean ring (OR/AND operations)
- `ConfidenceRule`: Rules in the confidence semiring (max/product operations)
- `RuleComposition`: Utilities for composing rules (exception, conditional, sequence)
- `RingPropertyVerifier`: Verifies ring axioms hold
- `RingHomomorphism`: Structure-preserving maps between rings
- `ComposedRule`: Tracks composition history for explainability

**Ring Operations**:
- Addition (+): Disjunction (OR) for BooleanRule, max for ConfidenceRule
- Multiplication (*): Conjunction (AND) for BooleanRule, product for ConfidenceRule
- Zero: Always-false rule (additive identity)
- One: Always-true rule (multiplicative identity)

**Example**:
```python
from loft.constraints.ring_structure import (
    BooleanRule, RuleComposition, RingPropertyVerifier
)

# Create rules
has_offer = BooleanRule(lambda c: c.get("offer", False), "has_offer")
has_acceptance = BooleanRule(lambda c: c.get("acceptance", False), "has_acceptance")
is_minor = BooleanRule(lambda c: c.get("minor", False), "is_minor")

# Compose: contract valid = (offer AND acceptance) unless minor
base = has_offer * has_acceptance
valid_contract = RuleComposition.exception(base, is_minor)

# Verify ring properties
verifier = RingPropertyVerifier(test_cases)
report = verifier.full_verification([has_offer, has_acceptance], BooleanRule)
assert report.is_ring
```

### 6. Constitutional Verification (`loft/constraints/constitutional.py`)

**Purpose**: Formal verification that core safety and correctness properties are preserved.

**Constitutional Properties (8 total)**:

| Property | Type | Description |
|----------|------|-------------|
| NO_CONTRADICTION | LOGICAL | No logical contradictions (P ∧ ¬P) |
| RULE_MONOTONICITY | SAFETY | Rules only added, not silently removed |
| PARTY_NEUTRALITY | FAIRNESS | Neutral rules treat parties symmetrically |
| QUERY_TERMINATION | META | All queries terminate (depth/timeout) |
| EXPLAINABILITY | META | All outcomes have explanations |
| CONFIDENCE_BOUNDS | LOGICAL | Confidence scores in [0, 1] |
| STRATIFICATION_VALID | LOGICAL | Valid ASP stratification |
| ROLLBACK_AVAILABLE | SAFETY | Can rollback to safe state |

**Key Classes**:
- `PropertyType`: Enum of property types (LOGICAL, SAFETY, FAIRNESS, LEGAL, META)
- `VerificationResult`: Enum of verification results (VERIFIED, FALSIFIED, UNKNOWN)
- `ConstitutionalProperty`: A property that must always hold
- `ConstitutionalVerifier`: Verifies properties are preserved
- `ConstitutionalGuard`: Runtime guard that blocks unconstitutional operations
- `SystemState`: Snapshot of system state for verification

**Example**:
```python
from loft.constraints.constitutional import (
    create_verifier, create_guard, SystemState, Rule, Fact
)

# Create system state
state = SystemState(
    rules=[Rule(rule_id="r1", confidence=0.8)],
    facts=[Fact("valid")],
    metadata={
        "max_query_depth": 100,
        "previous_state_id": "init",
    },
    timestamp=0.0,
)

# Verify all properties
verifier = create_verifier()
report = verifier.verify_all(state)
print(f"All verified: {report.all_verified}")

# Use guard to prevent unconstitutional operations
guard = create_guard()
new_state, report = guard.guard(operation, state, "operation_name")
if not report.transition_safe:
    print(f"Operation blocked: {report.violations}")
```

## Mathematical Foundations

### Ring Algebra for Rule Composition

A **ring** (R, +, ·, 0, 1) satisfies:
1. (R, +, 0) is an abelian group (addition is commutative, associative, has identity)
2. (R, ·, 1) is a monoid (multiplication is associative, has identity)
3. Multiplication distributes over addition: a·(b+c) = a·b + a·c

For legal rules:
- **Addition**: Disjunction (OR) - either rule applies
- **Multiplication**: Conjunction (AND) - both rules must apply
- **Zero**: Always-false rule (additive identity)
- **One**: Always-true rule (multiplicative identity)

### Measure-Theoretic Confidence

Legal rules are modeled as measurable functions R: CaseSpace → Outcomes.

**Monomial Potential**: φ(x) = ∏ᵢ xᵢ^αᵢ

Properties:
- φ(x) = 1 if all elements fully satisfied
- φ(x) = 0 if any required element is zero
- 0 < φ(x) < 1 for partial satisfaction

### Equivariance Properties

A function f is **equivariant** under transformation T if:
f(T(x)) = T(f(x))

For content-neutrality: swapping party names in input produces equivalent swap in output.

## Integration Guide

### Adding Custom Constraints

1. Create a new constraint class inheriting from `EquivarianceConstraint`:

```python
from loft.constraints.equivariance import EquivarianceConstraint

class MyCustomConstraint(EquivarianceConstraint):
    def transform(self, case):
        # Apply transformation to case
        return transformed_case

    def verify(self, rule, original, transformed):
        # Verify equivariance holds
        return original_result == transformed_result
```

2. Add to verifier:

```python
verifier = EquivarianceVerifier([
    PartyPermutationEquivariance(),
    MyCustomConstraint(),
])
```

### Extending Constitutional Properties

1. Create a new property:

```python
from loft.constraints.constitutional import ConstitutionalProperty, PropertyType

new_property = ConstitutionalProperty(
    name="MY_PROPERTY",
    description="Description of what it checks",
    property_type=PropertyType.SAFETY,
    formal_spec="∀x: my_condition(x)",
    checker=lambda state: check_my_condition(state),
)
```

2. Add to verifier:

```python
properties = create_standard_properties() + [new_property]
verifier = ConstitutionalVerifier(properties)
```

## Performance Characteristics

| Component | Typical Operation | Time Complexity |
|-----------|-------------------|-----------------|
| EquivarianceVerifier | verify_rule | O(n × t) where n=transformations, t=test cases |
| RingPropertyVerifier | full_verification | O(r³ × t) where r=rules, t=test cases |
| ConstitutionalVerifier | verify_all | O(p × t) where p=properties |
| TemporalConsistencyTester | test_shift_invariance | O(s × t) where s=shifts, t=test cases |

## Validation Commands

```bash
# Run all Phase 7 unit tests
python -m pytest tests/unit/constraints/ -v

# Run integration tests
python -m pytest tests/integration/constraints/ -v

# Run with coverage
python -m pytest tests/unit/constraints/ --cov=loft.constraints --cov-report=html

# Run specific component tests
python -m pytest tests/unit/constraints/test_equivariance.py -v
python -m pytest tests/unit/constraints/test_ring_structure.py -v
python -m pytest tests/unit/constraints/test_constitutional.py -v
```

## References

- ROADMAP.md: Phase 7 section
- thoughts.md: Theoretical foundations
- CLAUDE.md: Development guidelines
- Issue #233: Phase 7 Tracking Issue
