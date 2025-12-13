# Party Symmetry Invariance Testing

Party Symmetry Invariance Testing ensures that legal rules produce equivalent outcomes when party roles are swapped or permuted, verifying fairness and content-neutrality in legal reasoning. This document outlines how to define, test, and integrate party symmetry constraints into the LOFT system.

## 1. Defining Symmetry Constraints

Symmetry constraints are defined using the `PartySymmetryConstraint` dataclass in `loft/constraints/symmetry.py`. This class specifies the expected symmetry type for a given rule.

```python
from loft.constraints.symmetry import PartySymmetryConstraint, SymmetryType

# Example: A rule expected to have Full Symmetry
full_symmetry_constraint = PartySymmetryConstraint(
    rule_name="mutual_assent_rule",
    symmetry_type=SymmetryType.FULL,
    parties=["X", "Y"], # Variables in the rule representing parties
    equivalent_roles=[] # Not applicable for full symmetry
)

# Example: A rule expected to have Role-Based Symmetry (e.g., buyer/seller)
role_symmetry_constraint = PartySymmetryConstraint(
    rule_name="sale_valid_rule",
    symmetry_type=SymmetryType.ROLE_BASED,
    parties=["Buyer", "Seller"],
    equivalent_roles=[{"Buyer", "Seller"}] # Roles that can be swapped
)

# Example: Justified asymmetry
justified_asymmetry_constraint = PartySymmetryConstraint(
    rule_name="minor_protected_rule",
    symmetry_type=SymmetryType.ASYMMETRIC, # Explicitly asymmetric
    parties=["Minor", "Adult"],
    exceptions=["age_based_protection"] # Justification for asymmetry
)
```

### `SymmetryType` Enum

- `FULL`: The rule's outcome is unchanged under any permutation of its parties.
- `ROLE_BASED`: The rule's outcome is consistent when parties swap predefined roles (e.g., buyer and seller).
- `PARTIAL`: Only some permutations preserve the outcome. (Advanced, requires custom logic).
- `ASYMMETRIC`: The rule is inherently asymmetric, often for legally justified reasons.

## 2. Testing Party Symmetry

The `PartySymmetryTester` class is used to verify rules against their defined symmetry types.

### `PartySymmetryTester` Usage

```python
from loft.constraints.symmetry import PartySymmetryTester, SymmetryType
from loft.symbolic.asp_rule import ASPRule
from unittest.mock import MagicMock # For mocking rule.evaluate in tests

# Assume 'my_rule' is an ASPRule instance
my_rule = ASPRule(
    rule_id="example_rule_id",
    asp_text="example_rule(X, Y) :- fact(X), fact(Y).",
    stratification_level=StratificationLevel.TACTICAL,
    confidence=0.8,
    metadata=RuleMetadata(provenance="test", timestamp="2023-01-01T00:00:00Z")
)
my_rule.evaluate = MagicMock(side_effect=lambda case: True) # Mock evaluation

# Test cases for the rule
test_cases = [
    {"X": "alice", "Y": "bob", "fact_alice": True, "fact_bob": True},
    {"X": "charlie", "Y": "david", "fact_charlie": True, "fact_david": True},
]

# Initialize the tester with rules (or a single rule in this case)
tester = PartySymmetryTester([my_rule])

# Test the rule for full symmetry
report = tester.test_symmetry(my_rule, test_cases, SymmetryType.FULL)

if report.is_symmetric:
    print(f"Rule '{my_rule.name}' exhibits {report.symmetry_type.name} symmetry.")
else:
    print(f"Rule '{my_rule.name}' violates {report.symmetry_type.name} symmetry.")
    for violation in report.violations:
        print(violation.explain())
```

### Key Methods of `PartySymmetryTester`

- `detect_parties(rule: ASPRule) -> List[str]`: Identifies party variables within an `ASPRule`.
- `generate_permutations(parties: List[str], symmetry_type: SymmetryType) -> List[Dict[str, str]]`: Generates permutations of party identifiers based on the specified symmetry type.
- `apply_permutation(case: Dict[str, Any], permutation: Dict[str, str]) -> Dict[str, Any]`: Applies a party permutation to a given case, updating party identifiers in facts.
- `test_symmetry(rule: ASPRule, test_cases: List[Dict[str, Any]], expected_symmetry: SymmetryType) -> SymmetryTestReport`: Runs the symmetry tests and returns a `SymmetryTestReport`.
- `_outcomes_equivalent(out1: Any, out2: Any) -> bool`: Compares two evaluation outcomes to determine if they are equivalent under permutation.

## 3. Integrating with the Validation Pipeline

To automatically check for party symmetry, integrate the `PartySymmetryTester` into the `ValidationPipeline`.

### `ValidationPipeline` Usage

When initializing the `ValidationPipeline`, pass an instance of `PartySymmetryTester`.
When calling `validate_rule`, provide the `expected_symmetry_type`.

```python
from loft.validation.validation_pipeline import ValidationPipeline
from loft.constraints.symmetry import PartySymmetryTester, SymmetryType
from loft.symbolic.asp_rule import ASPRule, RuleMetadata, StratificationLevel
from loft.validation.validation_schemas import TestCase # Assuming TestCase is available

# Prepare your rule (e.g., an ASPRule instance)
my_rule_asp_text = "enforceable(C) :- contract(C), party(X, C), party(Y, C), agreed(X,Y)."
my_rule_instance = ASPRule(
    rule_id="enforceable_contract",
    asp_text=my_rule_asp_text,
    stratification_level=StratificationLevel.TACTICAL,
    confidence=0.9,
    metadata=RuleMetadata(provenance="test")
)
my_rule_instance.evaluate = MagicMock(side_effect=lambda case: True) # Mock its evaluation behavior

# Test cases for the rule, structured for symmetry testing
example_test_cases = [
    TestCase(
        description="Simple agreement between alice and bob",
        input_facts=[
            "contract(c1).",
            "party(c1, alice).",
            "party(c1, bob).",
            "agreed(alice, bob)."
        ],
        expected_outputs=["enforceable(c1)."]
    )
    # ... more test cases
]

# Initialize PartySymmetryTester
symmetry_tester = PartySymmetryTester([my_rule_instance]) # Pass relevant rules

# Initialize the validation pipeline with the symmetry tester
pipeline = ValidationPipeline(
    symmetry_tester=symmetry_tester,
    # ... other validators
)

# Validate a rule, specifying the expected symmetry type
report = pipeline.validate_rule(
    rule_asp=my_rule_asp_text,
    rule_id="enforceable_contract",
    test_cases=[{"party_X": "alice", "party_Y": "bob", "agreed_alice_bob": True}], # Simplified test case for evaluate mock
    expected_symmetry_type=SymmetryType.FULL # Expect full symmetry
)

if "symmetry" in report.stage_results:
    symmetry_report = report.stage_results["symmetry"]
    print(f"Symmetry Test Result: {'PASS' if symmetry_report.is_symmetric else 'FAIL'}")
    if not symmetry_report.is_symmetric:
        for violation in symmetry_report.violations:
            print(f"  - {violation.explain()}")
```

## 4. Contributing New Symmetry Tests

When adding new rules or modifying existing ones, consider their symmetry properties:

1.  **Identify Party Variables**: Determine which variables in the rule represent parties.
2.  **Choose Symmetry Type**: Decide whether the rule should exhibit `FULL`, `ROLE_BASED`, `PARTIAL`, or `ASYMMETRIC` symmetry.
3.  **Write Test Cases**: Create `test_cases` that effectively exercise permutations of party identifiers. Ensure these test cases are structured such that `rule.evaluate` can interpret them correctly.
4.  **Add Unit Tests**: Place unit tests for `PartySymmetryTester` components (e.g., `generate_permutations`, `apply_permutation`) in `tests/unit/constraints/test_symmetry.py`.
5.  **Add Integration Tests**: Include integration tests for specific rules or rule sets (e.g., Statute of Frauds) in `tests/integration/constraints/test_symmetry.py`.
