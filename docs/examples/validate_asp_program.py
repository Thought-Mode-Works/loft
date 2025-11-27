"""
Example: Validating ASP Programs

This example demonstrates how to use LOFT's validation framework to validate
ASP rules through multiple stages.

Run this example:
    PYTHONPATH=/path/to/loft:$PYTHONPATH python3 docs/examples/validate_asp_program.py
"""

from loft.validation import (
    ASPSemanticValidator,
    TestCaseValidator,
    ValidationCase,
    create_test_suite,
    validate_asp_program,
)


def example_1_quick_syntax_validation():
    """Example 1: Quick syntax validation"""
    print("=" * 60)
    print("Example 1: Quick Syntax Validation")
    print("=" * 60)

    # Valid ASP rule
    valid_rule = "enforceable(C) :- contract(C), has_writing(C)."
    result = validate_asp_program(valid_rule)
    print(f"\nRule: {valid_rule}")
    print(f"Valid: {result['overall_valid']}")
    print(f"Syntax Valid: {result['syntax_valid']}")

    # Invalid ASP rule (missing comma)
    invalid_rule = "enforceable(C) :- contract(C) has_writing(C)."
    result = validate_asp_program(invalid_rule)
    print(f"\nRule: {invalid_rule}")
    print(f"Valid: {result['overall_valid']}")
    if result["syntax_error"]:
        print(f"Error: {result['syntax_error']}")


def example_2_semantic_validation():
    """Example 2: Semantic consistency checking"""
    print("\n" + "=" * 60)
    print("Example 2: Semantic Validation")
    print("=" * 60)

    validator = ASPSemanticValidator()

    # Good program: Consistent
    good_program = """
    person(alice).
    person(bob).
    happy(X) :- person(X), not sad(X).
    sad(alice).
    """
    is_consistent, msg = validator.check_consistency(good_program)
    print(f"\nProgram: {good_program.strip()}")
    print(f"Consistent: {is_consistent}")
    print(f"Message: {msg}")

    # Bad program: Unsatisfiable (constraint rejects all models)
    bad_program = """
    person(alice).
    :- person(X).  % Constraint: no persons allowed!
    """
    is_consistent, msg = validator.check_consistency(bad_program)
    print(f"\nProgram: {bad_program.strip()}")
    print(f"Consistent: {is_consistent}")
    print(f"Message: {msg}")


def example_3_empirical_validation():
    """Example 3: Test case validation"""
    print("\n" + "=" * 60)
    print("Example 3: Empirical Validation with Test Cases")
    print("=" * 60)

    # Define ASP program (Statute of Frauds rules)
    asp_program = """
% Statute of Frauds: Certain contracts must be in writing
statute_of_frauds_applies(C) :- contract(C), real_estate(C).
statute_of_frauds_applies(C) :- contract(C), over_500(C).

% Contract is enforceable if has writing OR SoF doesn't apply
enforceable(C) :- contract(C), has_writing(C).
enforceable(C) :- contract(C), not statute_of_frauds_applies(C).
"""

    # Create test cases
    test_data = [
        {
            "case_id": "sof_001",
            "description": "Real estate contract with writing",
            "asp_facts": "contract(c1). real_estate(c1). has_writing(c1).",
            "expected_results": {"enforceable": True},
        },
        {
            "case_id": "sof_002",
            "description": "Real estate contract without writing",
            "asp_facts": "contract(c2). real_estate(c2).",
            "expected_results": {"enforceable": False},
        },
        {
            "case_id": "sof_003",
            "description": "Small purchase (under $500) without writing",
            "asp_facts": "contract(c3).",
            "expected_results": {"enforceable": True},
        },
        {
            "case_id": "sof_004",
            "description": "Large purchase (over $500) with writing",
            "asp_facts": "contract(c4). over_500(c4). has_writing(c4).",
            "expected_results": {"enforceable": True},
        },
    ]

    test_suite = create_test_suite(test_data)

    # Run validation
    validator = TestCaseValidator()
    stats = validator.batch_validate(asp_program, test_suite)

    print("\nTest Results:")
    print(f"  Total: {stats['total']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Accuracy: {stats['accuracy']:.1%}")

    if stats["failed_cases"]:
        print("\nFailed Cases:")
        for case_id, result in stats["failed_cases"]:
            print(f"  - {case_id}: {result.explanation}")


def example_4_individual_test_case():
    """Example 4: Running a single test case with detailed output"""
    print("\n" + "=" * 60)
    print("Example 4: Individual Test Case with Details")
    print("=" * 60)

    asp_program = """
liable(X) :- negligent(X), caused_harm(X), not has_defense(X).
has_defense(X) :- assumption_of_risk(X).
"""

    test_case = ValidationCase(
        case_id="tort_001",
        description="Negligent party with no defense",
        asp_facts="""
person(alice).
negligent(alice).
caused_harm(alice).
""",
        expected_results={"liable": True},
        reasoning_chain=[
            "alice is negligent",
            "alice caused harm",
            "alice has no defense",
            "therefore, alice is liable",
        ],
    )

    validator = TestCaseValidator()
    result, detailed_explanation = validator.validate_with_explanation(asp_program, test_case)

    print(f"\n{detailed_explanation}")


def example_5_syntax_validation_only():
    """Example 5: Batch syntax validation"""
    print("\n" + "=" * 60)
    print("Example 5: Batch Syntax Validation")
    print("=" * 60)

    test_rules = [
        "valid_rule :- premise1, premise2.",
        "invalid_rule :- premise1 premise2.",  # Missing comma
        "a(X) :- b(X), c(X).",
        "d(X) :- e(X)) f(X).",  # Extra parenthesis
    ]

    for rule in test_rules:
        result = validate_asp_program(rule)
        status = "✓" if result["syntax_valid"] else "✗"
        print(f"\n{status} {rule}")
        if not result["syntax_valid"]:
            print(f"  Error: {result['syntax_error']}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("LOFT Validation Framework Examples")
    print("=" * 60)

    example_1_quick_syntax_validation()
    example_2_semantic_validation()
    example_3_empirical_validation()
    example_4_individual_test_case()
    example_5_syntax_validation_only()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
