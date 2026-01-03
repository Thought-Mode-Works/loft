#!/usr/bin/env python3
"""
Rule Refinement and Feedback Loop Demo.

Demonstrates the complete feedback loop for improving underperforming rules:
1. Track rule performance on questions
2. Analyze feedback to identify issues
3. Propose refinements for underperforming rules
4. Compare before/after metrics

Issue #278: Rule Refinement and Feedback Loop

Usage:
    python examples/feedback_loop_demo.py
"""

from unittest.mock import Mock

from loft.feedback import (
    FeedbackAnalyzer,
    RulePerformanceTracker,
    RuleRefiner,
)
from loft.qa.schemas import Answer, QuestionResult


def create_sample_evaluation_results():
    """Create sample evaluation results for demo."""
    print("\n" + "=" * 70)
    print("CREATING SAMPLE EVALUATION DATA")
    print("=" * 70)

    results = [
        # Rule contract_formation_001: Good performance
        QuestionResult(
            question="Does offer plus acceptance create a binding contract?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="yes",
                confidence=0.95,
                explanation="Valid contract formed",
                rules_used=["contract_formation_001"],
            ),
            domain="contracts",
        ),
        QuestionResult(
            question="Can a contract exist without mutual assent?",
            expected_answer="no",
            actual_answer=Answer(
                answer="no",
                confidence=0.92,
                explanation="Mutual assent required",
                rules_used=["contract_formation_001"],
            ),
            domain="contracts",
        ),
        # Rule consideration_check_002: Poor performance
        QuestionResult(
            question="Is past consideration valid for new promises?",
            expected_answer="no",
            actual_answer=Answer(
                answer="yes",  # WRONG
                confidence=0.7,
                explanation="Consideration present",
                rules_used=["consideration_check_002"],
            ),
            domain="contracts",
        ),
        QuestionResult(
            question="Must consideration have economic value?",
            expected_answer="no",
            actual_answer=Answer(
                answer="yes",  # WRONG
                confidence=0.65,
                explanation="Value required",
                rules_used=["consideration_check_002"],
            ),
            domain="contracts",
        ),
        QuestionResult(
            question="Is promise for promise valid consideration?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="no",  # WRONG
                confidence=0.6,
                explanation="No value",
                rules_used=["consideration_check_002"],
            ),
            domain="contracts",
        ),
        QuestionResult(
            question="Does nominal consideration invalidate a contract?",
            expected_answer="no",
            actual_answer=Answer(
                answer="yes",  # WRONG
                confidence=0.55,
                explanation="Insufficient value",
                rules_used=["consideration_check_002"],
            ),
            domain="contracts",
        ),
        QuestionResult(
            question="Can forbearance serve as valid consideration?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="yes",  # CORRECT
                confidence=0.8,
                explanation="Forbearance is consideration",
                rules_used=["consideration_check_002"],
            ),
            domain="contracts",
        ),
        # Rule tort_negligence_003: Domain-specific issue
        QuestionResult(
            question="Does breach of duty establish negligence?",
            expected_answer="no",
            actual_answer=Answer(
                answer="yes",  # WRONG
                confidence=0.7,
                explanation="Breach equals negligence",
                rules_used=["tort_negligence_003"],
            ),
            domain="torts",
        ),
        QuestionResult(
            question="Is duty of care sufficient for negligence liability?",
            expected_answer="no",
            actual_answer=Answer(
                answer="yes",  # WRONG
                confidence=0.65,
                explanation="Duty establishes liability",
                rules_used=["tort_negligence_003"],
            ),
            domain="torts",
        ),
        QuestionResult(
            question="Must negligence include actual harm?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="yes",  # CORRECT
                confidence=0.85,
                explanation="Actual harm required",
                rules_used=["tort_negligence_003"],
            ),
            domain="torts",
        ),
    ]

    print(f"\nCreated {len(results)} question results")
    print(
        "Rules involved: contract_formation_001, consideration_check_002, tort_negligence_003"
    )
    return results


def demo_performance_tracking():
    """Demo 1: Track rule performance."""
    print("\n" + "=" * 70)
    print("DEMO 1: TRACKING RULE PERFORMANCE")
    print("=" * 70)

    # Create tracker
    tracker = RulePerformanceTracker()

    # Get sample results
    results = create_sample_evaluation_results()

    # Track performance
    print("\nTracking performance across evaluation results...")
    for result in results:
        tracker.record_question_result(result, difficulty="medium")

    # Display metrics
    print("\n" + "-" * 70)
    print("RULE PERFORMANCE METRICS")
    print("-" * 70)

    for rule_id, metrics in tracker.get_all_metrics().items():
        print(f"\nRule: {rule_id}")
        print(f"  Total questions: {metrics.total_questions}")
        print(f"  Times used: {metrics.times_used}")
        print(f"  Correct when used: {metrics.correct_when_used}")
        print(f"  Accuracy: {metrics.accuracy_when_used:.1%}")
        print(f"  Average confidence: {metrics.avg_confidence:.1%}")

        if metrics.by_domain:
            print("  By domain:")
            for domain, stats in metrics.by_domain.items():
                domain_acc = (
                    stats["correct"] / stats["used"] if stats["used"] > 0 else 0
                )
                print(
                    f"    - {domain}: {domain_acc:.1%} ({stats['correct']}/{stats['used']})"
                )

    return tracker


def demo_feedback_analysis(tracker):
    """Demo 2: Analyze feedback."""
    print("\n" + "=" * 70)
    print("DEMO 2: ANALYZING FEEDBACK")
    print("=" * 70)

    # Create analyzer
    analyzer = FeedbackAnalyzer(
        accuracy_threshold=0.7,
        usage_threshold=0.2,
        min_usage=2,
    )

    # Analyze performance
    print("\nAnalyzing rule performance...")
    report = analyzer.analyze(tracker.get_all_metrics())

    # Display report
    print("\n" + report.summary())

    # Show detailed issues
    if report.issues_found:
        print("\n" + "-" * 70)
        print("DETAILED ISSUES")
        print("-" * 70)

        for i, issue in enumerate(report.issues_found, 1):
            print(f"\n{i}. [{issue.issue_type.upper()}]")
            print(f"   Severity: {issue.severity:.2f}")
            print(f"   Description: {issue.description}")
            if issue.example_failures:
                print("   Example failures:")
                for example in issue.example_failures[:2]:
                    print(f"     - {example}")
            print(f"   Suggested action: {issue.suggested_action}")

    # Identify refinement candidates
    print("\n" + "-" * 70)
    print("REFINEMENT CANDIDATES")
    print("-" * 70)

    candidates = analyzer.identify_refinement_candidates(tracker.get_all_metrics())
    print(f"\nFound {len(candidates)} rules that should be refined:")
    for rule_id in candidates:
        metrics = tracker.get_rule_performance(rule_id)
        print(
            f"  - {rule_id}: {metrics.accuracy_when_used:.1%} accuracy ({metrics.times_used} uses)"
        )

    return report, analyzer


def demo_rule_refinement(tracker, report):
    """Demo 3: Generate refinement proposals."""
    print("\n" + "=" * 70)
    print("DEMO 3: GENERATING REFINEMENT PROPOSALS")
    print("=" * 70)

    # Create mock LLM for demo
    mock_llm = Mock()
    mock_llm.generate = Mock(
        return_value="""
REFINEMENT_TYPE: strengthen

REFINED_RULE:
```asp
valid_consideration(C) :-
    bargained_for(C),
    legal_value(C),
    not past_consideration(C),
    not illusory_promise(C).
```

RATIONALE:
The original rule was too permissive, accepting past consideration and illusory promises.
This refinement explicitly excludes these invalid forms of consideration, which should
reduce false positives on questions about past consideration and promise-for-promise scenarios.

EXPECTED_IMPACT:
Should improve accuracy from ~20% to ~80% on consideration questions by properly
excluding past consideration and requiring bargained-for exchange.

TEST_CASES:
1. Past consideration should be invalid
2. Illusory promise should be invalid
3. Bargained-for exchange with legal value should be valid
4. Nominal consideration should still be valid (not excluded)
"""
    )

    refiner = RuleRefiner(mock_llm)

    # Focus on the underperforming consideration rule
    rule_id = "consideration_check_002"
    rule_text = "valid_consideration(C) :- consideration_present(C), has_value(C)."

    print(f"\nGenerating refinement for: {rule_id}")
    print(f"Original rule: {rule_text}")

    metrics = tracker.get_rule_performance(rule_id)
    issues = [
        issue for issue in report.issues_found if rule_id[:16] in issue.description
    ]

    print(f"\nCurrent performance: {metrics.accuracy_when_used:.1%}")
    print(f"Issues identified: {len(issues)}")

    # Generate proposal
    proposal = refiner.propose_refinement(rule_text, rule_id, metrics, issues)

    if proposal:
        print("\n" + "-" * 70)
        print("REFINEMENT PROPOSAL")
        print("-" * 70)
        print(f"\nRefinement Type: {proposal.refinement_type}")
        print("\nProposed Rule:")
        print(proposal.proposed_asp_rule)
        print("\nRationale:")
        print(proposal.rationale)
        print("\nExpected Impact:")
        print(proposal.expected_impact)
        print(f"\nConfidence: {proposal.confidence:.0%}")

        if proposal.test_cases:
            print("\nTest Cases:")
            for i, test_case in enumerate(proposal.test_cases, 1):
                print(f"  {i}. {test_case}")

        print(f"\nIssues Addressed: {len(proposal.issues_addressed)}")

    return proposal


def demo_performance_comparison(analyzer):
    """Demo 4: Compare before/after metrics."""
    print("\n" + "=" * 70)
    print("DEMO 4: COMPARING BEFORE/AFTER PERFORMANCE")
    print("=" * 70)

    # Create mock baseline and improved metrics
    from loft.feedback.schemas import (
        RuleFeedbackEntry,
        RuleOutcome,
        RulePerformanceMetrics,
    )

    # Baseline (poor performance)
    baseline = RulePerformanceMetrics(rule_id="consideration_check_002")
    for i in range(5):
        baseline.update_from_entry(
            RuleFeedbackEntry(
                rule_id="consideration_check_002",
                question=f"Q{i}",
                expected_answer="yes",
                actual_answer="yes" if i == 4 else "no",
                outcome=RuleOutcome.CORRECT if i == 4 else RuleOutcome.INCORRECT,
                rule_used=True,
                confidence=0.65,
            )
        )

    # After refinement (improved)
    improved = RulePerformanceMetrics(rule_id="consideration_check_002_v2")
    for i in range(5):
        improved.update_from_entry(
            RuleFeedbackEntry(
                rule_id="consideration_check_002_v2",
                question=f"Q{i}",
                expected_answer="yes",
                actual_answer="yes" if i < 4 else "no",
                outcome=RuleOutcome.CORRECT if i < 4 else RuleOutcome.INCORRECT,
                rule_used=True,
                confidence=0.85,
            )
        )

    print("\nBaseline Performance:")
    print(f"  Accuracy: {baseline.accuracy_when_used:.1%}")
    print(f"  Confidence: {baseline.avg_confidence:.1%}")
    print(f"  Correct: {baseline.correct_when_used}/{baseline.times_used}")

    print("\nAfter Refinement:")
    print(f"  Accuracy: {improved.accuracy_when_used:.1%}")
    print(f"  Confidence: {improved.avg_confidence:.1%}")
    print(f"  Correct: {improved.correct_when_used}/{improved.times_used}")

    # Compare
    deltas = analyzer.compare_metrics(baseline, improved)

    print("\n" + "-" * 70)
    print("PERFORMANCE DELTA")
    print("-" * 70)
    print(f"\nAccuracy change: {deltas['accuracy_delta']:+.1%}")
    print(f"Confidence change: {deltas['confidence_delta']:+.1%}")
    print(f"Improvement: {'✓ Yes' if deltas['improvement'] else '✗ No'}")


def main():
    """Run all demos."""
    print("\n" + "#" * 70)
    print("# RULE REFINEMENT AND FEEDBACK LOOP DEMONSTRATION")
    print("#" + "#" * 68)
    print("# Issue #278: Rule Refinement and Feedback Loop")
    print("#" * 70)

    # Demo 1: Track performance
    tracker = demo_performance_tracking()

    # Demo 2: Analyze feedback
    report, analyzer = demo_feedback_analysis(tracker)

    # Demo 3: Generate refinements
    _proposal = demo_rule_refinement(tracker, report)  # noqa: F841

    # Demo 4: Compare performance
    demo_performance_comparison(analyzer)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Tracker monitors rule performance across questions")
    print("  2. Analyzer identifies underperforming rules and specific issues")
    print("  3. Refiner proposes targeted improvements using LLM")
    print("  4. System can compare before/after metrics to validate refinements")
    print("\nThis feedback loop enables continuous improvement of the rule base!")
    print()


if __name__ == "__main__":
    main()
