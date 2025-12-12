"""
Unit tests for human-in-the-loop review system.

Tests review queue, triggers, CLI interface, and integration.
"""

import tempfile
from datetime import datetime
from pathlib import Path


from loft.neural.rule_schemas import GeneratedRule
from loft.validation.review_integration import ReviewIntegration, create_review_workflow
from loft.validation.review_queue import ReviewQueue, ReviewStorage
from loft.validation.review_schemas import (
    ReviewConfig,
    ReviewDecision,
    ReviewItem,
    ReviewTriggerResult,
    RuleImpact,
)
from loft.validation.review_trigger import ReviewTrigger
from loft.validation.validation_schemas import (
    EmpiricalValidationResult,
    ValidationReport,
)


# ===== Review Schemas Tests =====


def test_review_item_creation():
    """Test creating ReviewItem."""
    rule = GeneratedRule(
        asp_rule="test(X) :- valid(X).",
        confidence=0.85,
        reasoning="Test rule",
        source_type="principle",
        source_text="Test source",
        predicates_used=["test", "valid"],
        new_predicates=[],
    )

    report = ValidationReport(
        rule_asp=rule.asp_rule,
        rule_id="test_1",
        target_layer="tactical",
    )
    report.final_decision = "accept"

    item = ReviewItem(
        id="review_001",
        rule=rule,
        validation_report=report,
        priority="high",
        reason="Test review item",
        created_at=datetime.now(),
    )

    assert item.id == "review_001"
    assert item.priority == "high"
    assert item.status == "pending"
    assert item.reviewer_id is None


def test_review_decision_creation():
    """Test creating ReviewDecision."""
    decision = ReviewDecision(
        item_id="review_001",
        decision="accept",
        reviewer_notes="Looks good",
        reviewed_at=datetime.now(),
        review_time_seconds=120.0,
    )

    assert decision.decision == "accept"
    assert decision.review_time_seconds == 120.0


def test_review_trigger_result():
    """Test ReviewTriggerResult."""
    result = ReviewTriggerResult(
        trigger_type="confidence_borderline",
        priority="medium",
        reason="Confidence near threshold",
        all_triggers=["confidence_borderline", "novel_predicate"],
    )

    assert result.trigger_type == "confidence_borderline"
    assert len(result.all_triggers) == 2
    summary = result.summary()
    assert "MEDIUM" in summary


def test_rule_impact():
    """Test RuleImpact assessment."""
    impact = RuleImpact(
        affects_rules=5,
        affects_test_cases=10,
        predicate_usage_frequency={"test": 3, "valid": 5},
        novelty_score=0.8,
    )

    # High novelty score (0.8) > threshold (0.5)
    assert impact.is_high_impact(threshold=0.5)
    # But affects_rules > 0, so still high impact even with higher threshold
    # Actually, the logic checks: affects_rules > 0 OR novelty_score > threshold
    # So this will be True because affects_rules=5 > 0
    assert impact.is_high_impact(threshold=0.9)


# ===== Review Queue Tests =====


def test_review_queue_initialization():
    """Test ReviewQueue initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)

        assert queue.storage is not None
        assert len(queue.priority_levels) == 4


def test_review_queue_add_item():
    """Test adding item to review queue."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)

        rule = GeneratedRule(
            asp_rule="test(X) :- valid(X).",
            confidence=0.75,
            reasoning="Test",
            source_type="principle",
            source_text="Test",
            predicates_used=[],
            new_predicates=[],
        )

        report = ValidationReport(
            rule_asp=rule.asp_rule,
            rule_id="test_1",
            target_layer="tactical",
        )

        item = queue.add(
            rule=rule,
            validation_report=report,
            priority="high",
            reason="Test review",
        )

        assert item.id.startswith("review_")
        assert item.priority == "high"
        assert item.status == "pending"


def test_review_queue_get_next():
    """Test getting next item from queue."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)

        # Add items with different priorities
        rule = GeneratedRule(
            asp_rule="test(X) :- valid(X).",
            confidence=0.75,
            reasoning="Test",
            source_type="principle",
            source_text="Test",
            predicates_used=[],
            new_predicates=[],
        )

        report = ValidationReport(rule_asp=rule.asp_rule, rule_id="test", target_layer="tactical")
        report.final_decision = "accept"

        queue.add(rule, report, priority="low", reason="Low priority")
        queue.add(rule, report, priority="critical", reason="Critical item")
        queue.add(rule, report, priority="medium", reason="Medium priority")

        # Should get critical first
        next_item = queue.get_next(reviewer_id="alice")
        assert next_item is not None
        assert next_item.priority == "critical"
        assert next_item.status == "in_review"
        assert next_item.reviewer_id == "alice"


def test_review_queue_submit_review():
    """Test submitting review decision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)

        rule = GeneratedRule(
            asp_rule="test(X) :- valid(X).",
            confidence=0.75,
            reasoning="Test",
            source_type="principle",
            source_text="Test",
            predicates_used=[],
            new_predicates=[],
        )

        report = ValidationReport(rule_asp=rule.asp_rule, rule_id="test", target_layer="tactical")
        report.final_decision = "accept"

        item = queue.add(rule, report, priority="medium", reason="Test")

        # Get and review
        next_item = queue.get_next("alice")
        decision = queue.submit_review(
            item_id=next_item.id,
            decision="accept",
            reviewer_notes="Looks good",
        )

        assert decision.decision == "accept"
        assert decision.reviewer_notes == "Looks good"

        # Check item is marked reviewed
        reviewed_item = queue.get_item(item.id)
        assert reviewed_item.status == "reviewed"
        assert reviewed_item.review_decision is not None


def test_review_queue_statistics():
    """Test queue statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)

        rule = GeneratedRule(
            asp_rule="test(X) :- valid(X).",
            confidence=0.75,
            reasoning="Test",
            source_type="principle",
            source_text="Test",
            predicates_used=[],
            new_predicates=[],
        )

        report = ValidationReport(rule_asp=rule.asp_rule, rule_id="test", target_layer="tactical")
        report.final_decision = "accept"

        # Add and review some items
        item1 = queue.add(rule, report, priority="high", reason="Test 1")
        _item2 = queue.add(rule, report, priority="low", reason="Test 2")

        queue.get_next("alice")
        queue.submit_review(item1.id, "accept", "OK")

        stats = queue.get_statistics()

        assert stats.total_items == 2
        assert stats.pending == 1
        assert stats.reviewed == 1
        assert stats.by_priority["high"] == 1
        assert stats.by_priority["low"] == 1


# ===== Review Trigger Tests =====


def test_review_trigger_initialization():
    """Test ReviewTrigger initialization."""
    trigger = ReviewTrigger()
    assert trigger.config is not None


def test_review_trigger_no_review_needed():
    """Test when no review is needed."""
    trigger = ReviewTrigger()

    rule = GeneratedRule(
        asp_rule="test(X) :- valid(X).",
        confidence=0.9,
        reasoning="High confidence",
        source_type="principle",
        source_text="Test",
        predicates_used=["test", "valid"],
        new_predicates=[],
    )

    report = ValidationReport(
        rule_asp=rule.asp_rule,
        rule_id="test",
        target_layer="tactical",
    )
    report.final_decision = "accept"

    result = trigger.should_review(rule, report)
    assert result is None


def test_review_trigger_confidence_borderline():
    """Test trigger for borderline confidence."""
    trigger = ReviewTrigger()

    rule = GeneratedRule(
        asp_rule="test(X) :- valid(X).",
        confidence=0.75,
        reasoning="Borderline",
        source_type="principle",
        source_text="Test",
        predicates_used=[],
        new_predicates=[],
    )

    report = ValidationReport(
        rule_asp=rule.asp_rule,
        rule_id="test",
        target_layer="tactical",
    )
    report.final_decision = "flag_for_review"
    report.metadata = {"flag_reason": "Confidence borderline"}

    result = trigger.should_review(rule, report)
    assert result is not None
    assert result.trigger_type == "confidence_borderline"
    assert result.priority == "medium"


def test_review_trigger_constitutional_layer():
    """Test trigger for constitutional layer."""
    trigger = ReviewTrigger()

    rule = GeneratedRule(
        asp_rule="fundamental_rights(X) :- person(X).",
        confidence=0.95,
        reasoning="Constitutional rule",
        source_type="principle",
        source_text="Test",
        predicates_used=["fundamental_rights", "person"],
        new_predicates=["fundamental_rights"],
    )

    report = ValidationReport(
        rule_asp=rule.asp_rule,
        rule_id="test",
        target_layer="strategic",
    )
    report.final_decision = "accept"

    result = trigger.should_review(rule, report)
    assert result is not None
    assert result.priority == "critical"
    assert "constitutional" in result.trigger_type.lower()


def test_review_trigger_novel_predicate():
    """Test trigger for novel predicates."""
    trigger = ReviewTrigger()

    rule = GeneratedRule(
        asp_rule="new_concept(X) :- existing(X).",
        confidence=0.85,
        reasoning="New concept",
        source_type="principle",
        source_text="Test",
        predicates_used=["new_concept", "existing"],
        new_predicates=["new_concept"],
    )

    report = ValidationReport(
        rule_asp=rule.asp_rule,
        rule_id="test",
        target_layer="tactical",
    )
    report.final_decision = "accept"

    existing_predicates = {"existing", "other"}

    result = trigger.should_review(rule, report, existing_predicates=existing_predicates)
    assert result is not None
    assert "novel_predicate" in result.all_triggers


def test_review_trigger_empirical_failures():
    """Test trigger for empirical failures."""
    trigger = ReviewTrigger()

    rule = GeneratedRule(
        asp_rule="test(X) :- valid(X).",
        confidence=0.85,
        reasoning="Test",
        source_type="principle",
        source_text="Test",
        predicates_used=[],
        new_predicates=[],
    )

    report = ValidationReport(
        rule_asp=rule.asp_rule,
        rule_id="test",
        target_layer="tactical",
    )

    # Add empirical validation with failures
    from loft.validation.validation_schemas import FailureCase, TestCase

    test_case = TestCase(
        case_id="tc1",
        description="Test case",
        facts="fact.",
        query="test",
        expected=True,
    )

    failure = FailureCase(
        test_case=test_case,
        expected=True,
        actual=False,
        failure_type="incorrect",
    )

    empirical_result = EmpiricalValidationResult(
        accuracy=0.5,
        baseline_accuracy=0.0,
        improvement=0.5,
        test_cases_passed=1,
        test_cases_failed=1,
        total_test_cases=2,
        failures=[failure],
        improvements=[],
        is_valid=False,
    )

    report.add_stage("empirical", empirical_result)

    result = trigger.should_review(rule, report)
    assert result is not None
    assert "empirical_failures" in result.all_triggers


# ===== Review Integration Tests =====


def test_review_integration_creation():
    """Test creating ReviewIntegration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)
        trigger = ReviewTrigger()

        integration = ReviewIntegration(queue, trigger)
        assert integration.review_queue is not None
        assert integration.review_trigger is not None


def test_review_integration_check_and_queue():
    """Test checking and queueing for review."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)
        trigger = ReviewTrigger()
        integration = ReviewIntegration(queue, trigger)

        rule = GeneratedRule(
            asp_rule="fundamental_rights(X) :- person(X).",
            confidence=0.9,
            reasoning="Test",
            source_type="principle",
            source_text="Test",
            predicates_used=["fundamental_rights"],
            new_predicates=[],
        )

        report = ValidationReport(
            rule_asp=rule.asp_rule,
            rule_id="test",
            target_layer="tactical",
        )
        report.final_decision = "accept"

        was_queued = integration.check_and_queue(rule, report)
        assert was_queued  # Should be queued due to constitutional predicate

        stats = queue.get_statistics()
        assert stats.pending == 1


def test_review_integration_apply_decision():
    """Test applying review decision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ReviewStorage(Path(tmpdir))
        queue = ReviewQueue(storage=storage)
        trigger = ReviewTrigger()
        integration = ReviewIntegration(queue, trigger)

        rule = GeneratedRule(
            asp_rule="test(X) :- valid(X).",
            confidence=0.75,
            reasoning="Test",
            source_type="principle",
            source_text="Test",
            predicates_used=[],
            new_predicates=[],
        )

        report = ValidationReport(
            rule_asp=rule.asp_rule,
            rule_id="test",
            target_layer="tactical",
        )
        report.final_decision = "accept"

        item = queue.add(rule, report, priority="medium", reason="Test")

        decision = ReviewDecision(
            item_id=item.id,
            decision="accept",
            reviewer_notes="OK",
            reviewed_at=datetime.now(),
            review_time_seconds=60.0,
        )

        accepted_rules = []

        def on_accept(rule_asp):
            accepted_rules.append(rule_asp)

        result = integration.apply_decision(decision, callback_accept=on_accept)
        assert result is True
        assert len(accepted_rules) == 1
        assert accepted_rules[0] == rule.asp_rule


def test_create_review_workflow():
    """Test creating review workflow helper."""
    workflow = create_review_workflow()
    assert isinstance(workflow, ReviewIntegration)
    assert workflow.review_queue is not None
    assert workflow.review_trigger is not None


# ===== Review Config Tests =====


def test_review_config_defaults():
    """Test ReviewConfig default values."""
    config = ReviewConfig()
    assert config.confidence_variance_threshold == 0.15
    assert config.consensus_strength_threshold == 0.6
    assert config.impact_threshold == 0.1
    assert "fundamental_rights" in config.constitutional_predicates


def test_review_config_custom():
    """Test custom ReviewConfig."""
    config = ReviewConfig(
        confidence_variance_threshold=0.2,
        consensus_strength_threshold=0.7,
        enable_novelty_detection=False,
    )
    assert config.confidence_variance_threshold == 0.2
    assert config.enable_novelty_detection is False
