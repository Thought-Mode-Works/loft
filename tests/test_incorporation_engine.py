"""
Unit tests for rule incorporation engine.

Tests stratified modification policies, safety checks, and rollback mechanisms.
"""

from loft.core.incorporation import (
    IncorporationResult,
    RuleIncorporationEngine,
    SimpleASPCore,
    SimpleVersionControl,
)
from loft.neural.rule_schemas import GeneratedRule
from loft.symbolic.stratification import (
    StratificationLevel,
    get_policy,
)
from loft.validation.validation_schemas import ValidationReport


class TestStratificationPolicies:
    """Test stratification level policies."""

    def test_constitutional_layer_blocks_autonomous(self):
        """Constitutional layer should block autonomous modifications."""
        policy = get_policy(StratificationLevel.CONSTITUTIONAL)

        assert policy.autonomous_allowed is False
        assert policy.requires_human_approval is True
        assert policy.max_modifications_per_session == 0

    def test_tactical_layer_allows_autonomous(self):
        """Tactical layer should allow autonomous modifications."""
        policy = get_policy(StratificationLevel.TACTICAL)

        assert policy.autonomous_allowed is True
        assert policy.confidence_threshold == 0.80
        assert policy.max_modifications_per_session == 10

    def test_policy_allows_modification(self):
        """Test policy.allows_modification() method."""
        tactical_policy = get_policy(StratificationLevel.TACTICAL)

        # High confidence, autonomous -> allowed
        assert tactical_policy.allows_modification(0.85, is_autonomous=True)

        # Low confidence -> not allowed
        assert not tactical_policy.allows_modification(0.75, is_autonomous=True)

        # Constitutional policy blocks autonomous
        const_policy = get_policy(StratificationLevel.CONSTITUTIONAL)
        assert not const_policy.allows_modification(0.99, is_autonomous=True)


class TestIncorporationEngine:
    """Test rule incorporation engine."""

    def create_test_rule(self, confidence: float = 0.85) -> GeneratedRule:
        """Create a test rule."""
        return GeneratedRule(
            asp_rule="test_rule(X) :- condition(X).",
            confidence=confidence,
            reasoning="Test rule",
            source_type="gap_fill",
            source_text="Test source",
            predicates_used=["test_rule", "condition"],
            new_predicates=["test_rule"],
        )

    def create_test_validation_report(self) -> ValidationReport:
        """Create a test validation report."""
        report = ValidationReport(
            rule_asp="test_rule(X) :- condition(X).",
            rule_id="test_1",
            target_layer="tactical",
        )
        report.final_decision = "accept"
        return report

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = RuleIncorporationEngine()

        assert engine.asp_core is not None
        assert engine.version_control is not None
        assert engine.test_suite is not None
        assert len(engine.modification_count) == 4  # 4 stratification levels

    def test_successful_incorporation_tactical(self):
        """Test successful rule incorporation into tactical layer."""
        engine = RuleIncorporationEngine()
        rule = self.create_test_rule(confidence=0.85)
        report = self.create_test_validation_report()

        result = engine.incorporate(
            rule=rule,
            target_layer=StratificationLevel.TACTICAL,
            validation_report=report,
            is_autonomous=True,
        )

        assert result.status == "success"
        assert result.modification_number == 1
        assert result.snapshot_id is not None
        assert len(engine.asp_core.rules) == 1

    def test_blocked_constitutional_autonomous(self):
        """Test that constitutional layer blocks autonomous modification."""
        engine = RuleIncorporationEngine()
        rule = self.create_test_rule(confidence=0.95)
        report = self.create_test_validation_report()

        result = engine.incorporate(
            rule=rule,
            target_layer=StratificationLevel.CONSTITUTIONAL,
            validation_report=report,
            is_autonomous=True,
        )

        assert result.status == "blocked"
        assert result.requires_human_review is True
        assert len(engine.asp_core.rules) == 0  # No rule added

    def test_rejected_low_confidence(self):
        """Test that low confidence rules are rejected."""
        engine = RuleIncorporationEngine()
        rule = self.create_test_rule(confidence=0.70)  # Below tactical threshold
        report = self.create_test_validation_report()

        result = engine.incorporate(
            rule=rule,
            target_layer=StratificationLevel.TACTICAL,
            validation_report=report,
            is_autonomous=True,
        )

        assert result.status == "rejected"
        assert "below threshold" in result.reason.lower()
        assert len(engine.asp_core.rules) == 0

    def test_modification_limit_enforcement(self):
        """Test that modification limits are enforced."""
        engine = RuleIncorporationEngine()
        report = self.create_test_validation_report()

        # Get tactical policy limit
        tactical_policy = get_policy(StratificationLevel.TACTICAL)
        limit = tactical_policy.max_modifications_per_session

        # Add rules up to limit
        for i in range(limit):
            rule = self.create_test_rule(confidence=0.85)
            result = engine.incorporate(
                rule=rule,
                target_layer=StratificationLevel.TACTICAL,
                validation_report=report,
                is_autonomous=True,
            )
            assert result.status == "success"

        # Next rule should be deferred
        rule = self.create_test_rule(confidence=0.85)
        result = engine.incorporate(
            rule=rule,
            target_layer=StratificationLevel.TACTICAL,
            validation_report=report,
            is_autonomous=True,
        )

        assert result.status == "deferred"
        assert "max modifications" in result.reason.lower()

    def test_session_reset(self):
        """Test that session reset works."""
        engine = RuleIncorporationEngine()
        rule = self.create_test_rule(confidence=0.85)
        report = self.create_test_validation_report()

        # Add a rule
        result = engine.incorporate(
            rule=rule,
            target_layer=StratificationLevel.TACTICAL,
            validation_report=report,
            is_autonomous=True,
        )
        assert result.status == "success"
        assert engine.modification_count["tactical"] == 1

        # Reset session
        engine.reset_session()

        # Counter should be reset
        assert engine.modification_count["tactical"] == 0

    def test_incorporation_history_tracking(self):
        """Test that incorporation history is tracked."""
        engine = RuleIncorporationEngine()
        rule = self.create_test_rule(confidence=0.85)
        report = self.create_test_validation_report()

        # Add a rule
        result = engine.incorporate(
            rule=rule,
            target_layer=StratificationLevel.TACTICAL,
            validation_report=report,
            is_autonomous=True,
        )

        assert result.status == "success"
        assert len(engine.incorporation_history) == 1

        history = engine.get_history()
        assert len(history) == 1
        assert history[0]["rule"] == rule.asp_rule
        assert history[0]["layer"] == "tactical"

    def test_get_statistics(self):
        """Test statistics generation."""
        engine = RuleIncorporationEngine()
        rule = self.create_test_rule(confidence=0.85)
        report = self.create_test_validation_report()

        # Add a rule
        engine.incorporate(
            rule=rule,
            target_layer=StratificationLevel.TACTICAL,
            validation_report=report,
            is_autonomous=True,
        )

        stats = engine.get_statistics()

        assert stats["total_modifications"] == 1
        assert stats["by_layer"]["tactical"] == 1
        assert "current_accuracy" in stats


class TestVersionControl:
    """Test version control functionality."""

    def test_snapshot_creation(self):
        """Test creating snapshots."""
        vc = SimpleVersionControl()
        state = {"rules": [], "count": 0}

        snapshot_id = vc.create_snapshot(state, message="Initial state")

        assert snapshot_id is not None
        assert snapshot_id in vc.snapshots
        assert vc.current_snapshot_id == snapshot_id

    def test_snapshot_retrieval(self):
        """Test retrieving snapshots."""
        vc = SimpleVersionControl()
        state = {"rules": ["rule1"], "count": 1}

        snapshot_id = vc.create_snapshot(state, message="With rule")
        retrieved = vc.get_snapshot(snapshot_id)

        assert retrieved == state

    def test_snapshot_list(self):
        """Test listing snapshots."""
        vc = SimpleVersionControl()

        vc.create_snapshot({"count": 0}, message="Snapshot 1")
        vc.create_snapshot({"count": 1}, message="Snapshot 2")

        snapshots = vc.list_snapshots()

        assert len(snapshots) == 2
        assert all("message" in s for s in snapshots)


class TestASPCore:
    """Test simplified ASP core."""

    def test_add_rule(self):
        """Test adding rules to core."""
        core = SimpleASPCore()

        core.add_rule(
            rule_text="test(X) :- condition(X).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata={"source": "test"},
        )

        assert len(core.rules) == 1
        assert core.rule_count == 1

    def test_get_state_and_restore(self):
        """Test state snapshot and restore."""
        core = SimpleASPCore()

        # Add a rule
        core.add_rule(
            rule_text="test(X) :- condition(X).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata={},
        )

        # Get state
        state = core.get_state()
        assert len(state["rules"]) == 1

        # Add another rule
        core.add_rule(
            rule_text="test2(X) :- condition2(X).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata={},
        )
        assert len(core.rules) == 2

        # Restore to previous state
        core.restore_state(state)
        assert len(core.rules) == 1

    def test_get_rules_by_layer(self):
        """Test filtering rules by layer."""
        core = SimpleASPCore()

        core.add_rule(
            "tactical_rule(X) :- cond(X).",
            StratificationLevel.TACTICAL,
            0.85,
            {},
        )
        core.add_rule(
            "operational_rule(X) :- cond(X).",
            StratificationLevel.OPERATIONAL,
            0.75,
            {},
        )

        tactical_rules = core.get_rules_by_layer(StratificationLevel.TACTICAL)
        operational_rules = core.get_rules_by_layer(StratificationLevel.OPERATIONAL)

        assert len(tactical_rules) == 1
        assert len(operational_rules) == 1


class TestIncorporationResult:
    """Test incorporation result class."""

    def test_is_success(self):
        """Test is_success() method."""
        success = IncorporationResult(status="success")
        rejected = IncorporationResult(status="rejected")

        assert success.is_success()
        assert not rejected.is_success()

    def test_summary_success(self):
        """Test summary for successful incorporation."""
        result = IncorporationResult(
            status="success",
            modification_number=1,
            accuracy_before=0.85,
            accuracy_after=0.90,
        )

        summary = result.summary()

        assert "SUCCESS" in summary
        assert "modification #1" in summary
        assert "85.0%" in summary or "85%" in summary
        assert "90.0%" in summary or "90%" in summary

    def test_summary_rejected(self):
        """Test summary for rejected incorporation."""
        result = IncorporationResult(status="rejected", reason="Confidence too low")

        summary = result.summary()

        assert "REJECTED" in summary
        assert "Confidence too low" in summary
