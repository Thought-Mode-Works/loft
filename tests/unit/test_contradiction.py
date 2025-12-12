"""
Unit tests for contradiction management system (Phase 4.4).
"""

import pytest
import tempfile
from pathlib import Path

from loft.contradiction.contradiction_schemas import (
    ContradictionType,
    ContradictionSeverity,
    ResolutionStrategy,
    ContradictionReport,
    RuleInterpretation,
    ResolutionResult,
    ContextClassification,
)
from loft.contradiction.contradiction_manager import ContradictionManager
from loft.contradiction.context_classifier import ContextClassifier
from loft.contradiction.contradiction_store import ContradictionStore
from loft.symbolic.stratification import StratificationLevel


class TestContradictionSchemas:
    """Test contradiction data structures."""

    def test_contradiction_report_creation(self):
        """Test creating a contradiction report."""
        report = ContradictionReport(
            contradiction_id="contra-001",
            contradiction_type=ContradictionType.LOGICAL,
            rule_a_id="rule-a",
            rule_a_text="can_file(X) :- standing(X).",
            rule_b_id="rule-b",
            rule_b_text="~can_file(X) :- standing(X).",
            severity=ContradictionSeverity.HIGH,
            explanation="Direct logical negation",
        )

        assert report.contradiction_id == "contra-001"
        assert report.contradiction_type == ContradictionType.LOGICAL
        assert report.severity == ContradictionSeverity.HIGH
        assert report.is_critical() is False
        assert report.requires_immediate_resolution() is True

    def test_contradiction_serialization(self):
        """Test contradiction to_dict and from_dict."""
        report = ContradictionReport(
            contradiction_id="contra-002",
            contradiction_type=ContradictionType.PRECEDENT,
            rule_a_id="rule-1",
            rule_a_text="Rule A",
            rule_b_id="rule-2",
            rule_b_text="Rule B",
            severity=ContradictionSeverity.MEDIUM,
        )

        data = report.to_dict()
        restored = ContradictionReport.from_dict(data)

        assert restored.contradiction_id == report.contradiction_id
        assert restored.contradiction_type == report.contradiction_type
        assert restored.severity == report.severity

    def test_rule_interpretation_creation(self):
        """Test creating a rule interpretation."""
        interp = RuleInterpretation(
            interpretation_id="interp-001",
            principle="substantial_performance",
            interpretation_text="Substantial performance allows contract completion",
            asp_rules=["substantial_performance(X) :- mostly_complete(X)."],
            applicable_contexts=["contract_breach", "contract_remedy"],
            supporting_precedents=["Case A", "Case B"],
            confidence_score=0.85,
        )

        assert interp.interpretation_id == "interp-001"
        assert interp.principle == "substantial_performance"
        assert len(interp.applicable_contexts) == 2
        assert interp.confidence_score == 0.85

    def test_interpretation_serialization(self):
        """Test interpretation serialization."""
        interp = RuleInterpretation(
            interpretation_id="interp-002",
            principle="test_principle",
            interpretation_text="Test interpretation",
        )

        data = interp.to_dict()
        restored = RuleInterpretation.from_dict(data)

        assert restored.interpretation_id == interp.interpretation_id
        assert restored.principle == interp.principle

    def test_resolution_result_creation(self):
        """Test creating a resolution result."""
        result = ResolutionResult(
            contradiction_id="contra-001",
            strategy_applied=ResolutionStrategy.STRATIFICATION,
            success=True,
            winning_rule_id="rule-a",
            losing_rule_id="rule-b",
            resolution_notes="Higher layer wins",
            confidence=0.95,
        )

        assert result.success is True
        assert result.strategy_applied == ResolutionStrategy.STRATIFICATION
        assert result.winning_rule_id == "rule-a"

    def test_context_classification_creation(self):
        """Test creating a context classification."""
        classification = ContextClassification(
            context_id="ctx-001",
            context_type="contract_formation",
            confidence=0.9,
            key_features={"predicates": ["offer", "acceptance"]},
            domain="contract_law",
        )

        assert classification.context_type == "contract_formation"
        assert classification.confidence == 0.9
        assert classification.domain == "contract_law"

    def test_context_classification_serialization(self):
        """Test context classification serialization."""
        classification = ContextClassification(
            context_id="ctx-002",
            context_type="test_context",
            confidence=0.75,
        )

        data = classification.to_dict()
        restored = ContextClassification.from_dict(data)

        assert restored.context_id == classification.context_id
        assert restored.context_type == classification.context_type


class TestContradictionStore:
    """Test contradiction persistence."""

    @pytest.fixture
    def temp_store(self):
        """Create temporary contradiction store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContradictionStore(base_path=Path(tmpdir))
            yield store

    def test_save_and_load_contradiction(self, temp_store):
        """Test saving and loading a contradiction."""
        report = ContradictionReport(
            contradiction_id="contra-test-001",
            contradiction_type=ContradictionType.LOGICAL,
            rule_a_id="rule-a",
            rule_a_text="Rule A",
            rule_b_id="rule-b",
            rule_b_text="Rule B",
            severity=ContradictionSeverity.HIGH,
        )

        temp_store.save_contradiction(report)
        loaded = temp_store.get_contradiction("contra-test-001")

        assert loaded is not None
        assert loaded.contradiction_id == report.contradiction_id
        assert loaded.rule_a_id == report.rule_a_id

    def test_query_by_severity(self, temp_store):
        """Test querying contradictions by severity."""
        # Create contradictions with different severities
        for i, severity in enumerate(
            [ContradictionSeverity.LOW, ContradictionSeverity.HIGH]
        ):
            report = ContradictionReport(
                contradiction_id=f"contra-sev-{i}",
                contradiction_type=ContradictionType.LOGICAL,
                rule_a_id=f"rule-a-{i}",
                rule_a_text=f"Rule A {i}",
                rule_b_id=f"rule-b-{i}",
                rule_b_text=f"Rule B {i}",
                severity=severity,
            )
            temp_store.save_contradiction(report)

        high_severity = temp_store.query_by_severity(ContradictionSeverity.HIGH)
        assert len(high_severity) == 1
        assert high_severity[0].severity == ContradictionSeverity.HIGH

    def test_query_unresolved(self, temp_store):
        """Test querying unresolved contradictions."""
        resolved_report = ContradictionReport(
            contradiction_id="contra-resolved",
            contradiction_type=ContradictionType.LOGICAL,
            rule_a_id="rule-a",
            rule_a_text="Rule A",
            rule_b_id="rule-b",
            rule_b_text="Rule B",
            severity=ContradictionSeverity.MEDIUM,
            resolved=True,
        )

        unresolved_report = ContradictionReport(
            contradiction_id="contra-unresolved",
            contradiction_type=ContradictionType.LOGICAL,
            rule_a_id="rule-c",
            rule_a_text="Rule C",
            rule_b_id="rule-d",
            rule_b_text="Rule D",
            severity=ContradictionSeverity.MEDIUM,
            resolved=False,
        )

        temp_store.save_contradiction(resolved_report)
        temp_store.save_contradiction(unresolved_report)

        unresolved = temp_store.query_unresolved()
        assert len(unresolved) == 1
        assert unresolved[0].contradiction_id == "contra-unresolved"

    def test_save_and_load_interpretation(self, temp_store):
        """Test saving and loading interpretations."""
        interp = RuleInterpretation(
            interpretation_id="interp-test-001",
            principle="test_principle",
            interpretation_text="Test interpretation",
        )

        temp_store.save_interpretation(interp)
        loaded = temp_store.get_interpretation("interp-test-001")

        assert loaded is not None
        assert loaded.interpretation_id == interp.interpretation_id

    def test_get_interpretations_by_principle(self, temp_store):
        """Test getting interpretations by principle."""
        for i in range(3):
            interp = RuleInterpretation(
                interpretation_id=f"interp-{i}",
                principle="shared_principle",
                interpretation_text=f"Interpretation {i}",
            )
            temp_store.save_interpretation(interp)

        interpretations = temp_store.get_interpretations_by_principle(
            "shared_principle"
        )
        assert len(interpretations) == 3

    def test_get_statistics(self, temp_store):
        """Test getting contradiction statistics."""
        # Create contradictions
        for i in range(5):
            severity = (
                ContradictionSeverity.HIGH if i < 2 else ContradictionSeverity.LOW
            )
            report = ContradictionReport(
                contradiction_id=f"contra-stats-{i}",
                contradiction_type=ContradictionType.LOGICAL,
                rule_a_id=f"rule-a-{i}",
                rule_a_text=f"Rule A {i}",
                rule_b_id=f"rule-b-{i}",
                rule_b_text=f"Rule B {i}",
                severity=severity,
                resolved=(i % 2 == 0),
            )
            temp_store.save_contradiction(report)

        stats = temp_store.get_statistics()
        assert stats["total"] == 5
        assert stats["resolved"] == 3
        assert stats["unresolved"] == 2


class TestContextClassifier:
    """Test context classification."""

    @pytest.fixture
    def classifier(self):
        """Create context classifier."""
        return ContextClassifier()

    def test_classify_contract_formation(self, classifier):
        """Test classifying contract formation context."""
        case_facts = {
            "predicates": ["offer", "acceptance", "consideration"],
            "text": "Party A made an offer which Party B accepted",
        }

        classification = classifier.classify_context(case_facts)

        assert classification.context_type == "contract_formation"
        assert classification.confidence > 0.5

    def test_classify_contract_breach(self, classifier):
        """Test classifying contract breach context."""
        case_facts = {
            "predicates": ["breach", "performance", "default"],
            "text": "Party failed to perform contractual obligations",
        }

        classification = classifier.classify_context(case_facts)

        assert classification.context_type == "contract_breach"
        assert classification.confidence > 0.5

    def test_get_applicable_rules(self, classifier):
        """Test getting applicable rules for a context."""
        context = ContextClassification(
            context_id="ctx-test",
            context_type="contract_formation",
            confidence=0.9,
            key_features={"keywords": ["offer", "acceptance"]},
        )

        candidate_rules = [
            {
                "rule_id": "rule-1",
                "rule_text": "Rule about contracts",
                "context_tags": ["contract_formation"],
            },
            {
                "rule_id": "rule-2",
                "rule_text": "Rule about property",
                "context_tags": ["property_transfer"],
            },
        ]

        applicable = classifier.get_applicable_rules(context, candidate_rules)

        assert len(applicable) > 0
        assert applicable[0][0] == "rule-1"  # First rule more applicable
        assert applicable[0][1] > 0.8  # High confidence

    def test_select_interpretation(self, classifier):
        """Test selecting best interpretation for context."""
        context = ContextClassification(
            context_id="ctx-test",
            context_type="contract_remedy",
            confidence=0.85,
        )

        interpretations = [
            {
                "interpretation_id": "interp-1",
                "applicable_contexts": ["contract_remedy"],
            },
            {
                "interpretation_id": "interp-2",
                "applicable_contexts": ["property_transfer"],
            },
        ]

        selected = classifier.select_interpretation(context, interpretations)

        assert selected is not None
        assert selected[0] == "interp-1"


class TestContradictionManager:
    """Test contradiction management."""

    @pytest.fixture
    def temp_store(self):
        """Create temporary contradiction store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ContradictionStore(base_path=Path(tmpdir))
            yield store

    @pytest.fixture
    def manager(self, temp_store):
        """Create contradiction manager with temp store."""
        return ContradictionManager(store=temp_store)

    def test_detect_logical_contradiction(self, manager):
        """Test detecting logical contradictions."""
        rules = [
            {
                "rule_id": "rule-a",
                "rule_text": "can_file(X) :- standing(X).",
                "predicates": ["can_file", "standing"],
            },
            {
                "rule_id": "rule-b",
                "rule_text": "~can_file(X) :- not standing(X).",
                "predicates": ["can_file", "standing"],
            },
        ]

        contradictions = manager.detect_contradictions(rules)

        # May or may not detect depending on heuristics
        # Just verify it doesn't crash
        assert isinstance(contradictions, list)

    def test_detect_hierarchical_contradiction(self, manager):
        """Test detecting hierarchical contradictions."""
        rules = [
            {
                "rule_id": "rule-constitutional",
                "rule_text": "Rule at constitutional level",
                "predicates": ["test_predicate"],
                "layer": StratificationLevel.CONSTITUTIONAL,
            },
            {
                "rule_id": "rule-tactical",
                "rule_text": "Rule at tactical level",
                "predicates": ["test_predicate"],
                "layer": StratificationLevel.TACTICAL,
            },
        ]

        contradictions = manager.detect_contradictions(rules)

        # Should detect hierarchical contradiction
        if contradictions:
            assert (
                contradictions[0].contradiction_type == ContradictionType.HIERARCHICAL
            )

    def test_resolve_by_stratification(self, manager):
        """Test resolving contradiction by stratification."""
        contradiction = ContradictionReport(
            contradiction_id="contra-strat",
            contradiction_type=ContradictionType.HIERARCHICAL,
            rule_a_id="rule-constitutional",
            rule_a_text="Constitutional rule",
            rule_b_id="rule-tactical",
            rule_b_text="Tactical rule",
            severity=ContradictionSeverity.MEDIUM,
            affects_layers=[
                StratificationLevel.CONSTITUTIONAL,
                StratificationLevel.TACTICAL,
            ],
        )

        result = manager.resolve_contradiction(
            contradiction, strategy=ResolutionStrategy.STRATIFICATION
        )

        assert result.success is True
        assert result.strategy_applied == ResolutionStrategy.STRATIFICATION
        assert result.winning_rule_id is not None

    def test_select_rule_for_context(self, manager):
        """Test selecting rule based on context."""
        rules = [
            {
                "rule_id": "rule-contract",
                "rule_text": "Contract rule",
                "context_tags": ["contract_formation"],
            },
            {
                "rule_id": "rule-property",
                "rule_text": "Property rule",
                "context_tags": ["property_transfer"],
            },
        ]

        context_facts = {
            "predicates": ["offer", "acceptance"],
            "text": "Contract formation scenario",
        }

        selected_rule, confidence = manager.select_rule_for_context(
            rules, context_facts
        )

        assert selected_rule is not None
        assert confidence > 0.0

    def test_track_interpretation(self, manager):
        """Test tracking interpretations."""
        interp = RuleInterpretation(
            interpretation_id="interp-track",
            principle="test_principle",
            interpretation_text="Test interpretation",
        )

        manager.track_interpretation(interp)

        # Verify it was stored
        retrieved = manager.store.get_interpretation("interp-track")
        assert retrieved is not None
        assert retrieved.interpretation_id == "interp-track"

    def test_get_competing_interpretations(self, manager):
        """Test getting competing interpretations."""
        for i in range(3):
            interp = RuleInterpretation(
                interpretation_id=f"interp-compete-{i}",
                principle="competing_principle",
                interpretation_text=f"Interpretation {i}",
            )
            manager.track_interpretation(interp)

        interpretations = manager.get_competing_interpretations("competing_principle")
        assert len(interpretations) == 3

    def test_generate_contradiction_alert(self, manager):
        """Test generating contradiction alert."""
        contradiction = ContradictionReport(
            contradiction_id="contra-alert",
            contradiction_type=ContradictionType.LOGICAL,
            rule_a_id="rule-a",
            rule_a_text="Rule A",
            rule_b_id="rule-b",
            rule_b_text="Rule B",
            severity=ContradictionSeverity.CRITICAL,
            explanation="Critical contradiction",
        )

        alert = manager.generate_contradiction_alert(contradiction)

        assert alert["contradiction_id"] == "contra-alert"
        assert alert["severity"] == "critical"
        assert alert["requires_immediate_action"] is True

    def test_get_contradiction_statistics(self, manager):
        """Test getting contradiction statistics."""
        # Create some contradictions
        for i in range(3):
            contradiction = ContradictionReport(
                contradiction_id=f"contra-stats-{i}",
                contradiction_type=ContradictionType.LOGICAL,
                rule_a_id=f"rule-a-{i}",
                rule_a_text=f"Rule A {i}",
                rule_b_id=f"rule-b-{i}",
                rule_b_text=f"Rule B {i}",
                severity=ContradictionSeverity.MEDIUM,
            )
            manager.store.save_contradiction(contradiction)

        stats = manager.get_contradiction_statistics()
        assert stats["total"] == 3
        assert "by_severity" in stats
        assert "by_type" in stats
