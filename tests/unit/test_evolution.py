"""
Unit tests for Rule Evolution Tracking System (Phase 4.3).
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from loft.evolution.evolution_schemas import (
    RuleVersion,
    RuleLineage,
    EvolutionContext,
    EvolutionMethod,
    PerformanceSnapshot,
)
from loft.evolution.evolution_store import RuleEvolutionStore
from loft.evolution.evolution_tracker import RuleEvolutionTracker


@pytest.fixture
def temp_store_path():
    """Create temporary directory for test store."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def evolution_store(temp_store_path):
    """Create evolution store for testing."""
    return RuleEvolutionStore(base_path=temp_store_path)


@pytest.fixture
def evolution_tracker(evolution_store):
    """Create evolution tracker for testing."""
    return RuleEvolutionTracker(store=evolution_store)


class TestEvolutionSchemas:
    """Test evolution data structures."""

    def test_rule_version_creation(self):
        """Test creating a rule version."""
        ctx = EvolutionContext(
            evolution_method=EvolutionMethod.DIALECTICAL,
            reasoning="Test rule from debate",
        )

        perf = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.85,
            confidence=0.90,
            test_cases_passed=17,
            test_cases_total=20,
            success_rate=0.85,
        )

        version = RuleVersion(
            rule_id="test-rule-1",
            rule_family_id="test-family",
            version="1.0.0",
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            predicates_used=["contract", "signed"],
            evolution_context=ctx,
            performance=perf,
        )

        assert version.rule_id == "test-rule-1"
        assert version.version == "1.0.0"
        assert version.evolution_context.evolution_method == EvolutionMethod.DIALECTICAL
        assert version.performance.confidence == 0.90

    def test_rule_version_serialization(self):
        """Test rule version to/from dict."""
        version = RuleVersion(
            rule_id="test-rule-2",
            rule_family_id="test-family",
            version="1.0.1",
            asp_rule="test_rule(X) :- condition(X).",
            predicates_used=["condition"],
            evolution_context=EvolutionContext(
                evolution_method=EvolutionMethod.MANUAL,
                reasoning="Manual test rule",
            ),
        )

        # Serialize
        data = version.to_dict()
        assert data["rule_id"] == "test-rule-2"
        assert data["version"] == "1.0.1"

        # Deserialize
        restored = RuleVersion.from_dict(data)
        assert restored.rule_id == version.rule_id
        assert restored.asp_rule == version.asp_rule
        assert restored.evolution_context.evolution_method == EvolutionMethod.MANUAL

    def test_rule_lineage_creation(self):
        """Test creating rule lineage."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="rule1.",
        )

        v2 = RuleVersion(
            rule_id="v2",
            rule_family_id="fam1",
            version="1.0.1",
            asp_rule="rule2.",
            parent_version="v1",
        )

        lineage = RuleLineage(
            rule_family_id="fam1",
            root_version=v1,
            all_versions=[v1, v2],
            current_version=v2,
            total_iterations=2,
            overall_improvement=0.05,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        assert lineage.total_iterations == 2
        assert len(lineage.all_versions) == 2

    def test_lineage_evolution_path(self):
        """Test getting evolution path."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="rule1.",
        )

        v2 = RuleVersion(
            rule_id="v2",
            rule_family_id="fam1",
            version="1.0.1",
            asp_rule="rule2.",
            parent_version="v1",
        )

        v3 = RuleVersion(
            rule_id="v3",
            rule_family_id="fam1",
            version="1.0.2",
            asp_rule="rule3.",
            parent_version="v2",
        )

        lineage = RuleLineage(
            rule_family_id="fam1",
            root_version=v1,
            all_versions=[v1, v2, v3],
            current_version=v3,
            total_iterations=3,
            overall_improvement=0.10,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        path = lineage.get_evolution_path()
        assert len(path) == 3
        assert path[0].rule_id == "v1"
        assert path[1].rule_id == "v2"
        assert path[2].rule_id == "v3"

    def test_lineage_branching(self):
        """Test branching lineage."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="rule1.",
        )

        # Two branches from v1
        v2a = RuleVersion(
            rule_id="v2a",
            rule_family_id="fam1",
            version="1.1.0",
            asp_rule="rule2a.",
            parent_version="v1",
        )

        v2b = RuleVersion(
            rule_id="v2b",
            rule_family_id="fam1",
            version="1.2.0",
            asp_rule="rule2b.",
            parent_version="v1",
        )

        v1.children_versions = ["v2a", "v2b"]

        lineage = RuleLineage(
            rule_family_id="fam1",
            root_version=v1,
            all_versions=[v1, v2a, v2b],
            current_version=v2a,
            total_iterations=3,
            overall_improvement=0.05,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        tree = lineage.get_version_tree()
        assert "v1" in tree
        assert len(tree["v1"]) == 2
        assert "v2a" in tree["v1"]
        assert "v2b" in tree["v1"]


class TestEvolutionStore:
    """Test evolution storage."""

    def test_save_and_load_version(self, evolution_store):
        """Test saving and loading a version."""
        version = RuleVersion(
            rule_id="test-v1",
            rule_family_id="test-fam",
            version="1.0.0",
            asp_rule="test_rule(X) :- test(X).",
            predicates_used=["test"],
        )

        evolution_store.save_version(version)

        loaded = evolution_store.get_version("test-fam", "test-v1")
        assert loaded is not None
        assert loaded.rule_id == "test-v1"
        assert loaded.asp_rule == version.asp_rule

    def test_get_all_versions(self, evolution_store):
        """Test getting all versions for a family."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="rule1.",
        )

        v2 = RuleVersion(
            rule_id="v2",
            rule_family_id="fam1",
            version="1.0.1",
            asp_rule="rule2.",
            parent_version="v1",
        )

        evolution_store.save_version(v1)
        evolution_store.save_version(v2)

        versions = evolution_store.get_all_versions("fam1")
        assert len(versions) == 2

    def test_save_and_load_lineage(self, evolution_store):
        """Test saving and loading complete lineage."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="rule1.",
        )

        lineage = RuleLineage(
            rule_family_id="fam1",
            root_version=v1,
            all_versions=[v1],
            current_version=v1,
            total_iterations=1,
            overall_improvement=0.0,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        evolution_store.save_lineage(lineage)

        loaded = evolution_store.get_lineage("fam1")
        assert loaded is not None
        assert loaded.rule_family_id == "fam1"
        assert len(loaded.all_versions) == 1

    def test_query_by_method(self, evolution_store):
        """Test querying versions by evolution method."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="rule1.",
            evolution_context=EvolutionContext(evolution_method=EvolutionMethod.DIALECTICAL),
        )

        v2 = RuleVersion(
            rule_id="v2",
            rule_family_id="fam2",
            version="1.0.0",
            asp_rule="rule2.",
            evolution_context=EvolutionContext(evolution_method=EvolutionMethod.MANUAL),
        )

        evolution_store.save_version(v1)
        evolution_store.save_version(v2)

        dialectical_versions = evolution_store.query_by_method("dialectical")
        assert len(dialectical_versions) == 1
        assert dialectical_versions[0].rule_id == "v1"

    def test_query_active_rules(self, evolution_store):
        """Test querying active rules."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="active_rule.",
            is_active=True,
        )

        v2 = RuleVersion(
            rule_id="v2",
            rule_family_id="fam2",
            version="1.0.0",
            asp_rule="inactive_rule.",
            is_active=False,
        )

        lineage1 = RuleLineage(
            rule_family_id="fam1",
            root_version=v1,
            all_versions=[v1],
            current_version=v1,
            total_iterations=1,
            overall_improvement=0.0,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        lineage2 = RuleLineage(
            rule_family_id="fam2",
            root_version=v2,
            all_versions=[v2],
            current_version=v2,
            total_iterations=1,
            overall_improvement=0.0,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        evolution_store.save_lineage(lineage1)
        evolution_store.save_lineage(lineage2)

        active = evolution_store.query_active_rules()
        assert len(active) == 1
        assert active[0].rule_id == "v1"

    def test_export_asp_file(self, evolution_store, temp_store_path):
        """Test exporting to ASP file."""
        v1 = RuleVersion(
            rule_id="v1",
            rule_family_id="fam1",
            version="1.0.0",
            asp_rule="rule1(X) :- condition1(X).",
            is_active=True,
            performance=PerformanceSnapshot(
                timestamp=datetime.now(),
                accuracy=0.8,
                confidence=0.85,
                test_cases_passed=8,
                test_cases_total=10,
                success_rate=0.8,
            ),
        )

        lineage = RuleLineage(
            rule_family_id="fam1",
            root_version=v1,
            all_versions=[v1],
            current_version=v1,
            total_iterations=1,
            overall_improvement=0.0,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        evolution_store.save_lineage(lineage)

        output_path = temp_store_path / "test_output.lp"
        evolution_store.export_asp_file(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "rule1(X) :- condition1(X)." in content
        assert "Confidence: 0.85" in content


class TestEvolutionTracker:
    """Test evolution tracker."""

    def test_track_new_version(self, evolution_tracker):
        """Test tracking a new version."""
        ctx = EvolutionContext(
            evolution_method=EvolutionMethod.MANUAL,
            reasoning="Initial version",
        )

        perf = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.8,
            confidence=0.85,
            test_cases_passed=8,
            test_cases_total=10,
            success_rate=0.8,
        )

        version = evolution_tracker.track_new_version(
            asp_rule="new_rule(X) :- condition(X).",
            evolution_context=ctx,
            performance=perf,
            predicates_used=["condition"],
        )

        assert version is not None
        assert version.version == "1.0.0"
        assert version.asp_rule == "new_rule(X) :- condition(X)."

    def test_track_child_version(self, evolution_tracker):
        """Test tracking a child version."""
        ctx1 = EvolutionContext(evolution_method=EvolutionMethod.MANUAL)
        perf1 = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.7,
            confidence=0.75,
            test_cases_passed=7,
            test_cases_total=10,
            success_rate=0.7,
        )

        v1 = evolution_tracker.track_new_version(
            asp_rule="rule1.",
            evolution_context=ctx1,
            performance=perf1,
        )

        ctx2 = EvolutionContext(evolution_method=EvolutionMethod.REFINEMENT)
        perf2 = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.85,
            confidence=0.90,
            test_cases_passed=8,
            test_cases_total=10,
            success_rate=0.85,
        )

        v2 = evolution_tracker.track_new_version(
            asp_rule="rule2.",
            evolution_context=ctx2,
            performance=perf2,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        assert v2.parent_version == v1.rule_id
        assert v2.version == "1.0.1"
        assert abs(v2.improvement_over_parent - 0.15) < 0.01  # 0.90 - 0.75 (with float tolerance)

    def test_track_from_debate(self, evolution_tracker):
        """Test tracking from dialectical debate."""
        version = evolution_tracker.track_from_debate(
            asp_rule="debated_rule(X) :- debate_condition(X).",
            debate_cycle_id="debate-123",
            round_number=2,
            thesis_rule="original_rule(X) :- condition(X).",
            critique_summary="Missing edge cases",
            confidence=0.88,
            predicates_used=["debate_condition"],
        )

        assert version.evolution_context.evolution_method == EvolutionMethod.DIALECTICAL
        assert version.evolution_context.dialectical_cycle_id == "debate-123"
        assert version.evolution_context.debate_round == 2

    def test_update_performance(self, evolution_tracker):
        """Test updating performance metrics."""
        ctx = EvolutionContext(evolution_method=EvolutionMethod.MANUAL)
        perf1 = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.7,
            confidence=0.75,
            test_cases_passed=7,
            test_cases_total=10,
            success_rate=0.7,
        )

        version = evolution_tracker.track_new_version(
            asp_rule="test_rule.",
            evolution_context=ctx,
            performance=perf1,
        )

        # Update performance
        perf2 = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.85,
            confidence=0.90,
            test_cases_passed=9,
            test_cases_total=10,
            success_rate=0.9,
        )

        evolution_tracker.update_performance(
            version.rule_family_id,
            version.rule_id,
            perf2,
        )

        # Verify update
        loaded = evolution_tracker.store.get_version(version.rule_family_id, version.rule_id)
        assert loaded.performance.confidence == 0.90

    def test_deprecate_version(self, evolution_tracker):
        """Test deprecating a version."""
        ctx = EvolutionContext(evolution_method=EvolutionMethod.MANUAL)
        v1 = evolution_tracker.track_new_version(
            asp_rule="old_rule.",
            evolution_context=ctx,
        )

        v2 = evolution_tracker.track_new_version(
            asp_rule="new_rule.",
            evolution_context=ctx,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        evolution_tracker.deprecate_version(
            v1.rule_family_id,
            v1.rule_id,
            replaced_by=v2.rule_id,
            reason="Improved version available",
        )

        loaded = evolution_tracker.store.get_version(v1.rule_family_id, v1.rule_id)
        assert loaded.is_deprecated is True
        assert loaded.is_active is False
        assert loaded.replaced_by == v2.rule_id

    def test_get_evolution_path(self, evolution_tracker):
        """Test getting evolution path."""
        ctx = EvolutionContext(evolution_method=EvolutionMethod.MANUAL)

        v1 = evolution_tracker.track_new_version(
            asp_rule="rule1.",
            evolution_context=ctx,
        )

        v2 = evolution_tracker.track_new_version(
            asp_rule="rule2.",
            evolution_context=ctx,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        v3 = evolution_tracker.track_new_version(
            asp_rule="rule3.",
            evolution_context=ctx,
            parent_id=v2.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        path = evolution_tracker.get_evolution_path(v1.rule_family_id)
        assert len(path) == 3
        assert path[0].rule_id == v1.rule_id
        assert path[1].rule_id == v2.rule_id
        assert path[2].rule_id == v3.rule_id

    def test_analyze_evolution_patterns(self, evolution_tracker):
        """Test analyzing evolution patterns."""
        # Create multiple versions with different methods
        ctx_manual = EvolutionContext(evolution_method=EvolutionMethod.MANUAL)
        ctx_dialectical = EvolutionContext(evolution_method=EvolutionMethod.DIALECTICAL)

        perf1 = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.7,
            confidence=0.75,
            test_cases_passed=7,
            test_cases_total=10,
            success_rate=0.7,
        )

        perf2 = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.85,
            confidence=0.90,
            test_cases_passed=9,
            test_cases_total=10,
            success_rate=0.9,
        )

        v1 = evolution_tracker.track_new_version(
            asp_rule="rule1.",
            evolution_context=ctx_manual,
            performance=perf1,
        )

        evolution_tracker.track_new_version(
            asp_rule="rule2.",
            evolution_context=ctx_dialectical,
            performance=perf2,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        analysis = evolution_tracker.analyze_evolution_patterns()
        assert analysis["total_families"] >= 1
        assert "manual" in analysis["methods_used"]
        assert "dialectical" in analysis["methods_used"]

    def test_rollback_to_version(self, evolution_tracker):
        """Test rolling back to previous version."""
        ctx = EvolutionContext(evolution_method=EvolutionMethod.MANUAL)

        v1 = evolution_tracker.track_new_version(
            asp_rule="rule1.",
            evolution_context=ctx,
        )

        evolution_tracker.track_new_version(
            asp_rule="rule2.",
            evolution_context=ctx,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        # Rollback to v1
        rolled_back = evolution_tracker.rollback_to_version(v1.rule_family_id, v1.rule_id)

        assert rolled_back.rule_id == v1.rule_id
        assert rolled_back.is_active is True

        # Verify lineage was updated
        lineage = evolution_tracker.get_lineage(v1.rule_family_id)
        assert lineage.current_version.rule_id == v1.rule_id
