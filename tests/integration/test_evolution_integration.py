"""
Integration tests for rule evolution tracking with debate framework.

Tests the complete flow:
1. Debate framework generates rules
2. Evolution tracker records versions
3. Performance updates from validation
4. Complete lineage tracking
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

from loft.evolution.evolution_tracker import RuleEvolutionTracker
from loft.evolution.evolution_store import RuleEvolutionStore
from loft.evolution.evolution_schemas import (
    EvolutionMethod,
    StratificationLevel,
    PerformanceSnapshot,
)
from loft.dialectical.debate_framework import (
    DebateFramework,
)
from loft.dialectical.critic import CriticSystem
from loft.dialectical.synthesizer import Synthesizer


class TestEvolutionIntegration:
    """Integration tests for evolution tracking with debate framework."""

    @pytest.fixture
    def temp_store(self):
        """Create temporary evolution store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuleEvolutionStore(base_path=Path(tmpdir))
            yield store

    @pytest.fixture
    def tracker(self, temp_store):
        """Create evolution tracker with temporary store."""
        return RuleEvolutionTracker(store=temp_store)

    @pytest.fixture
    def mock_debate_framework(self):
        """Create mock debate framework."""
        mock = Mock(spec=DebateFramework)
        mock.critic = Mock(spec=CriticSystem)
        mock.synthesizer = Mock(spec=Synthesizer)
        return mock

    def test_track_debate_cycle(self, tracker):
        """Test tracking a complete debate cycle."""
        # Round 1: Initial thesis
        v1 = tracker.track_from_debate(
            asp_rule="can_file(Party, Doc) :- has_standing(Party), relevant_document(Doc).",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="can_file(Party, Doc) :- has_standing(Party), relevant_document(Doc).",
            critique_summary="Initial thesis - needs refinement",
            confidence=0.7,
            predicates_used=["can_file", "has_standing", "relevant_document"],
        )

        assert v1.version == "1.0.0"
        assert v1.evolution_context.evolution_method == EvolutionMethod.DIALECTICAL
        assert v1.evolution_context.debate_round == 1
        assert v1.performance.confidence == 0.7

        # Round 2: Refined after critique
        v2 = tracker.track_from_debate(
            asp_rule="can_file(Party, Doc) :- has_standing(Party), relevant_document(Doc), timely_filed(Doc).",
            debate_cycle_id="cycle-001",
            round_number=2,
            thesis_rule="can_file(Party, Doc) :- has_standing(Party), relevant_document(Doc), timely_filed(Doc).",
            critique_summary="Added timeliness requirement",
            confidence=0.85,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
            predicates_used=[
                "can_file",
                "has_standing",
                "relevant_document",
                "timely_filed",
            ],
        )

        assert v2.version == "1.0.1"
        assert v2.parent_version == v1.rule_id
        assert v2.evolution_context.debate_round == 2
        assert abs(v2.improvement_over_parent - 0.15) < 0.01

        # Verify lineage
        lineage = tracker.get_lineage(v1.rule_family_id)
        assert lineage is not None
        assert len(lineage.all_versions) == 2
        assert lineage.total_iterations == 2

        path = lineage.get_evolution_path()
        assert len(path) == 2
        assert path[0].rule_id == v1.rule_id
        assert path[1].rule_id == v2.rule_id

    def test_multiple_debate_cycles(self, tracker):
        """Test tracking multiple independent debate cycles."""
        # First rule family
        family1_v1 = tracker.track_from_debate(
            asp_rule="rule1_v1.",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="rule1_v1.",
            critique_summary="Family 1 initial",
            confidence=0.6,
        )

        # Second rule family
        family2_v1 = tracker.track_from_debate(
            asp_rule="rule2_v1.",
            debate_cycle_id="cycle-002",
            round_number=1,
            thesis_rule="rule2_v1.",
            critique_summary="Family 2 initial",
            confidence=0.7,
        )

        assert family1_v1.rule_family_id != family2_v1.rule_family_id

        # Continue family 1
        tracker.track_from_debate(
            asp_rule="rule1_v2.",
            debate_cycle_id="cycle-001",
            round_number=2,
            thesis_rule="rule1_v2.",
            critique_summary="Family 1 refined",
            confidence=0.75,
            parent_id=family1_v1.rule_id,
            rule_family_id=family1_v1.rule_family_id,
        )

        # Verify independent lineages
        lineage1 = tracker.get_lineage(family1_v1.rule_family_id)
        lineage2 = tracker.get_lineage(family2_v1.rule_family_id)

        assert len(lineage1.all_versions) == 2
        assert len(lineage2.all_versions) == 1

    def test_performance_update_integration(self, tracker):
        """Test updating performance after validation."""
        # Track initial version
        v1 = tracker.track_from_debate(
            asp_rule="test_rule.",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="test_rule.",
            critique_summary="Initial",
            confidence=0.6,
        )

        # Simulate validation updating performance
        updated_performance = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.85,
            confidence=0.82,
            test_cases_passed=17,
            test_cases_total=20,
            success_rate=0.85,
            validation_report_id="val-001",
            validation_method="automated_test_suite",
        )

        tracker.update_performance(
            rule_family_id=v1.rule_family_id,
            version_id=v1.rule_id,
            performance=updated_performance,
        )

        # Verify update by reloading version from store
        updated_version = tracker.store.get_version(v1.rule_family_id, v1.rule_id)

        assert updated_version.performance.accuracy == 0.85
        assert updated_version.performance.test_cases_passed == 17
        assert updated_version.performance.validation_report_id == "val-001"

    def test_export_to_asp_file(self, tracker, temp_store):
        """Test exporting active rules to ASP file format."""
        # Track multiple versions
        tracker.track_from_debate(
            asp_rule="rule1 :- condition1.",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="rule1 :- condition1.",
            critique_summary="Rule 1",
            confidence=0.8,
        )

        tracker.track_from_debate(
            asp_rule="rule2 :- condition2.",
            debate_cycle_id="cycle-002",
            round_number=1,
            thesis_rule="rule2 :- condition2.",
            critique_summary="Rule 2",
            confidence=0.85,
        )

        # Export to ASP file
        output_path = Path(temp_store.base_path) / "output" / "rules.lp"
        temp_store.export_asp_file(output_path)

        # Verify file content
        assert output_path.exists()
        content = output_path.read_text()

        assert "rule1 :- condition1." in content
        assert "rule2 :- condition2." in content
        assert "Version:" in content
        assert "Confidence:" in content

    def test_stratification_tracking(self, tracker):
        """Test tracking rules at different stratification levels."""
        # Constitutional level
        v_constitutional = tracker.track_from_debate(
            asp_rule="constitutional_rule.",
            debate_cycle_id="cycle-const",
            round_number=1,
            thesis_rule="constitutional_rule.",
            critique_summary="Top level",
            confidence=0.95,
        )

        # Update to constitutional level
        version = tracker.store.get_version(
            v_constitutional.rule_family_id, v_constitutional.rule_id
        )
        version.stratification_level = StratificationLevel.CONSTITUTIONAL
        tracker.store.save_version(version)

        # Tactical level
        v_tactical = tracker.track_from_debate(
            asp_rule="tactical_rule.",
            debate_cycle_id="cycle-tact",
            round_number=1,
            thesis_rule="tactical_rule.",
            critique_summary="Mid level",
            confidence=0.8,
        )

        # Query by stratification
        constitutional_rules = tracker.store.query_by_stratification("constitutional")
        tactical_rules = tracker.store.query_by_stratification("tactical")

        assert len(constitutional_rules) == 1
        assert len(tactical_rules) == 1
        assert constitutional_rules[0].rule_id == v_constitutional.rule_id
        assert tactical_rules[0].rule_id == v_tactical.rule_id

    def test_evolution_analysis(self, tracker):
        """Test analyzing evolution patterns."""
        # Create multiple versions with different methods
        # Dialectical
        v1 = tracker.track_from_debate(
            asp_rule="rule1.",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="rule1.",
            critique_summary="Dialectical",
            confidence=0.7,
        )

        tracker.track_from_debate(
            asp_rule="rule1_v2.",
            debate_cycle_id="cycle-001",
            round_number=2,
            thesis_rule="rule1_v2.",
            critique_summary="Dialectical refined",
            confidence=0.85,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        # Analyze patterns
        analysis = tracker.analyze_evolution_patterns()

        assert analysis["total_families"] >= 1
        assert analysis["total_versions"] >= 2
        assert "dialectical" in analysis["methods_used"]
        assert analysis["methods_used"]["dialectical"] >= 2
        assert "dialectical" in analysis["avg_improvement_by_method"]

    def test_deprecation_workflow(self, tracker):
        """Test deprecating and replacing versions."""
        # Original version
        v1 = tracker.track_from_debate(
            asp_rule="old_rule.",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="old_rule.",
            critique_summary="Original",
            confidence=0.6,
        )

        # Better replacement
        v2 = tracker.track_from_debate(
            asp_rule="new_rule.",
            debate_cycle_id="cycle-001",
            round_number=2,
            thesis_rule="new_rule.",
            critique_summary="Improved replacement",
            confidence=0.9,
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        # Deprecate old version
        tracker.deprecate_version(
            rule_family_id=v1.rule_family_id,
            version_id=v1.rule_id,
            replaced_by=v2.rule_id,
            reason="Superseded by improved version with higher confidence",
        )

        # Verify deprecation
        v1_updated = tracker.store.get_version(v1.rule_family_id, v1.rule_id)
        assert v1_updated.is_deprecated is True
        assert v1_updated.is_active is False
        assert v1_updated.replaced_by == v2.rule_id

    def test_rollback_integration(self, tracker):
        """Test rolling back to previous version."""
        # Create progression
        v1 = tracker.track_from_debate(
            asp_rule="rule_v1.",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="rule_v1.",
            critique_summary="Version 1",
            confidence=0.8,
        )

        tracker.track_from_debate(
            asp_rule="rule_v2.",
            debate_cycle_id="cycle-001",
            round_number=2,
            thesis_rule="rule_v2.",
            critique_summary="Version 2",
            confidence=0.7,  # Worse!
            parent_id=v1.rule_id,
            rule_family_id=v1.rule_family_id,
        )

        # Rollback because v2 performed worse
        rolled_back = tracker.rollback_to_version(v1.rule_family_id, v1.rule_id)

        assert rolled_back.rule_id == v1.rule_id
        assert rolled_back.is_active is True

        # Verify lineage updated
        lineage = tracker.get_lineage(v1.rule_family_id)
        assert lineage.current_version.rule_id == v1.rule_id

    def test_top_performers_query(self, tracker):
        """Test querying top performing rules."""
        # Create rules with different performance
        rules = []
        for i in range(5):
            v = tracker.track_from_debate(
                asp_rule=f"rule_{i}.",
                debate_cycle_id=f"cycle-{i:03d}",
                round_number=1,
                thesis_rule=f"rule_{i}.",
                critique_summary=f"Rule {i}",
                confidence=0.5 + (i * 0.1),
            )
            rules.append(v)

        # Query top 3
        top_performers = tracker.get_top_performers(n=3)

        assert len(top_performers) == 3
        # Should be sorted by confidence descending
        assert (
            top_performers[0].performance.confidence
            >= top_performers[1].performance.confidence
        )
        assert (
            top_performers[1].performance.confidence
            >= top_performers[2].performance.confidence
        )

    def test_most_iterated_families(self, tracker):
        """Test finding most iterated rule families."""
        # Create family with many iterations
        v1 = tracker.track_from_debate(
            asp_rule="rule1.",
            debate_cycle_id="cycle-001",
            round_number=1,
            thesis_rule="rule1.",
            critique_summary="V1",
            confidence=0.6,
        )

        # Add multiple iterations
        current_parent = v1.rule_id
        for i in range(2, 6):
            v = tracker.track_from_debate(
                asp_rule=f"rule1_v{i}.",
                debate_cycle_id="cycle-001",
                round_number=i,
                thesis_rule=f"rule1_v{i}.",
                critique_summary=f"V{i}",
                confidence=0.6 + (i * 0.05),
                parent_id=current_parent,
                rule_family_id=v1.rule_family_id,
            )
            current_parent = v.rule_id

        # Create another family with fewer iterations
        tracker.track_from_debate(
            asp_rule="rule2.",
            debate_cycle_id="cycle-002",
            round_number=1,
            thesis_rule="rule2.",
            critique_summary="V1",
            confidence=0.8,
        )

        # Query most iterated
        most_iterated = tracker.get_most_iterated(n=2)

        assert len(most_iterated) >= 1
        # First should be the family with 5 versions
        assert most_iterated[0][0] == v1.rule_family_id
        assert most_iterated[0][1] == 5
