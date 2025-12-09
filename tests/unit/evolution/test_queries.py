"""Tests for rule evolution queries."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from loft.evolution.tracking import (
    RuleMetadata,
    ValidationResult,
    StratificationLayer,
    RuleStatus,
)
from loft.evolution.storage import (
    RuleEvolutionStorage,
    StorageConfig,
)
from loft.evolution.queries import (
    EvolutionQuery,
    QueryResult,
    RuleEvolutionDB,
    SortField,
    SortOrder,
)


@pytest.fixture
def temp_db():
    """Create a temporary database with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(base_path=Path(tmpdir))
        storage = RuleEvolutionStorage(config)
        db = RuleEvolutionDB(storage)

        # Add sample rules
        base_time = datetime.now() - timedelta(days=30)

        for i in range(5):
            rule = RuleMetadata(
                rule_id=f"rule_{i}",
                rule_text=f"rule text {i}",
                natural_language=f"description {i}",
                created_by="test",
                version=f"{i + 1}.0",
                created_at=base_time + timedelta(days=i * 5),
                status=RuleStatus.ACTIVE if i % 2 == 0 else RuleStatus.CANDIDATE,
                current_layer=(
                    StratificationLayer.OPERATIONAL
                    if i < 3
                    else StratificationLayer.TACTICAL
                ),
            )
            # Add validation result
            rule.validation_results.append(
                ValidationResult(
                    timestamp=base_time + timedelta(days=i * 5),
                    test_cases_evaluated=100,
                    passed=int((0.6 + i * 0.08) * 100),
                    failed=int((0.4 - i * 0.08) * 100),
                    accuracy=0.6 + i * 0.08,
                )
            )
            storage.save_rule(rule)

        yield db


class TestEvolutionQuery:
    """Tests for EvolutionQuery class."""

    def test_default_query(self):
        """Test default query parameters."""
        query = EvolutionQuery()

        assert query.status_filter is None
        assert query.layer_filter is None
        assert query.sort_field == SortField.CREATED_AT
        assert query.sort_order == SortOrder.DESC

    def test_fluent_interface(self):
        """Test fluent query building."""
        query = (
            EvolutionQuery()
            .with_status(RuleStatus.ACTIVE)
            .with_min_accuracy(0.8)
            .in_layer(StratificationLayer.TACTICAL)
            .sort_by(SortField.ACCURACY, SortOrder.DESC)
            .limit(10)
        )

        assert query.status_filter == RuleStatus.ACTIVE
        assert query.min_accuracy == 0.8
        assert query.layer_filter == StratificationLayer.TACTICAL
        assert query.sort_field == SortField.ACCURACY
        assert query.limit_count == 10

    def test_matches_status_filter(self):
        """Test status filter matching."""
        query = EvolutionQuery().with_status(RuleStatus.ACTIVE)

        active_rule = RuleMetadata(
            rule_id="active",
            rule_text="test",
            natural_language="test",
            created_by="test",
            status=RuleStatus.ACTIVE,
        )
        candidate_rule = RuleMetadata(
            rule_id="candidate",
            rule_text="test",
            natural_language="test",
            created_by="test",
            status=RuleStatus.CANDIDATE,
        )

        assert query.matches(active_rule)
        assert not query.matches(candidate_rule)

    def test_matches_accuracy_filter(self):
        """Test accuracy filter matching."""
        query = EvolutionQuery().with_min_accuracy(0.7).with_max_accuracy(0.9)

        low_rule = RuleMetadata(
            rule_id="low",
            rule_text="test",
            natural_language="test",
            created_by="test",
        )
        low_rule.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=50,
                failed=50,
                accuracy=0.5,
            )
        )

        mid_rule = RuleMetadata(
            rule_id="mid",
            rule_text="test",
            natural_language="test",
            created_by="test",
        )
        mid_rule.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=80,
                failed=20,
                accuracy=0.8,
            )
        )

        high_rule = RuleMetadata(
            rule_id="high",
            rule_text="test",
            natural_language="test",
            created_by="test",
        )
        high_rule.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=95,
                failed=5,
                accuracy=0.95,
            )
        )

        assert not query.matches(low_rule)
        assert query.matches(mid_rule)
        assert not query.matches(high_rule)

    def test_matches_text_search(self):
        """Test text search matching."""
        query = EvolutionQuery().containing_text("contract")

        matching_rule = RuleMetadata(
            rule_id="match",
            rule_text="enforceable(Contract)",
            natural_language="Contract must be valid",
            created_by="test",
        )
        non_matching = RuleMetadata(
            rule_id="no_match",
            rule_text="valid(X)",
            natural_language="Something else",
            created_by="test",
        )

        assert query.matches(matching_rule)
        assert not query.matches(non_matching)

    def test_custom_filter(self):
        """Test custom predicate filter."""
        query = EvolutionQuery().where(
            lambda r: r.version == "2.0" or r.version == "3.0"
        )

        v1 = RuleMetadata(
            rule_id="v1",
            rule_text="test",
            natural_language="test",
            created_by="test",
            version="1.0",
        )
        v2 = RuleMetadata(
            rule_id="v2",
            rule_text="test",
            natural_language="test",
            created_by="test",
            version="2.0",
        )

        assert not query.matches(v1)
        assert query.matches(v2)


class TestRuleEvolutionDB:
    """Tests for RuleEvolutionDB class."""

    def test_execute_basic_query(self, temp_db):
        """Test executing a basic query."""
        result = temp_db.execute(EvolutionQuery())

        assert result.count == 5
        assert result.total_count == 5

    def test_execute_with_status_filter(self, temp_db):
        """Test query with status filter."""
        result = temp_db.execute(EvolutionQuery().with_status(RuleStatus.ACTIVE))

        assert result.count == 3  # Rules 0, 2, 4 are active
        assert all(r.status == RuleStatus.ACTIVE for r in result.rules)

    def test_execute_with_layer_filter(self, temp_db):
        """Test query with layer filter."""
        result = temp_db.execute(
            EvolutionQuery().in_layer(StratificationLayer.TACTICAL)
        )

        assert result.count == 2  # Rules 3, 4 are tactical
        assert all(
            r.current_layer == StratificationLayer.TACTICAL for r in result.rules
        )

    def test_execute_with_sorting(self, temp_db):
        """Test query with sorting."""
        result = temp_db.execute(
            EvolutionQuery().sort_by(SortField.ACCURACY, SortOrder.DESC)
        )

        accuracies = [r.current_accuracy for r in result.rules]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_execute_with_limit(self, temp_db):
        """Test query with limit."""
        result = temp_db.execute(EvolutionQuery().limit(2))

        assert result.count == 2
        assert result.total_count == 5
        assert result.has_more

    def test_execute_with_offset(self, temp_db):
        """Test query with offset."""
        result = temp_db.execute(EvolutionQuery().offset(3))

        assert result.count == 2  # 5 total - 3 offset = 2 remaining

    def test_get_rule_by_id(self, temp_db):
        """Test getting rule by ID."""
        rule = temp_db.get_rule_by_id("rule_0")

        assert rule is not None
        assert rule.rule_id == "rule_0"

    def test_get_nonexistent_rule(self, temp_db):
        """Test getting non-existent rule."""
        rule = temp_db.get_rule_by_id("nonexistent")

        assert rule is None

    def test_get_evolution_stats(self, temp_db):
        """Test getting evolution statistics."""
        stats = temp_db.get_evolution_stats()

        assert stats["total_rules"] == 5
        assert stats["status_distribution"]["active"] == 3
        assert stats["status_distribution"]["candidate"] == 2

    def test_find_improvement_candidates(self, temp_db):
        """Test finding improvement candidates."""
        candidates = temp_db.find_improvement_candidates(
            min_accuracy=0.6, max_accuracy=0.8
        )

        assert len(candidates) > 0
        for c in candidates:
            assert 0.6 <= c.current_accuracy <= 0.8

    def test_get_top_performers(self, temp_db):
        """Test getting top performers."""
        top = temp_db.get_top_performers(limit=3)

        assert len(top) == 3
        # Should be sorted by accuracy descending
        assert top[0].current_accuracy >= top[1].current_accuracy
        assert top[1].current_accuracy >= top[2].current_accuracy

    def test_get_recently_modified(self, temp_db):
        """Test getting recently modified rules."""
        recent = temp_db.get_recently_modified(limit=2)

        assert len(recent) == 2
        # Should be sorted by created_at descending
        assert recent[0].created_at >= recent[1].created_at


class TestQueryResult:
    """Tests for QueryResult class."""

    def test_result_properties(self):
        """Test result properties."""
        rules = [
            RuleMetadata(
                rule_id=f"rule_{i}",
                rule_text="test",
                natural_language="test",
                created_by="test",
            )
            for i in range(3)
        ]

        result = QueryResult(
            rules=rules,
            total_count=10,
            query=EvolutionQuery().limit(3),
        )

        assert result.count == 3
        assert result.has_more
        assert result.first().rule_id == "rule_0"
        assert len(result.ids()) == 3

    def test_empty_result(self):
        """Test empty result."""
        result = QueryResult(
            rules=[],
            total_count=0,
            query=EvolutionQuery(),
        )

        assert result.count == 0
        assert not result.has_more
        assert result.first() is None
        assert result.ids() == []
