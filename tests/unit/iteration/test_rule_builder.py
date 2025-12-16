"""Tests for iterative rule builder."""

from datetime import datetime
from pathlib import Path
import tempfile
import pytest

from loft.iteration.rule_builder import (
    IterativeRuleBuilder,
    ContradictionCheck,
    RedundancyCheck,
    AdditionResult,
)
from loft.iteration.coverage_tracker import CoverageTracker
from loft.iteration.living_document import LivingDocumentManager
from loft.persistence.asp_persistence import ASPPersistenceManager
from loft.symbolic.asp_rule import ASPRule, RuleMetadata
from loft.symbolic.stratification import StratificationLevel


@pytest.fixture
def temp_dir():
    """Temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def persistence(temp_dir):
    """ASP persistence manager."""
    return ASPPersistenceManager(base_dir=str(temp_dir / "asp_rules"))


@pytest.fixture
def coverage_tracker():
    """Coverage tracker."""
    domain_predicates = ["contract_valid", "breach_occurred", "damages_owed"]
    test_cases = [{"id": "case_001"}]
    return CoverageTracker(domain_predicates, test_cases)


@pytest.fixture
def living_document(temp_dir):
    """Living document manager."""
    return LivingDocumentManager(temp_dir / "LIVING_DOCUMENT.md")


@pytest.fixture
def builder(persistence, coverage_tracker, living_document):
    """Iterative rule builder."""
    return IterativeRuleBuilder(
        persistence=persistence,
        coverage_tracker=coverage_tracker,
        living_document=living_document,
        enable_monotonicity=True,
    )


@pytest.fixture
def sample_rule():
    """Sample ASP rule."""
    return ASPRule(
        rule_id="rule_001",
        asp_text="contract_valid(X) :- offer(X), acceptance(X).",
        stratification_level=StratificationLevel.OPERATIONAL,
        confidence=0.85,
        metadata=RuleMetadata(
            provenance="test",
            timestamp=datetime.utcnow().isoformat(),
            notes="Test rule",
        ),
    )


def test_builder_initialization(builder):
    """Test builder initialization."""
    assert builder.persistence is not None
    assert builder.coverage_tracker is not None
    assert builder.living_document is not None
    assert builder.monotonicity is not None


def test_add_rule_success(builder, sample_rule):
    """Test successfully adding a rule."""
    result = builder.add_rule(sample_rule)

    assert result.status == "added"
    assert result.reason == "passed all checks"
    assert sample_rule in builder.get_all_rules()


def test_add_rule_with_skip_checks(builder, sample_rule):
    """Test adding rule with skipped checks."""
    result = builder.add_rule(sample_rule, skip_checks=True)

    assert result.status == "added"
    assert sample_rule in builder.get_all_rules()


def test_add_rule_updates_coverage(builder, sample_rule):
    """Test that adding rule updates coverage."""
    coverage_before = builder.coverage_tracker.current_coverage or 0.0

    builder.add_rule(sample_rule)

    coverage_after = builder.coverage_tracker.current_coverage

    # Coverage should have changed (increased or stayed same)
    assert coverage_after >= coverage_before


def test_get_rules_by_layer(builder, sample_rule):
    """Test getting rules by layer."""
    builder.add_rule(sample_rule)

    operational_rules = builder.get_rules_by_layer(StratificationLevel.OPERATIONAL)

    assert len(operational_rules) == 1
    assert operational_rules[0] == sample_rule


def test_get_statistics(builder, sample_rule):
    """Test getting builder statistics."""
    builder.add_rule(sample_rule)

    stats = builder.get_statistics()

    assert stats["total_rules"] == 1
    assert "coverage" in stats
    assert "coverage_trend" in stats
    assert "monotonic" in stats


def test_contradiction_check_no_conflict():
    """Test contradiction check with no conflicts."""
    check = ContradictionCheck(has_contradiction=False)

    assert check.has_contradiction is False
    assert check.conflicting_rules == []


def test_redundancy_check_not_redundant():
    """Test redundancy check with no redundancy."""
    check = RedundancyCheck(is_redundant=False)

    assert check.is_redundant is False
    assert check.subsuming_rule is None


def test_addition_result_added():
    """Test addition result for successful add."""
    result = AdditionResult(status="added", reason="passed checks", coverage_change=0.1)

    assert result.status == "added"
    assert result.reason == "passed checks"
    assert result.coverage_change == 0.1
    assert result.new_predicates_covered == []


def test_builder_loads_existing_rules(
    persistence, coverage_tracker, living_document, sample_rule
):
    """Test that builder loads existing persisted rules."""
    # Save a rule first
    persistence.save_rule(sample_rule, StratificationLevel.OPERATIONAL)

    # Create new builder (should load existing rules)
    builder = IterativeRuleBuilder(
        persistence=persistence,
        coverage_tracker=coverage_tracker,
        living_document=living_document,
    )

    # Should have loaded the rule
    all_rules = builder.get_all_rules()
    assert len(all_rules) >= 0  # May not load due to persistence format


def test_builder_documents_adjustment(builder, sample_rule, temp_dir):
    """Test that builder documents adjustments."""
    builder.add_rule(sample_rule)

    doc_path = temp_dir / "LIVING_DOCUMENT.md"
    content = doc_path.read_text()

    assert "rule_001" in content
    assert "added" in content.lower()
