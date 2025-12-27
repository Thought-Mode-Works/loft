"""
Unit tests for knowledge database operations.

Tests CRUD operations, search queries, and performance tracking.

Issue #271: Persistent Legal Knowledge Database
"""

import pytest

from loft.knowledge.database import KnowledgeDatabase


class TestKnowledgeDatabase:
    """Unit tests for KnowledgeDatabase class."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test.db"
        return KnowledgeDatabase(f"sqlite:///{db_path}")

    def test_init_creates_tables(self, tmp_path):
        """Test database initialization creates all tables."""
        db_path = tmp_path / "test.db"
        db = KnowledgeDatabase(f"sqlite:///{db_path}")

        # Check database file exists
        assert db_path.exists()

        db.close()

    def test_add_rule_success(self, db):
        """Test adding a rule to the database."""
        rule_id = db.add_rule(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
            domain="contracts",
            doctrine="offer-acceptance",
            confidence=0.95,
            reasoning="A valid contract requires three essential elements",
        )

        assert rule_id is not None
        assert len(rule_id) == 36  # UUID format

    def test_add_duplicate_rule_raises_error(self, db):
        """Test that adding duplicate rule raises ValueError."""
        asp_rule = "test_rule(X) :- condition(X)."

        # Add first time
        db.add_rule(asp_rule=asp_rule, domain="test")

        # Try to add again - should raise error
        with pytest.raises(ValueError, match="already exists"):
            db.add_rule(asp_rule=asp_rule, domain="test")

    def test_get_rule_by_id(self, db):
        """Test retrieving a rule by ID."""
        rule_id = db.add_rule(
            asp_rule="test_rule(X) :- test(X).",
            domain="test",
            confidence=0.9,
        )

        rule = db.get_rule(rule_id)

        assert rule is not None
        assert rule.rule_id == rule_id
        assert rule.asp_rule == "test_rule(X) :- test(X)."
        assert rule.domain == "test"
        assert rule.confidence == 0.9

    def test_get_nonexistent_rule_returns_none(self, db):
        """Test that getting nonexistent rule returns None."""
        rule = db.get_rule("nonexistent-id")
        assert rule is None

    def test_search_rules_by_domain(self, db):
        """Test searching rules by domain."""
        # Add rules in different domains
        db.add_rule(asp_rule="contract_rule(X) :- offer(X).", domain="contracts")
        db.add_rule(asp_rule="tort_rule(X) :- negligence(X).", domain="torts")
        db.add_rule(asp_rule="property_rule(X) :- ownership(X).", domain="property")

        # Search for contracts
        results = db.search_rules(domain="contracts")

        assert len(results) == 1
        assert results[0].domain == "contracts"

    def test_search_rules_by_confidence(self, db):
        """Test searching rules by minimum confidence."""
        db.add_rule(asp_rule="high_conf(X).", confidence=0.95)
        db.add_rule(asp_rule="med_conf(X).", confidence=0.75)
        db.add_rule(asp_rule="low_conf(X).", confidence=0.50)

        # Search for high confidence rules
        results = db.search_rules(min_confidence=0.8)

        assert len(results) == 1
        assert results[0].confidence >= 0.8

    def test_search_rules_pagination(self, db):
        """Test rule search pagination."""
        # Add 15 rules
        for i in range(15):
            db.add_rule(asp_rule=f"rule_{i}(X).", domain="test")

        # Get first page
        page1 = db.search_rules(domain="test", limit=5, offset=0)
        assert len(page1) == 5

        # Get second page
        page2 = db.search_rules(domain="test", limit=5, offset=5)
        assert len(page2) == 5

        # Ensure pages don't overlap
        page1_ids = {r.rule_id for r in page1}
        page2_ids = {r.rule_id for r in page2}
        assert len(page1_ids & page2_ids) == 0

    def test_update_rule(self, db):
        """Test updating a rule."""
        rule_id = db.add_rule(asp_rule="original(X).", domain="test", confidence=0.7)

        # Update confidence
        success = db.update_rule(rule_id, confidence=0.9)
        assert success is True

        # Verify update
        rule = db.get_rule(rule_id)
        assert rule.confidence == 0.9

    def test_update_nonexistent_rule_returns_false(self, db):
        """Test updating nonexistent rule returns False."""
        success = db.update_rule("nonexistent-id", confidence=0.9)
        assert success is False

    def test_update_rule_performance_success(self, db):
        """Test updating rule performance after successful use."""
        rule_id = db.add_rule(asp_rule="test(X).", domain="test")

        # Record success
        db.update_rule_performance(rule_id, success=True)

        rule = db.get_rule(rule_id)
        assert rule.validation_count == 1
        assert rule.success_count == 1
        assert rule.failure_count == 0
        assert rule.last_success_date is not None

    def test_update_rule_performance_failure(self, db):
        """Test updating rule performance after failure."""
        rule_id = db.add_rule(asp_rule="test(X).", domain="test")

        # Record failure
        db.update_rule_performance(rule_id, success=False)

        rule = db.get_rule(rule_id)
        assert rule.validation_count == 1
        assert rule.success_count == 0
        assert rule.failure_count == 1
        assert rule.last_failure_date is not None

    def test_mark_rule_used(self, db):
        """Test marking a rule as used."""
        rule_id = db.add_rule(asp_rule="test(X).", domain="test")

        # Initially last_used is None
        rule = db.get_rule(rule_id)
        assert rule.last_used is None

        # Mark as used
        db.mark_rule_used(rule_id)

        # Verify last_used is set
        rule = db.get_rule(rule_id)
        assert rule.last_used is not None

    def test_archive_rule(self, db):
        """Test archiving a rule."""
        rule_id = db.add_rule(asp_rule="test(X).", domain="test")

        # Archive
        success = db.archive_rule(rule_id, reason="No longer relevant")
        assert success is True

        # Verify archived
        rule = db.get_rule(rule_id)
        assert rule.is_active is False
        assert rule.is_archived is True

        # Archived rules not returned in default search
        results = db.search_rules(domain="test", is_active=True)
        assert len(results) == 0

    def test_delete_rule(self, db):
        """Test permanently deleting a rule."""
        rule_id = db.add_rule(asp_rule="test(X).", domain="test")

        # Delete
        success = db.delete_rule(rule_id)
        assert success is True

        # Verify deleted
        rule = db.get_rule(rule_id)
        assert rule is None

    def test_add_question(self, db):
        """Test recording a question."""
        question_id = db.add_question(
            question_text="Is a contract valid without consideration?",
            asp_query="?- valid_contract(c).",
            answer="no",
            confidence=0.95,
            correct=True,
            domain="contracts",
        )

        assert question_id is not None
        assert len(question_id) == 36

    def test_get_coverage_stats(self, db):
        """Test getting coverage statistics for a domain."""
        # Add rules and questions
        db.add_rule(asp_rule="rule1(X).", domain="contracts", confidence=0.9)
        db.add_rule(asp_rule="rule2(X).", domain="contracts", confidence=0.95)
        db.add_question(
            question_text="Q1",
            answer="yes",
            correct=True,
            domain="contracts",
        )

        stats = db.get_coverage_stats("contracts")

        assert stats is not None
        assert stats.domain == "contracts"
        assert stats.rule_count == 2
        assert stats.question_count == 1
        assert stats.accuracy == 1.0  # 1/1 correct

    def test_get_database_stats(self, db):
        """Test getting overall database statistics."""
        # Add some data
        db.add_rule(asp_rule="c1(X).", domain="contracts")
        db.add_rule(asp_rule="t1(X).", domain="torts")
        db.add_question(question_text="Q1", domain="contracts")

        stats = db.get_database_stats()

        assert stats.total_rules == 2
        assert stats.active_rules == 2
        assert stats.total_questions == 1
        assert "contracts" in stats.domains
        assert "torts" in stats.domains

    def test_export_to_asp_files(self, db, tmp_path):
        """Test exporting rules to ASP files."""
        # Add rules at different stratification levels
        db.add_rule(
            asp_rule="tactical_rule(X) :- condition(X).",
            domain="test",
            stratification_level="tactical",
        )
        db.add_rule(
            asp_rule="strategic_rule(X) :- goal(X).",
            domain="test",
            stratification_level="strategic",
        )

        # Export
        output_dir = tmp_path / "exported"
        stats = db.export_to_asp_files(str(output_dir))

        assert stats["files_created"] == 2
        assert stats["rules_exported"] == 2

        # Check files exist
        assert (output_dir / "tactical.lp").exists()
        assert (output_dir / "strategic.lp").exists()

        # Check content
        tactical_content = (output_dir / "tactical.lp").read_text()
        assert "tactical_rule(X) :- condition(X)." in tactical_content


class TestDatabaseEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test.db"
        return KnowledgeDatabase(f"sqlite:///{db_path}")

    def test_add_rule_with_all_metadata(self, db):
        """Test adding a rule with all possible metadata fields."""
        rule_id = db.add_rule(
            asp_rule="complex_rule(X) :- a(X), b(X).",
            domain="contracts",
            jurisdiction="CA",
            doctrine="offer-acceptance",
            stratification_level="tactical",
            source_type="case_analysis",
            source_cases=["case_001", "case_002"],
            generator_model="claude-3-opus",
            generator_prompt_version="v1.0",
            confidence=0.92,
            reasoning="This rule handles complex scenarios",
            tags=["validated", "high-confidence"],
            metadata={"custom": "data"},
        )

        rule = db.get_rule(rule_id)

        assert rule.domain == "contracts"
        assert rule.jurisdiction == "CA"
        assert rule.doctrine == "offer-acceptance"
        assert rule.stratification_level == "tactical"
        assert rule.rule_metadata == {"custom": "data"}

    def test_search_with_no_results(self, db):
        """Test search that returns no results."""
        results = db.search_rules(domain="nonexistent")
        assert len(results) == 0

    def test_add_rule_with_minimal_data(self, db):
        """Test adding a rule with only required fields."""
        rule_id = db.add_rule(asp_rule="simple(X).")

        rule = db.get_rule(rule_id)
        assert rule.asp_rule == "simple(X)."
        assert rule.domain is None  # Optional fields are None
