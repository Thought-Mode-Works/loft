"""
Integration tests for knowledge database.

Tests complete workflows including migration and multi-component integration.

Issue #271: Persistent Legal Knowledge Database
"""

import pytest

from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.migration import ASPFileMigrator


class TestDatabaseIntegration:
    """Integration tests for full database workflows."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test.db"
        return KnowledgeDatabase(f"sqlite:///{db_path}")

    @pytest.fixture
    def sample_asp_files(self, tmp_path):
        """Create sample ASP files for testing migration."""
        asp_dir = tmp_path / "asp_rules"
        asp_dir.mkdir()

        # Create tactical.lp
        tactical_file = asp_dir / "tactical.lp"
        tactical_file.write_text(
            """% TACTICAL Layer Rules
% Generated: 2024-01-01

% Contract validity requires three elements
valid_contract(X) :- offer(X), acceptance(X), consideration(X).

% Minors can void contracts
voidable_by_minor(X) :- contract_party(X, P), minor(P).
"""
        )

        # Create contracts subdirectory
        contracts_dir = asp_dir / "contracts"
        contracts_dir.mkdir()

        contract_file = contracts_dir / "offer_acceptance.lp"
        contract_file.write_text(
            """% Offer and acceptance rules
% Confidence: 0.95

offer_valid(X) :- offer(X), definite_terms(X).

acceptance_valid(X) :- acceptance(X), mirror_image(X).
"""
        )

        return asp_dir

    def test_migration_from_asp_files(self, db, sample_asp_files):
        """Test complete migration workflow from ASP files."""
        migrator = ASPFileMigrator(db)

        result = migrator.migrate_directory(
            str(sample_asp_files),
            default_domain="contracts",
        )

        # Check migration statistics
        assert result.files_processed == 2
        assert result.rules_imported >= 4  # At least 4 rules in test files
        assert result.errors == 0

        # Verify rules were imported
        rules = db.search_rules(limit=100)
        assert len(rules) >= 4

        # Check tactical rule was imported correctly
        tactical_rules = db.search_rules(stratification_level="tactical")
        assert len(tactical_rules) >= 2

    def test_round_trip_export_import(self, db, tmp_path):
        """Test exporting and re-importing rules."""
        # Add rules to database
        db.add_rule(
            asp_rule="test_rule_1(X) :- condition(X).",
            domain="test",
            stratification_level="tactical",
            confidence=0.9,
        )
        db.add_rule(
            asp_rule="test_rule_2(X) :- other(X).",
            domain="test",
            stratification_level="strategic",
            confidence=0.85,
        )

        # Export to files
        export_dir = tmp_path / "exported"
        export_stats = db.export_to_asp_files(str(export_dir))

        assert export_stats["rules_exported"] == 2

        # Create new database
        db2_path = tmp_path / "test2.db"
        db2 = KnowledgeDatabase(f"sqlite:///{db2_path}")

        # Import from exported files
        migrator = ASPFileMigrator(db2)
        import_result = migrator.migrate_directory(str(export_dir))

        assert import_result.rules_imported == 2

        # Verify rules match
        original_rules = sorted([r.asp_rule for r in db.search_rules(limit=100)])
        imported_rules = sorted([r.asp_rule for r in db2.search_rules(limit=100)])

        assert original_rules == imported_rules

    def test_question_answering_workflow(self, db):
        """Test complete question answering workflow."""
        # 1. Add rules
        rule1_id = db.add_rule(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X).",
            domain="contracts",
            confidence=0.95,
        )

        rule2_id = db.add_rule(
            asp_rule="enforceable(X) :- valid_contract(X), consideration(X).",
            domain="contracts",
            confidence=0.90,
        )

        # 2. Record a question
        _question_id = db.add_question(
            question_text="Is the contract enforceable?",
            asp_query="?- enforceable(c1).",
            answer="yes",
            reasoning="Contract has offer, acceptance, and consideration",
            rules_used=[rule1_id, rule2_id],
            confidence=0.92,
            correct=True,
            domain="contracts",
        )

        # 3. Update rule performance
        db.update_rule_performance(rule1_id, success=True)
        db.update_rule_performance(rule2_id, success=True)

        # 4. Check coverage stats
        stats = db.get_coverage_stats("contracts")

        assert stats.rule_count == 2
        assert stats.question_count == 1
        assert stats.accuracy == 1.0

        # 5. Verify rules were marked as used
        rule1 = db.get_rule(rule1_id)
        assert rule1.validation_count == 1
        assert rule1.success_count == 1

    def test_performance_degradation_detection(self, db):
        """Test tracking rule performance over time."""
        rule_id = db.add_rule(
            asp_rule="test_rule(X) :- condition(X).",
            domain="test",
            confidence=0.95,
        )

        # Simulate multiple uses with varying success
        for i in range(10):
            success = i < 7  # 70% success rate
            db.update_rule_performance(rule_id, success=success)

        rule = db.get_rule(rule_id)

        assert rule.validation_count == 10
        assert rule.success_count == 7
        assert rule.failure_count == 3

        # Success rate is 70%
        success_rate = rule.success_count / rule.validation_count
        assert success_rate == 0.7

        # This could trigger refinement in a real system

    def test_concurrent_rule_additions(self, db):
        """Test adding rules in rapid succession."""
        rule_ids = []

        for i in range(50):
            rule_id = db.add_rule(
                asp_rule=f"rule_{i}(X) :- condition_{i}(X).",
                domain=f"domain_{i % 5}",  # 5 different domains
                confidence=0.8 + (i % 20) / 100,  # Varying confidence
            )
            rule_ids.append(rule_id)

        # Verify all rules were added
        assert len(rule_ids) == 50
        assert len(set(rule_ids)) == 50  # All unique

        # Check database stats
        stats = db.get_database_stats()
        assert stats.total_rules == 50
        assert len(stats.domains) == 5

    def test_coverage_statistics_accuracy(self, db):
        """Test accuracy of coverage statistics calculations."""
        # Add rules
        db.add_rule(asp_rule="r1(X).", domain="test", confidence=0.9)
        db.add_rule(asp_rule="r2(X).", domain="test", confidence=0.95)
        db.add_rule(asp_rule="r3(X).", domain="test", confidence=0.85)

        # Add questions with varying correctness
        db.add_question(question_text="Q1", domain="test", correct=True)
        db.add_question(question_text="Q2", domain="test", correct=True)
        db.add_question(question_text="Q3", domain="test", correct=False)
        db.add_question(question_text="Q4", domain="test", correct=True)

        # Get coverage stats
        stats = db.get_coverage_stats("test")

        assert stats.rule_count == 3
        assert stats.question_count == 4
        assert stats.accuracy == 0.75  # 3/4 correct
        assert stats.avg_confidence == 0.9  # (0.9 + 0.95 + 0.85) / 3


class TestMigrationEdgeCases:
    """Test edge cases in ASP file migration."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test.db"
        return KnowledgeDatabase(f"sqlite:///{db_path}")

    def test_migrate_file_with_multi_line_rules(self, db, tmp_path):
        """Test migrating ASP file with multi-line rules."""
        asp_file = tmp_path / "multiline.lp"
        asp_file.write_text(
            """% Complex multi-line rule
complex_rule(X) :-
    condition_a(X),
    condition_b(X),
    condition_c(X).

% Another rule
simple_rule(Y) :- fact(Y).
"""
        )

        migrator = ASPFileMigrator(db)
        result = migrator.migrate_file(asp_file)

        assert result["rules_imported"] == 2

        rules = db.search_rules(limit=10)
        assert len(rules) == 2

    def test_migrate_empty_file(self, db, tmp_path):
        """Test migrating an empty ASP file."""
        asp_file = tmp_path / "empty.lp"
        asp_file.write_text("% Just comments\n% No actual rules\n")

        migrator = ASPFileMigrator(db)
        result = migrator.migrate_file(asp_file)

        assert result["rules_imported"] == 0
        assert result["rules_skipped"] == 0

    def test_migrate_nonexistent_directory_raises_error(self, db):
        """Test that migrating nonexistent directory raises error."""
        migrator = ASPFileMigrator(db)

        with pytest.raises(ValueError, match="does not exist"):
            migrator.migrate_directory("/nonexistent/path")


class TestDatabasePersistence:
    """Test database persistence across sessions."""

    def test_data_persists_across_connections(self, tmp_path):
        """Test that data persists when closing and reopening database."""
        db_path = tmp_path / "persist.db"

        # Session 1: Add data
        db1 = KnowledgeDatabase(f"sqlite:///{db_path}")
        rule_id = db1.add_rule(
            asp_rule="persist_test(X).",
            domain="test",
            confidence=0.95,
        )
        db1.close()

        # Session 2: Retrieve data
        db2 = KnowledgeDatabase(f"sqlite:///{db_path}")
        rule = db2.get_rule(rule_id)

        assert rule is not None
        assert rule.asp_rule == "persist_test(X)."
        assert rule.confidence == 0.95

        db2.close()
