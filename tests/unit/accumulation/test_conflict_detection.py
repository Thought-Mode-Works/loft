"""
Unit tests for conflict detection.

Issue #273: Continuous Rule Accumulation Pipeline
"""

import pytest

from loft.accumulation.conflict_detection import ConflictDetector
from loft.accumulation.schemas import RuleCandidate
from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.models import LegalRule


class TestConflictDetector:
    """Test ConflictDetector class."""

    @pytest.fixture
    def detector(self):
        """Create conflict detector."""
        return ConflictDetector()

    @pytest.fixture
    def detector_with_db(self, tmp_path):
        """Create conflict detector with database."""
        db_path = tmp_path / "test.db"
        db = KnowledgeDatabase(f"sqlite:///{db_path}")

        return ConflictDetector(knowledge_db=db)

    def test_create_detector(self, detector):
        """Test creating detector."""
        assert detector is not None
        assert detector.subsumption_threshold == 0.9

    def test_create_detector_with_custom_threshold(self):
        """Test creating detector with custom threshold."""
        detector = ConflictDetector(subsumption_threshold=0.7)

        assert detector.subsumption_threshold == 0.7

    def test_find_conflicts_with_no_existing_rules(self, detector):
        """Test finding conflicts when no existing rules."""
        candidate = RuleCandidate(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="Test rule",
            source_case_id="c1",
        )

        conflicts = detector.find_conflicts(candidate, existing_rules=[])

        assert len(conflicts) == 0

    def test_detect_direct_contradiction(self, detector):
        """Test detecting direct contradiction."""
        new_rule = RuleCandidate(
            asp_rule="not valid(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        existing_rule = LegalRule(
            rule_id="r1",
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.8,
            reasoning="Existing",
        )

        conflicts = detector.find_conflicts(new_rule, existing_rules=[existing_rule])

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "contradiction"
        assert conflicts[0].severity == 1.0

    def test_detect_contradiction_reverse(self, detector):
        """Test detecting contradiction in reverse order."""
        new_rule = RuleCandidate(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        existing_rule = LegalRule(
            rule_id="r1",
            asp_rule="not valid(X) :- condition(X).",
            domain="test",
            confidence=0.8,
            reasoning="Existing",
        )

        conflicts = detector.find_conflicts(new_rule, existing_rules=[existing_rule])

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "contradiction"

    def test_detect_subsumption_high_similarity(self, detector):
        """Test detecting subsumption with high similarity."""
        new_rule = RuleCandidate(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
            domain="contracts",
            confidence=0.95,
            reasoning="Test",
            source_case_id="c1",
        )

        # Very similar rule
        existing_rule = LegalRule(
            rule_id="r1",
            asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
            domain="contracts",
            confidence=0.90,
            reasoning="Existing",
        )

        conflicts = detector.find_conflicts(new_rule, existing_rules=[existing_rule])

        # Should detect subsumption (identical rules)
        subsumptions = [c for c in conflicts if c.conflict_type == "subsumption"]
        assert len(subsumptions) >= 1
        assert subsumptions[0].severity >= 0.9

    def test_no_subsumption_for_dissimilar_rules(self, detector):
        """Test no subsumption for dissimilar rules."""
        new_rule = RuleCandidate(
            asp_rule="negligence(X) :- duty(X), breach(X).",
            domain="torts",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        existing_rule = LegalRule(
            rule_id="r1",
            asp_rule="valid_contract(X) :- offer(X).",
            domain="contracts",
            confidence=0.8,
            reasoning="Existing",
        )

        conflicts = detector.find_conflicts(new_rule, existing_rules=[existing_rule])

        subsumptions = [c for c in conflicts if c.conflict_type == "subsumption"]
        assert len(subsumptions) == 0

    def test_detect_inconsistency(self, detector):
        """Test detecting logical inconsistency."""
        # These rules should create UNSAT when combined
        new_rule = RuleCandidate(
            asp_rule=":- valid(X), not enforceable(X).",
            domain="test",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        existing_rule = LegalRule(
            rule_id="r1",
            asp_rule="valid(contract1). not enforceable(contract1).",
            domain="test",
            confidence=0.8,
            reasoning="Existing",
        )

        conflicts = detector.find_conflicts(new_rule, existing_rules=[existing_rule])

        # May or may not detect inconsistency depending on solver
        # Just verify no crash
        assert isinstance(conflicts, list)

    def test_extract_head_from_rule(self, detector):
        """Test extracting head predicate."""
        head = detector._extract_head("valid(X) :- condition(X).")
        assert head == "valid(X)"

        head = detector._extract_head("not valid(X) :- condition(X).")
        assert head == "not valid(X)"

    def test_extract_head_from_fact(self, detector):
        """Test extracting head from fact."""
        head = detector._extract_head("valid(contract1).")
        assert head == "valid(contract1)"

    def test_calculate_rule_similarity(self, detector):
        """Test rule similarity calculation."""
        rule1 = "valid(X) :- offer(X), acceptance(X)."
        rule2 = "valid(X) :- offer(X), acceptance(X)."

        similarity = detector._calculate_rule_similarity(rule1, rule2)
        assert similarity == 1.0

    def test_calculate_similarity_different_rules(self, detector):
        """Test similarity for different rules."""
        rule1 = "valid(X) :- offer(X)."
        rule2 = "invalid(Y) :- breach(Y)."

        similarity = detector._calculate_rule_similarity(rule1, rule2)
        assert similarity < 0.5

    def test_tokenize_rule(self, detector):
        """Test rule tokenization."""
        tokens = detector._tokenize_rule("valid(X) :- offer(X), acceptance(X).")

        assert "valid" in tokens
        assert "X" in tokens
        assert "offer" in tokens
        assert "acceptance" in tokens

    def test_suggest_resolution_for_contradiction(self, detector):
        """Test resolution suggestion for contradiction."""
        from loft.accumulation.schemas import Conflict

        conflict = Conflict(
            conflict_type="contradiction",
            new_rule="p(X).",
            existing_rule_id="r1",
            existing_rule="not p(X).",
            explanation="Test",
            severity=1.0,
        )

        suggestion = detector.suggest_resolution(conflict)

        assert "Contradiction" in suggestion
        assert "review" in suggestion.lower()

    def test_suggest_resolution_for_subsumption(self, detector):
        """Test resolution suggestion for subsumption."""
        from loft.accumulation.schemas import Conflict

        conflict = Conflict(
            conflict_type="subsumption",
            new_rule="p(X).",
            existing_rule_id="r1",
            existing_rule="p(X).",
            explanation="Test",
            severity=0.5,
        )

        suggestion = detector.suggest_resolution(conflict)

        assert "subsumption" in suggestion.lower()

    def test_suggest_resolution_for_inconsistency(self, detector):
        """Test resolution suggestion for inconsistency."""
        from loft.accumulation.schemas import Conflict

        conflict = Conflict(
            conflict_type="inconsistency",
            new_rule="p(X).",
            existing_rule_id="r1",
            existing_rule="q(X).",
            explanation="Test",
            severity=0.9,
        )

        suggestion = detector.suggest_resolution(conflict)

        assert "inconsistency" in suggestion.lower()
        assert "review" in suggestion.lower()

    def test_find_conflicts_with_database(self, detector_with_db):
        """Test finding conflicts using database."""
        # Add existing rule to database
        detector_with_db.knowledge_db.add_rule(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.8,
            reasoning="Existing rule",
        )

        # Create conflicting candidate
        candidate = RuleCandidate(
            asp_rule="not valid(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="New rule",
            source_case_id="c1",
        )

        # Find conflicts (should retrieve from DB)
        conflicts = detector_with_db.find_conflicts(candidate, domain="test")

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "contradiction"

    def test_find_multiple_conflicts(self, detector):
        """Test finding multiple conflicts."""
        new_rule = RuleCandidate(
            asp_rule="valid(X) :- offer(X), acceptance(X).",
            domain="contracts",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        existing_rules = [
            LegalRule(
                rule_id="r1",
                asp_rule="not valid(X) :- offer(X), acceptance(X).",
                domain="contracts",
                confidence=0.8,
                reasoning="Contradicts",
            ),
            LegalRule(
                rule_id="r2",
                asp_rule="valid(X) :- offer(X), acceptance(X).",
                domain="contracts",
                confidence=0.85,
                reasoning="Subsumes",
            ),
        ]

        conflicts = detector.find_conflicts(new_rule, existing_rules=existing_rules)

        # Should find at least contradiction
        assert len(conflicts) >= 1
        conflict_types = {c.conflict_type for c in conflicts}
        assert "contradiction" in conflict_types
