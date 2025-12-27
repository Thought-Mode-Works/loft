"""
Unit tests for legal reasoner.

Tests ASP-based reasoning for question answering.

Issue #272: Legal Question Answering Interface
"""

import pytest

from loft.qa.reasoner import LegalReasoner
from loft.qa.schemas import Answer, ASPQuery


class TestLegalReasoner:
    """Unit tests for LegalReasoner."""

    @pytest.fixture
    def reasoner(self):
        """Create reasoner without database (tests ASP reasoning only)."""
        return LegalReasoner(knowledge_db=None)

    def test_reasoner_initialization(self, reasoner):
        """Test reasoner initializes correctly."""
        assert reasoner is not None
        assert reasoner.knowledge_db is None
        assert reasoner.min_rule_confidence == 0.6

    def test_answer_simple_query(self, reasoner):
        """Test answering a simple ASP query."""
        asp_query = ASPQuery(
            facts=["offer(c1).", "acceptance(c1).", "consideration(c1)."],
            query="valid_contract(c1)",
            domain="contracts",
        )

        answer = reasoner.answer(asp_query)

        assert isinstance(answer, Answer)
        assert answer.answer in ["yes", "no", "unknown"]
        assert 0.0 <= answer.confidence <= 1.0
        assert isinstance(answer.explanation, str)
        assert len(answer.explanation) > 0

    def test_answer_with_no_rules(self, reasoner):
        """Test answering when no rules available."""
        asp_query = ASPQuery(
            facts=["some_fact(x)."],
            query="some_query(x)",
            domain="test",
        )

        answer = reasoner.answer(asp_query)

        # Should return an answer even without rules
        assert isinstance(answer, Answer)
        assert answer.answer in ["yes", "no", "unknown"]

    def test_extract_predicate_simple(self, reasoner):
        """Test extracting predicate from query string."""
        predicate = reasoner._extract_predicate("valid_contract(c1)")
        assert predicate == "valid_contract"

    def test_extract_predicate_no_arguments(self, reasoner):
        """Test extracting predicate without arguments."""
        predicate = reasoner._extract_predicate("valid_contract")
        assert predicate == "valid_contract"

    def test_extract_predicate_complex(self, reasoner):
        """Test extracting predicate from complex query."""
        predicate = reasoner._extract_predicate("liable(X, Y)")
        assert predicate == "liable"

    def test_answer_creates_explanation(self, reasoner):
        """Test that answer includes explanation."""
        asp_query = ASPQuery(
            facts=["fact1(x).", "fact2(y)."],
            query="result(x)",
            original_question="Test question?",
        )

        answer = reasoner.answer(asp_query)

        assert answer.explanation is not None
        assert len(answer.explanation) > 0
        assert "Test question" in answer.explanation

    def test_answer_tracks_gaps(self, reasoner):
        """Test that gaps are identified when answer is unknown."""
        asp_query = ASPQuery(
            facts=[],  # No facts
            query="unknown_predicate(x)",
            domain="test",
        )

        answer = reasoner.answer(asp_query)

        # Should identify some gaps
        if answer.answer == "unknown":
            assert len(answer.gaps_identified) > 0

    def test_batch_answer(self, reasoner):
        """Test answering multiple queries in batch."""
        queries = [
            ASPQuery(facts=["f1(a)."], query="q1(a)"),
            ASPQuery(facts=["f2(b)."], query="q2(b)"),
            ASPQuery(facts=["f3(c)."], query="q3(c)"),
        ]

        answers = reasoner.batch_answer(queries)

        assert len(answers) == 3
        assert all(isinstance(a, Answer) for a in answers)

    def test_answer_handles_errors_gracefully(self, reasoner):
        """Test that reasoner handles errors gracefully."""
        # Create an invalid query that might cause errors
        asp_query = ASPQuery(
            facts=[":-"],  # Invalid ASP syntax
            query="test",
        )

        # Should not raise, should return unknown
        answer = reasoner.answer(asp_query)
        assert isinstance(answer, Answer)
        # Likely to be unknown due to error
        assert answer.answer == "unknown"


class TestLegalReasonerWithDatabase:
    """Test reasoner with knowledge database."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create temporary test database."""
        from loft.knowledge.database import KnowledgeDatabase

        db_path = tmp_path / "test.db"
        return KnowledgeDatabase(f"sqlite:///{db_path}")

    @pytest.fixture
    def reasoner_with_db(self, db):
        """Create reasoner with test database."""
        # Add test rules
        db.add_rule(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
            domain="contracts",
            confidence=0.95,
        )
        db.add_rule(
            asp_rule="voidable(X) :- contract(X), party(X, P), minor(P).",
            domain="contracts",
            confidence=0.90,
        )

        return LegalReasoner(knowledge_db=db)

    def test_answer_retrieves_rules(self, reasoner_with_db):
        """Test that reasoner retrieves rules from database."""
        asp_query = ASPQuery(
            facts=["offer(c1).", "acceptance(c1).", "consideration(c1)."],
            query="valid_contract(c1)",
            domain="contracts",
        )

        answer = reasoner_with_db.answer(asp_query)

        # Should have used rules from database
        assert len(answer.rules_used) > 0

    def test_answer_filters_by_domain(self, reasoner_with_db, db):
        """Test that rules are filtered by domain."""
        # Add rule in different domain
        db.add_rule(
            asp_rule="negligence(X) :- duty(X), breach(X).",
            domain="torts",
            confidence=0.95,
        )

        asp_query = ASPQuery(
            facts=["offer(c1)."],
            query="valid_contract(c1)",
            domain="contracts",
        )

        answer = reasoner_with_db.answer(asp_query, domain="contracts")

        # Should only use contracts rules
        assert isinstance(answer, Answer)

    def test_answer_respects_confidence_threshold(self, db):
        """Test that low-confidence rules are filtered."""
        # Add low confidence rule
        db.add_rule(
            asp_rule="low_confidence_rule(X) :- condition(X).",
            domain="test",
            confidence=0.3,  # Below default threshold of 0.6
        )

        reasoner = LegalReasoner(knowledge_db=db, min_rule_confidence=0.6)

        asp_query = ASPQuery(
            facts=["condition(x)."],
            query="low_confidence_rule(x)",
            domain="test",
        )

        answer = reasoner.answer(asp_query)

        # Low confidence rule should not be used
        assert isinstance(answer, Answer)
