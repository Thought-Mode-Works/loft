"""
Tests for ConflictResolver class.

Tests for:
- Conflict detection
- Resolution strategies (specificity, confidence, legal default)
- Integration with ASPReasoner
- Statistics tracking
"""

from loft.symbolic.conflict_resolver import (
    ConflictResolver,
    ResolutionStrategy,
    ResolutionResult,
    ConflictStats,
)


class TestResolutionResult:
    """Test ResolutionResult dataclass."""

    def test_result_fields(self):
        """Result should contain all expected fields."""
        result = ResolutionResult(
            prediction="enforceable",
            confidence=0.85,
            method="specificity",
            conflict_detected=True,
            resolution_details={"positive_specificity": 3},
        )
        assert result.prediction == "enforceable"
        assert result.confidence == 0.85
        assert result.method == "specificity"
        assert result.conflict_detected
        assert result.resolution_details["positive_specificity"] == 3

    def test_to_dict(self):
        """Result should serialize to dictionary."""
        result = ResolutionResult(
            prediction="unenforceable",
            confidence=0.7,
            method="confidence",
            conflict_detected=True,
        )
        d = result.to_dict()
        assert d["prediction"] == "unenforceable"
        assert d["confidence"] == 0.7
        assert d["method"] == "confidence"


class TestConflictStats:
    """Test ConflictStats dataclass."""

    def test_initial_values(self):
        """Stats should initialize to zero."""
        stats = ConflictStats()
        assert stats.total_resolutions == 0
        assert stats.conflicts_detected == 0
        assert stats.resolved_by_specificity == 0

    def test_to_dict(self):
        """Stats should serialize to dictionary."""
        stats = ConflictStats(
            total_resolutions=10,
            conflicts_detected=3,
            resolved_by_specificity=2,
            resolved_by_confidence=1,
        )
        d = stats.to_dict()
        assert d["total_resolutions"] == 10
        assert d["conflict_rate"] == 0.3


class TestConflictResolverBasics:
    """Test basic ConflictResolver functionality."""

    def test_no_conflict_enforceable(self):
        """Enforceable without conflict should return directly."""
        resolver = ConflictResolver()
        derived = {"enforceable(c1)", "claim(c1)"}
        fired_rules = [("enforceable(X) :- claim(X).", 0.9)]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "enforceable"
        assert result.confidence == 0.9
        assert result.method == "no_conflict"
        assert not result.conflict_detected

    def test_no_conflict_unenforceable(self):
        """Unenforceable without conflict should return directly."""
        resolver = ConflictResolver()
        derived = {"unenforceable(c1)", "claim(c1)"}
        fired_rules = [("unenforceable(X) :- claim(X).", 0.9)]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "unenforceable"
        assert result.confidence == 0.9
        assert not result.conflict_detected

    def test_no_derivation(self):
        """No relevant atoms should return unknown."""
        resolver = ConflictResolver()
        derived = {"claim(c1)", "party(c1, alice)"}
        fired_rules = []

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "unknown"
        assert result.confidence == 0.0
        assert result.method == "no_derivation"

    def test_conflict_detected(self):
        """Both atoms should trigger conflict resolution."""
        resolver = ConflictResolver()
        derived = {"enforceable(c1)", "unenforceable(c1)", "claim(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X), b(X), c(X).", 0.9),
            ("unenforceable(X) :- a(X).", 0.8),
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.conflict_detected


class TestSpecificityResolution:
    """Test specificity-based conflict resolution."""

    def test_more_conditions_wins(self):
        """Rule with more conditions should win."""
        resolver = ConflictResolver(strategy=ResolutionStrategy.SPECIFICITY)
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X), b(X), c(X).", 0.9),  # 3 conditions
            ("unenforceable(X) :- a(X).", 0.8),  # 1 condition
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "enforceable"
        assert result.method == "specificity"
        assert result.resolution_details["positive_specificity"] == 3
        assert result.resolution_details["negative_specificity"] == 1

    def test_negative_more_specific(self):
        """More specific negative rule should win."""
        resolver = ConflictResolver(strategy=ResolutionStrategy.SPECIFICITY)
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X).", 0.9),  # 1 condition
            ("unenforceable(X) :- a(X), b(X), c(X), d(X).", 0.8),  # 4 conditions
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "unenforceable"
        assert result.method == "specificity"

    def test_specificity_tie_uses_default(self):
        """Equal specificity should fall back to legal default."""
        resolver = ConflictResolver(
            strategy=ResolutionStrategy.SPECIFICITY, legal_default="unenforceable"
        )
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X), b(X).", 0.9),  # 2 conditions
            ("unenforceable(X) :- c(X), d(X).", 0.8),  # 2 conditions
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "unenforceable"
        assert "tie" in result.method or "default" in result.method


class TestConfidenceResolution:
    """Test confidence-based conflict resolution."""

    def test_higher_confidence_wins(self):
        """Rule with higher confidence should win."""
        resolver = ConflictResolver(strategy=ResolutionStrategy.CONFIDENCE)
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X).", 0.95),
            ("unenforceable(X) :- b(X).", 0.75),
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "enforceable"
        assert result.method == "confidence"
        assert result.confidence == 0.95

    def test_negative_higher_confidence(self):
        """Higher confidence negative rule should win."""
        resolver = ConflictResolver(strategy=ResolutionStrategy.CONFIDENCE)
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X).", 0.6),
            ("unenforceable(X) :- b(X).", 0.9),
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "unenforceable"
        assert result.confidence == 0.9


class TestSpecificityThenConfidenceResolution:
    """Test combined specificity-then-confidence resolution."""

    def test_specificity_takes_priority(self):
        """Specificity should be checked first."""
        resolver = ConflictResolver(
            strategy=ResolutionStrategy.SPECIFICITY_THEN_CONFIDENCE
        )
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X), b(X), c(X).", 0.7),  # 3 conditions, lower conf
            ("unenforceable(X) :- a(X).", 0.95),  # 1 condition, higher conf
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        # Specificity wins over confidence
        assert result.prediction == "enforceable"
        assert result.method == "specificity"

    def test_confidence_breaks_specificity_tie(self):
        """Confidence should break specificity ties."""
        resolver = ConflictResolver(
            strategy=ResolutionStrategy.SPECIFICITY_THEN_CONFIDENCE
        )
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X), b(X).", 0.9),  # 2 conditions
            ("unenforceable(X) :- c(X), d(X).", 0.7),  # 2 conditions
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        # Confidence breaks the tie
        assert result.prediction == "enforceable"
        assert result.method == "confidence"


class TestLegalDefaultResolution:
    """Test legal default resolution."""

    def test_uses_legal_default(self):
        """Should use configured legal default."""
        resolver = ConflictResolver(
            strategy=ResolutionStrategy.LEGAL_DEFAULT, legal_default="unenforceable"
        )
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X).", 0.9),
            ("unenforceable(X) :- b(X).", 0.9),
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "unenforceable"
        assert result.confidence == 0.5
        assert "default" in result.method

    def test_custom_legal_default(self):
        """Custom legal default should be used."""
        resolver = ConflictResolver(
            strategy=ResolutionStrategy.LEGAL_DEFAULT, legal_default="enforceable"
        )
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X).", 0.9),
            ("unenforceable(X) :- b(X).", 0.9),
        ]

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "enforceable"


class TestConflictDetection:
    """Test conflict detection functionality."""

    def test_detect_single_conflict(self):
        """Should detect single conflicting entity."""
        resolver = ConflictResolver()
        derived = {"enforceable(c1)", "unenforceable(c1)", "claim(c1)"}

        conflicts = resolver.detect_conflicts(derived)

        assert "c1" in conflicts
        assert len(conflicts) == 1

    def test_detect_multiple_conflicts(self):
        """Should detect multiple conflicting entities."""
        resolver = ConflictResolver()
        derived = {
            "enforceable(c1)",
            "unenforceable(c1)",
            "enforceable(c2)",
            "unenforceable(c2)",
            "enforceable(c3)",  # No conflict for c3
        }

        conflicts = resolver.detect_conflicts(derived)

        assert "c1" in conflicts
        assert "c2" in conflicts
        assert "c3" not in conflicts

    def test_no_conflicts(self):
        """Should return empty list when no conflicts."""
        resolver = ConflictResolver()
        derived = {"enforceable(c1)", "unenforceable(c2)"}

        conflicts = resolver.detect_conflicts(derived)

        assert len(conflicts) == 0


class TestResolveAllConflicts:
    """Test resolving all conflicts at once."""

    def test_resolve_multiple(self):
        """Should resolve all conflicts in derived atoms."""
        resolver = ConflictResolver(strategy=ResolutionStrategy.SPECIFICITY)
        derived = {
            "enforceable(c1)",
            "unenforceable(c1)",
            "enforceable(c2)",
            "unenforceable(c2)",
        }
        fired_rules = [
            ("enforceable(X) :- a(X), b(X), c(X).", 0.9),
            ("unenforceable(X) :- a(X).", 0.8),
        ]

        results = resolver.resolve_all_conflicts(derived, fired_rules)

        assert "c1" in results
        assert "c2" in results
        assert results["c1"].prediction == "enforceable"
        assert results["c2"].prediction == "enforceable"


class TestStatisticsTracking:
    """Test statistics tracking."""

    def test_stats_track_resolutions(self):
        """Resolutions should be tracked."""
        resolver = ConflictResolver()
        derived = {"enforceable(c1)"}
        fired_rules = [("enforceable(X) :- a(X).", 0.9)]

        resolver.resolve(derived, fired_rules, "c1")

        stats = resolver.get_stats()
        assert stats.total_resolutions == 1

    def test_stats_track_conflicts(self):
        """Conflicts should be tracked."""
        resolver = ConflictResolver()
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X), b(X).", 0.9),
            ("unenforceable(X) :- a(X).", 0.8),
        ]

        resolver.resolve(derived, fired_rules, "c1")

        stats = resolver.get_stats()
        assert stats.conflicts_detected == 1

    def test_stats_track_resolution_methods(self):
        """Resolution methods should be tracked."""
        resolver = ConflictResolver(strategy=ResolutionStrategy.SPECIFICITY)
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X), b(X), c(X).", 0.9),
            ("unenforceable(X) :- a(X).", 0.8),
        ]

        resolver.resolve(derived, fired_rules, "c1")

        stats = resolver.get_stats()
        assert stats.resolved_by_specificity == 1

    def test_stats_reset(self):
        """Stats should be resettable."""
        resolver = ConflictResolver()
        derived = {"enforceable(c1)", "unenforceable(c1)"}
        fired_rules = [
            ("enforceable(X) :- a(X).", 0.9),
            ("unenforceable(X) :- b(X).", 0.8),
        ]

        resolver.resolve(derived, fired_rules, "c1")
        resolver.reset_stats()

        stats = resolver.get_stats()
        assert stats.total_resolutions == 0
        assert stats.conflicts_detected == 0


class TestConditionCounting:
    """Test condition counting in rules."""

    def test_count_simple_conditions(self):
        """Should count simple predicates."""
        resolver = ConflictResolver()
        rule = "enforceable(X) :- a(X), b(X), c(X)."

        count = resolver._count_conditions(rule)

        assert count == 3

    def test_count_complex_conditions(self):
        """Should handle complex predicates with multiple args."""
        resolver = ConflictResolver()
        rule = "enforceable(X) :- claim(X, Y), party(X, alice), years(X, N), N >= 20."

        count = resolver._count_conditions(rule)

        assert count == 4

    def test_count_no_body(self):
        """Facts should have 0 conditions."""
        resolver = ConflictResolver()
        rule = "enforceable(c1)."

        count = resolver._count_conditions(rule)

        assert count == 0

    def test_count_nested_parentheses(self):
        """Should handle nested parentheses."""
        resolver = ConflictResolver()
        rule = "enforceable(X) :- check(foo(bar(X))), other(X)."

        count = resolver._count_conditions(rule)

        assert count == 2


class TestCustomPredicates:
    """Test custom positive/negative predicates."""

    def test_custom_positive(self):
        """Should recognize custom positive predicates."""
        resolver = ConflictResolver(positive_predicates={"valid", "passes"})
        derived = {"valid(c1)"}
        fired_rules = []

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "enforceable"

    def test_custom_negative(self):
        """Should recognize custom negative predicates."""
        resolver = ConflictResolver(negative_predicates={"invalid", "fails"})
        derived = {"invalid(c1)"}
        fired_rules = []

        result = resolver.resolve(derived, fired_rules, "c1")

        assert result.prediction == "unenforceable"


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Repr should show configuration."""
        resolver = ConflictResolver(
            strategy=ResolutionStrategy.CONFIDENCE, legal_default="enforceable"
        )

        repr_str = repr(resolver)

        assert "ConflictResolver" in repr_str
        assert "confidence" in repr_str
        assert "enforceable" in repr_str
