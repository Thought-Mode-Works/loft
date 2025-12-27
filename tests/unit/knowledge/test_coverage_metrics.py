"""
Unit tests for knowledge coverage metrics.

Issue #274: Knowledge Coverage Metrics
"""

import pytest

from loft.knowledge.coverage_calculator import CoverageCalculator
from loft.knowledge.coverage_schemas import (
    CoverageGap,
    CoverageMetrics,
    DomainMetrics,
    QualityMetrics,
)
from loft.knowledge.database import KnowledgeDatabase


class TestCoverageSchemas:
    """Test coverage schema dataclasses."""

    def test_domain_metrics_coverage_score(self):
        """Test coverage score calculation."""
        domain = DomainMetrics(
            domain="contracts",
            rule_count=50,
            avg_confidence=0.9,
            accuracy=0.85,
        )

        score = domain.coverage_score
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high with good metrics

    def test_domain_metrics_coverage_score_zero_rules(self):
        """Test coverage score with no rules."""
        domain = DomainMetrics(domain="empty")

        assert domain.coverage_score == 0.0

    def test_quality_metrics_quality_score(self):
        """Test quality score calculation."""
        quality = QualityMetrics(
            total_rules=100,
            high_confidence_rules=50,
            medium_confidence_rules=30,
            low_confidence_rules=20,
            rules_with_reasoning=80,
            rules_with_sources=60,
        )

        score = quality.quality_score
        assert 0.0 <= score <= 1.0
        assert score > 0.6  # Should be good quality

    def test_coverage_metrics_overall_accuracy(self):
        """Test overall accuracy calculation."""
        metrics = CoverageMetrics()

        domain1 = DomainMetrics(
            domain="contracts",
            answered_question_count=10,
            accuracy=0.9,
        )
        domain2 = DomainMetrics(
            domain="torts",
            answered_question_count=10,
            accuracy=0.8,
        )

        metrics.domains = {
            "contracts": domain1,
            "torts": domain2,
        }
        metrics.answered_questions = 20

        accuracy = metrics.overall_accuracy
        assert accuracy is not None
        assert 0.8 <= accuracy <= 0.9  # Should be between the two

    def test_coverage_gap_string(self):
        """Test coverage gap string representation."""
        gap = CoverageGap(
            area="contracts",
            gap_type="missing_rules",
            severity=0.8,
            description="Only 5 rules",
            suggested_action="Add more rules",
        )

        gap_str = str(gap)
        assert "missing_rules" in gap_str
        assert "contracts" in gap_str


class TestCoverageCalculator:
    """Test coverage calculator."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create test database with sample data."""
        db_path = tmp_path / "test.db"
        db = KnowledgeDatabase(f"sqlite:///{db_path}")

        # Add sample rules
        db.add_rule(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X).",
            domain="contracts",
            confidence=0.95,
            reasoning="Test rule 1",
        )

        db.add_rule(
            asp_rule="enforceable(X) :- valid_contract(X).",
            domain="contracts",
            doctrine="contract-formation",
            confidence=0.85,
        )

        db.add_rule(
            asp_rule="negligence(X) :- duty(X), breach(X).",
            domain="torts",
            confidence=0.90,
            reasoning="Test rule 2",
        )

        return db

    @pytest.fixture
    def calculator(self, db):
        """Create calculator with test database."""
        return CoverageCalculator(db)

    def test_calculate_metrics(self, calculator):
        """Test metrics calculation."""
        metrics = calculator.calculate_metrics()

        assert metrics.total_rules == 3
        assert metrics.active_rules == 3
        assert metrics.domain_count == 2
        assert "contracts" in metrics.domains
        assert "torts" in metrics.domains

    def test_domain_metrics_calculation(self, calculator):
        """Test domain-specific metrics."""
        metrics = calculator.calculate_metrics()

        contracts = metrics.domains.get("contracts")
        assert contracts is not None
        assert contracts.rule_count == 2
        assert contracts.avg_confidence > 0.8

    def test_quality_metrics_calculation(self, calculator):
        """Test quality metrics."""
        metrics = calculator.calculate_metrics()

        assert metrics.quality.total_rules == 3
        assert metrics.quality.avg_confidence > 0.0
        # All rules have confidence >= 0.85
        assert metrics.quality.high_confidence_rules >= 1
        assert metrics.quality.rules_with_reasoning >= 2

    def test_identify_gaps_low_rule_count(self):
        """Test gap identification for low rule count."""
        # Create domain with few rules
        metrics = CoverageMetrics()
        metrics.domains = {"contracts": DomainMetrics(domain="contracts", rule_count=5)}

        calculator = CoverageCalculator(None)
        gaps = calculator.identify_gaps(metrics)

        # Should identify missing rules gap
        gap_types = [g.gap_type for g in gaps]
        assert "missing_rules" in gap_types

    def test_identify_gaps_low_accuracy(self):
        """Test gap identification for low accuracy."""
        metrics = CoverageMetrics()
        metrics.domains = {
            "contracts": DomainMetrics(
                domain="contracts",
                rule_count=20,
                accuracy=0.5,
            )
        }

        calculator = CoverageCalculator(None)
        gaps = calculator.identify_gaps(metrics)

        gap_types = [g.gap_type for g in gaps]
        assert "low_accuracy" in gap_types

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        metrics = CoverageMetrics()
        metrics.domains = {"contracts": DomainMetrics(domain="contracts", rule_count=5)}
        metrics.quality = QualityMetrics(total_rules=10, low_confidence_rules=5)

        calculator = CoverageCalculator(None)
        gaps = calculator.identify_gaps(metrics)
        recommendations = calculator.generate_recommendations(metrics, gaps)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_generate_report(self, calculator):
        """Test full report generation."""
        report = calculator.generate_report()

        assert report.metrics is not None
        assert isinstance(report.gaps, list)
        assert isinstance(report.recommendations, list)

    def test_coverage_report_to_markdown(self, calculator):
        """Test report markdown conversion."""
        report = calculator.generate_report()

        markdown = report.to_markdown()

        assert "# Knowledge Coverage Report" in markdown
        assert "Overall Metrics" in markdown
        assert isinstance(markdown, str)
        assert len(markdown) > 100


class TestMetricsTrend:
    """Test metrics trend tracking."""

    def test_trend_add_sample(self):
        """Test adding samples to trend."""
        from datetime import datetime

        from loft.knowledge.coverage_schemas import MetricsTrend

        trend = MetricsTrend(metric_name="total_rules")

        trend.add_sample(datetime.utcnow(), 10.0)
        trend.add_sample(datetime.utcnow(), 15.0)

        assert len(trend.values) == 2
        assert trend.latest_value == 15.0

    def test_trend_direction_increasing(self):
        """Test trend direction for increasing values."""
        from datetime import datetime

        from loft.knowledge.coverage_schemas import MetricsTrend

        trend = MetricsTrend(metric_name="total_rules")

        for i in range(10):
            trend.add_sample(datetime.utcnow(), float(i * 10))

        assert trend.trend_direction == "increasing"

    def test_trend_direction_decreasing(self):
        """Test trend direction for decreasing values."""
        from datetime import datetime

        from loft.knowledge.coverage_schemas import MetricsTrend

        trend = MetricsTrend(metric_name="accuracy")

        for i in range(10):
            trend.add_sample(datetime.utcnow(), float(100 - i * 10))

        assert trend.trend_direction == "decreasing"

    def test_trend_direction_stable(self):
        """Test trend direction for stable values."""
        from datetime import datetime

        from loft.knowledge.coverage_schemas import MetricsTrend

        trend = MetricsTrend(metric_name="confidence")

        for _ in range(10):
            trend.add_sample(datetime.utcnow(), 50.0)

        assert trend.trend_direction == "stable"
