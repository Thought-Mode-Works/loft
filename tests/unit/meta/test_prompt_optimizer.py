"""
Unit tests for the prompt optimizer module.

Tests cover:
- PromptVersion creation and management
- PromptMetrics tracking
- EffectivenessReport generation
- PromptOptimizer functionality
- PromptABTester A/B testing
- Factory functions
"""

import pytest
from datetime import datetime

from loft.meta.prompt_optimizer import (
    ABTestConfig,
    ABTestResult,
    EffectivenessReport,
    ImprovementCandidate,
    PromptABTester,
    PromptCategory,
    PromptMetrics,
    PromptOptimizer,
    PromptVersion,
    TestStatus,
    create_ab_tester,
    create_prompt_optimizer,
)


class TestPromptCategory:
    """Tests for PromptCategory enum."""

    def test_all_categories_exist(self):
        """Test all prompt categories exist."""
        assert PromptCategory.RULE_GENERATION.value == "rule_generation"
        assert PromptCategory.VALIDATION.value == "validation"
        assert PromptCategory.TRANSLATION.value == "translation"
        assert PromptCategory.CRITIC.value == "critic"
        assert PromptCategory.SYNTHESIS.value == "synthesis"
        assert PromptCategory.GENERAL.value == "general"


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_all_statuses_exist(self):
        """Test all test statuses exist."""
        assert TestStatus.PENDING.value == "pending"
        assert TestStatus.RUNNING.value == "running"
        assert TestStatus.COMPLETED.value == "completed"
        assert TestStatus.CANCELLED.value == "cancelled"


class TestPromptVersion:
    """Tests for PromptVersion dataclass."""

    def test_basic_creation(self):
        """Test basic prompt version creation."""
        prompt = PromptVersion(
            prompt_id="test_prompt",
            version=1,
            template="Convert this: {input}",
        )
        assert prompt.prompt_id == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Convert this: {input}"
        assert prompt.category == PromptCategory.GENERAL

    def test_variable_extraction(self):
        """Test automatic variable extraction."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Process {input} with {context} and {options}",
        )
        assert set(prompt.variables) == {"input", "context", "options"}

    def test_full_id_property(self):
        """Test full_id property."""
        prompt = PromptVersion(
            prompt_id="my_prompt",
            version=3,
            template="test",
        )
        assert prompt.full_id == "my_prompt_v3"

    def test_template_hash(self):
        """Test template hash generation."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="test template",
        )
        assert len(prompt.template_hash) == 8

    def test_format_method(self):
        """Test template formatting."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Hello {name}, your score is {score}",
        )
        result = prompt.format(name="Alice", score=95)
        assert result == "Hello Alice, your score is 95"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="test: {input}",
            category=PromptCategory.VALIDATION,
            modification_reason="Initial version",
        )
        data = prompt.to_dict()
        assert data["prompt_id"] == "test"
        assert data["version"] == 1
        assert data["category"] == "validation"
        assert data["modification_reason"] == "Initial version"
        assert "input" in data["variables"]


class TestPromptMetrics:
    """Tests for PromptMetrics dataclass."""

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        metrics = PromptMetrics()
        assert metrics.total_uses == 0
        assert metrics.successful_uses == 0
        assert metrics.success_rate == 0.0

    def test_record_success(self):
        """Test recording successful use."""
        metrics = PromptMetrics()
        metrics.record(success=True, confidence=0.9, latency_ms=100.0)

        assert metrics.total_uses == 1
        assert metrics.successful_uses == 1
        assert metrics.success_rate == 1.0
        assert metrics.average_confidence == 0.9
        assert metrics.average_latency_ms == 100.0

    def test_record_failure(self):
        """Test recording failed use."""
        metrics = PromptMetrics()
        metrics.record(success=False, confidence=0.3)

        assert metrics.total_uses == 1
        assert metrics.failed_uses == 1
        assert metrics.success_rate == 0.0

    def test_multiple_records(self):
        """Test multiple recordings."""
        metrics = PromptMetrics()

        for i in range(8):
            metrics.record(success=True, confidence=0.8)
        for i in range(2):
            metrics.record(success=False, confidence=0.2)

        assert metrics.total_uses == 10
        assert metrics.successful_uses == 8
        assert metrics.success_rate == 0.8

    def test_domain_performance(self):
        """Test domain-specific performance tracking."""
        metrics = PromptMetrics()

        metrics.record(success=True, domain="contracts")
        metrics.record(success=True, domain="contracts")
        metrics.record(success=False, domain="contracts")
        metrics.record(success=True, domain="torts")

        assert metrics.get_domain_success_rate("contracts") == pytest.approx(
            2 / 3, rel=0.01
        )
        assert metrics.get_domain_success_rate("torts") == 1.0
        assert metrics.get_domain_success_rate("unknown") == 0.0

    def test_syntax_validity_rate(self):
        """Test syntax validity tracking."""
        metrics = PromptMetrics()

        metrics.record(success=True, syntax_valid=True)
        metrics.record(success=True, syntax_valid=True)
        metrics.record(success=True, syntax_valid=False)

        assert metrics.syntax_validity_rate == pytest.approx(2 / 3, rel=0.01)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PromptMetrics()
        metrics.record(success=True, confidence=0.9, latency_ms=50.0, cost_usd=0.01)

        data = metrics.to_dict()
        assert data["total_uses"] == 1
        assert data["success_rate"] == 1.0
        assert data["average_confidence"] == 0.9
        assert data["total_cost_usd"] == 0.01


class TestEffectivenessReport:
    """Tests for EffectivenessReport dataclass."""

    def test_report_creation(self):
        """Test effectiveness report creation."""
        report = EffectivenessReport(
            prompt_id="test_v1",
            analysis_period_start=datetime(2024, 1, 1),
            analysis_period_end=datetime(2024, 1, 31),
            total_samples=100,
            overall_effectiveness=0.85,
            confidence_interval=(0.78, 0.92),
            trend="improving",
            domain_breakdown={"contracts": 0.9, "torts": 0.8},
            weakness_areas=["Low performance in property domain"],
            strength_areas=["High performance in contracts"],
            recommendations=["Consider testing variations"],
        )

        assert report.prompt_id == "test_v1"
        assert report.overall_effectiveness == 0.85
        assert report.trend == "improving"
        assert len(report.recommendations) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = EffectivenessReport(
            prompt_id="test_v1",
            analysis_period_start=datetime(2024, 1, 1),
            analysis_period_end=datetime(2024, 1, 31),
            total_samples=50,
            overall_effectiveness=0.75,
            confidence_interval=(0.65, 0.85),
            trend="stable",
            domain_breakdown={},
            weakness_areas=[],
            strength_areas=[],
            recommendations=[],
        )

        data = report.to_dict()
        assert data["prompt_id"] == "test_v1"
        assert data["total_samples"] == 50
        assert data["trend"] == "stable"


class TestImprovementCandidate:
    """Tests for ImprovementCandidate dataclass."""

    def test_candidate_creation(self):
        """Test improvement candidate creation."""
        candidate = ImprovementCandidate(
            candidate_id="cand_001",
            original_prompt_id="test_v1",
            new_template="Improved: {input}",
            improvement_type="clarification",
            expected_improvement=0.10,
            rationale="Added clearer instructions",
        )

        assert candidate.candidate_id == "cand_001"
        assert candidate.improvement_type == "clarification"
        assert candidate.expected_improvement == 0.10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        candidate = ImprovementCandidate(
            candidate_id="cand_001",
            original_prompt_id="test_v1",
            new_template="New template",
            improvement_type="structure",
            expected_improvement=0.08,
            rationale="Better structure",
        )

        data = candidate.to_dict()
        assert data["improvement_type"] == "structure"
        assert data["expected_improvement"] == 0.08


class TestABTestConfig:
    """Tests for ABTestConfig dataclass."""

    def test_config_creation(self):
        """Test A/B test config creation."""
        prompt_a = PromptVersion(prompt_id="test", version=1, template="A: {input}")
        prompt_b = PromptVersion(prompt_id="test", version=2, template="B: {input}")

        config = ABTestConfig(
            test_id="test_001",
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            min_samples_per_variant=50,
        )

        assert config.test_id == "test_001"
        assert config.min_samples_per_variant == 50
        assert config.significance_level == 0.05
        assert config.allocation_ratio == 0.5


class TestABTestResult:
    """Tests for ABTestResult dataclass."""

    def test_result_creation(self):
        """Test A/B test result creation."""
        metrics_a = PromptMetrics()
        metrics_b = PromptMetrics()

        result = ABTestResult(
            test_id="test_001",
            prompt_a_id="test_v1",
            prompt_b_id="test_v2",
            prompt_a_metrics=metrics_a,
            prompt_b_metrics=metrics_b,
            winner="test_v2",
            p_value=0.03,
            effect_size=0.25,
            confidence_interval=(-0.05, 0.15),
            is_significant=True,
            recommendation="Adopt variant B",
        )

        assert result.winner == "test_v2"
        assert result.is_significant is True
        assert result.p_value == 0.03

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics_a = PromptMetrics()
        metrics_b = PromptMetrics()

        result = ABTestResult(
            test_id="test_001",
            prompt_a_id="test_v1",
            prompt_b_id="test_v2",
            prompt_a_metrics=metrics_a,
            prompt_b_metrics=metrics_b,
            winner=None,
            p_value=0.15,
            effect_size=0.05,
            confidence_interval=(-0.1, 0.1),
            is_significant=False,
            recommendation="No significant difference",
        )

        data = result.to_dict()
        assert data["is_significant"] is False
        assert data["winner"] is None


class TestPromptOptimizer:
    """Tests for PromptOptimizer class."""

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer = PromptOptimizer()
        assert optimizer.improvement_threshold == 0.15
        assert optimizer.min_samples_for_analysis == 10

    def test_register_prompt(self):
        """Test prompt registration."""
        optimizer = PromptOptimizer()
        prompt = PromptVersion(prompt_id="test", version=1, template="Test: {input}")

        optimizer.register_prompt(prompt)

        assert optimizer.get_prompt("test_v1") is not None
        assert optimizer.get_metrics("test_v1") is not None

    def test_track_performance(self):
        """Test performance tracking."""
        optimizer = PromptOptimizer()
        prompt = PromptVersion(prompt_id="test", version=1, template="Test: {input}")
        optimizer.register_prompt(prompt)

        optimizer.track_prompt_performance("test_v1", success=True, confidence=0.9)
        optimizer.track_prompt_performance("test_v1", success=True, confidence=0.85)
        optimizer.track_prompt_performance("test_v1", success=False, confidence=0.3)

        metrics = optimizer.get_metrics("test_v1")
        assert metrics.total_uses == 3
        assert metrics.successful_uses == 2

    def test_track_performance_with_domain(self):
        """Test performance tracking with domain."""
        optimizer = PromptOptimizer()
        prompt = PromptVersion(prompt_id="test", version=1, template="Test")
        optimizer.register_prompt(prompt)

        optimizer.track_prompt_performance("test_v1", success=True, domain="contracts")
        optimizer.track_prompt_performance("test_v1", success=False, domain="torts")

        metrics = optimizer.get_metrics("test_v1")
        assert metrics.get_domain_success_rate("contracts") == 1.0
        assert metrics.get_domain_success_rate("torts") == 0.0

    def test_analyze_prompt_effectiveness(self):
        """Test effectiveness analysis."""
        optimizer = PromptOptimizer(min_samples_for_analysis=5)
        prompt = PromptVersion(prompt_id="test", version=1, template="Test")
        optimizer.register_prompt(prompt)

        # Add enough samples
        for i in range(8):
            optimizer.track_prompt_performance(
                "test_v1", success=True, confidence=0.9, domain="contracts"
            )
        for i in range(2):
            optimizer.track_prompt_performance(
                "test_v1", success=False, confidence=0.3, domain="torts"
            )

        report = optimizer.analyze_prompt_effectiveness("test_v1")

        assert report.total_samples == 10
        assert report.overall_effectiveness == 0.8
        assert "contracts" in report.domain_breakdown

    def test_analyze_insufficient_data(self):
        """Test analysis with insufficient data."""
        optimizer = PromptOptimizer(min_samples_for_analysis=10)
        prompt = PromptVersion(prompt_id="test", version=1, template="Test")
        optimizer.register_prompt(prompt)

        optimizer.track_prompt_performance("test_v1", success=True)

        with pytest.raises(ValueError, match="Insufficient data"):
            optimizer.analyze_prompt_effectiveness("test_v1")

    def test_generate_improvement_candidates(self):
        """Test improvement candidate generation."""
        optimizer = PromptOptimizer()
        prompt = PromptVersion(
            prompt_id="test", version=1, template="Process this: {input}"
        )
        optimizer.register_prompt(prompt)

        candidates = optimizer.generate_improvement_candidates(
            "test_v1", num_candidates=3
        )

        assert len(candidates) == 3
        assert all(isinstance(c, ImprovementCandidate) for c in candidates)

        # Check different improvement types
        improvement_types = {c.improvement_type for c in candidates}
        assert len(improvement_types) >= 2

    def test_generate_candidates_unknown_prompt(self):
        """Test candidate generation for unknown prompt."""
        optimizer = PromptOptimizer()

        with pytest.raises(ValueError, match="not found"):
            optimizer.generate_improvement_candidates("unknown_v1")

    def test_identify_underperforming(self):
        """Test identification of underperforming prompts."""
        optimizer = PromptOptimizer(min_samples_for_analysis=5)

        # Good prompt
        good_prompt = PromptVersion(prompt_id="good", version=1, template="Good")
        optimizer.register_prompt(good_prompt)
        for _ in range(10):
            optimizer.track_prompt_performance("good_v1", success=True)

        # Bad prompt
        bad_prompt = PromptVersion(prompt_id="bad", version=1, template="Bad")
        optimizer.register_prompt(bad_prompt)
        for _ in range(10):
            optimizer.track_prompt_performance("bad_v1", success=False)

        underperformers = optimizer.identify_underperforming_prompts(threshold=0.7)

        assert len(underperformers) == 1
        assert underperformers[0][0] == "bad_v1"

    def test_version_history(self):
        """Test version history tracking."""
        optimizer = PromptOptimizer()

        v1 = PromptVersion(prompt_id="test", version=1, template="V1")
        v2 = PromptVersion(prompt_id="test", version=2, template="V2")
        v3 = PromptVersion(prompt_id="test", version=3, template="V3")

        optimizer.register_prompt(v1)
        optimizer.register_prompt(v2)
        optimizer.register_prompt(v3)

        history = optimizer.get_version_history("test")
        assert len(history) == 3

    def test_create_new_version(self):
        """Test creating new prompt version."""
        optimizer = PromptOptimizer()
        prompt = PromptVersion(prompt_id="test", version=1, template="Original")
        optimizer.register_prompt(prompt)

        new_version = optimizer.create_new_version(
            original_prompt_id="test_v1",
            new_template="Improved template",
            modification_reason="Added clarity",
        )

        assert new_version.version == 2
        assert new_version.parent_version == "test_v1"
        assert new_version.modification_reason == "Added clarity"


class TestPromptABTester:
    """Tests for PromptABTester class."""

    def test_tester_creation(self):
        """Test A/B tester creation."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer)

        assert tester.min_samples_per_variant == 30
        assert tester.significance_level == 0.05

    def test_create_test(self):
        """Test creating an A/B test."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer)

        prompt_a = PromptVersion(prompt_id="test", version=1, template="A")
        prompt_b = PromptVersion(prompt_id="test", version=2, template="B")

        config = tester.create_test(prompt_a, prompt_b)

        assert config.test_id.startswith("test_")
        assert config.prompt_a == prompt_a
        assert config.prompt_b == prompt_b

    def test_get_variant(self):
        """Test getting variant for allocation."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer)

        prompt_a = PromptVersion(prompt_id="test", version=1, template="A")
        prompt_b = PromptVersion(prompt_id="test", version=2, template="B")

        config = tester.create_test(prompt_a, prompt_b, allocation_ratio=0.5)

        # Get many variants and check distribution
        variants = [tester.get_variant(config.test_id) for _ in range(100)]
        a_count = sum(1 for v in variants if v.version == 1)
        b_count = sum(1 for v in variants if v.version == 2)

        # Should be roughly 50/50
        assert 30 < a_count < 70
        assert 30 < b_count < 70

    def test_get_variant_unknown_test(self):
        """Test getting variant for unknown test."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer)

        with pytest.raises(ValueError, match="not found"):
            tester.get_variant("unknown_test")

    def test_record_outcome(self):
        """Test recording test outcomes."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer)

        prompt_a = PromptVersion(prompt_id="test", version=1, template="A")
        prompt_b = PromptVersion(prompt_id="test", version=2, template="B")

        config = tester.create_test(prompt_a, prompt_b)

        tester.record_outcome(config.test_id, "test_v1", success=True)
        tester.record_outcome(config.test_id, "test_v2", success=False)

        metrics_a = optimizer.get_metrics("test_v1")
        metrics_b = optimizer.get_metrics("test_v2")

        assert metrics_a.successful_uses == 1
        assert metrics_b.failed_uses == 1

    def test_check_completion(self):
        """Test checking test completion."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer, min_samples_per_variant=5)

        prompt_a = PromptVersion(prompt_id="test", version=1, template="A")
        prompt_b = PromptVersion(prompt_id="test", version=2, template="B")

        config = tester.create_test(prompt_a, prompt_b)

        # Not complete yet
        assert tester.check_test_completion(config.test_id) is False

        # Add samples
        for _ in range(5):
            tester.record_outcome(config.test_id, "test_v1", success=True)
            tester.record_outcome(config.test_id, "test_v2", success=True)

        # Now complete
        assert tester.check_test_completion(config.test_id) is True

    def test_analyze_test(self):
        """Test analyzing completed test."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer, min_samples_per_variant=10)

        prompt_a = PromptVersion(prompt_id="test", version=1, template="A")
        prompt_b = PromptVersion(prompt_id="test", version=2, template="B")

        config = tester.create_test(prompt_a, prompt_b)

        # Variant A: 60% success
        for i in range(10):
            tester.record_outcome(
                config.test_id, "test_v1", success=(i < 6), confidence=0.7
            )

        # Variant B: 90% success
        for i in range(10):
            tester.record_outcome(
                config.test_id, "test_v2", success=(i < 9), confidence=0.9
            )

        result = tester.analyze_test(config.test_id)

        assert result.prompt_a_metrics.success_rate == 0.6
        assert result.prompt_b_metrics.success_rate == 0.9
        assert result.effect_size > 0

    def test_select_winner(self):
        """Test selecting test winner."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer, min_samples_per_variant=20)

        prompt_a = PromptVersion(prompt_id="test", version=1, template="A")
        prompt_b = PromptVersion(prompt_id="test", version=2, template="B")

        config = tester.create_test(prompt_a, prompt_b)

        # Variant A: 50% success
        for i in range(30):
            tester.record_outcome(
                config.test_id, "test_v1", success=(i < 15), confidence=0.5
            )

        # Variant B: 90% success
        for i in range(30):
            tester.record_outcome(
                config.test_id, "test_v2", success=(i < 27), confidence=0.9
            )

        winner = tester.select_winner(config.test_id)

        # B should win with high confidence
        assert winner is not None
        assert winner.version == 2

    def test_get_active_tests(self):
        """Test getting list of active tests."""
        optimizer = PromptOptimizer()
        tester = PromptABTester(optimizer)

        prompt_a = PromptVersion(prompt_id="test1", version=1, template="A")
        prompt_b = PromptVersion(prompt_id="test1", version=2, template="B")
        prompt_c = PromptVersion(prompt_id="test2", version=1, template="C")
        prompt_d = PromptVersion(prompt_id="test2", version=2, template="D")

        tester.create_test(prompt_a, prompt_b)
        tester.create_test(prompt_c, prompt_d)

        active = tester.get_active_tests()
        assert len(active) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prompt_optimizer(self):
        """Test create_prompt_optimizer factory."""
        optimizer = create_prompt_optimizer()
        assert isinstance(optimizer, PromptOptimizer)
        assert optimizer.improvement_threshold == 0.15

    def test_create_prompt_optimizer_custom(self):
        """Test create_prompt_optimizer with custom params."""
        optimizer = create_prompt_optimizer(improvement_threshold=0.20, min_samples=20)
        assert optimizer.improvement_threshold == 0.20
        assert optimizer.min_samples_for_analysis == 20

    def test_create_ab_tester(self):
        """Test create_ab_tester factory."""
        tester = create_ab_tester()
        assert isinstance(tester, PromptABTester)

    def test_create_ab_tester_with_optimizer(self):
        """Test create_ab_tester with existing optimizer."""
        optimizer = PromptOptimizer()
        tester = create_ab_tester(optimizer=optimizer)
        assert tester.optimizer is optimizer

    def test_create_ab_tester_custom_params(self):
        """Test create_ab_tester with custom params."""
        tester = create_ab_tester(min_samples=50, significance_level=0.01)
        assert tester.min_samples_per_variant == 50
        assert tester.significance_level == 0.01


class TestIntegration:
    """Integration tests for prompt optimization workflow."""

    def test_full_optimization_cycle(self):
        """Test complete optimization workflow."""
        # Create optimizer
        optimizer = create_prompt_optimizer(min_samples=5)

        # Register initial prompt
        prompt_v1 = PromptVersion(
            prompt_id="rule_gen",
            version=1,
            template="Generate a rule for: {principle}",
            category=PromptCategory.RULE_GENERATION,
        )
        optimizer.register_prompt(prompt_v1)

        # Track performance
        for i in range(7):
            optimizer.track_prompt_performance(
                "rule_gen_v1",
                success=(i < 5),  # 71% success
                confidence=0.7,
                domain="contracts",
            )

        # Analyze effectiveness
        report = optimizer.analyze_prompt_effectiveness("rule_gen_v1")
        assert report.overall_effectiveness == pytest.approx(5 / 7, rel=0.01)

        # Generate improvement candidates
        candidates = optimizer.generate_improvement_candidates("rule_gen_v1")
        assert len(candidates) > 0

        # Create new version from candidate
        new_version = optimizer.create_new_version(
            "rule_gen_v1",
            candidates[0].new_template,
            candidates[0].rationale,
        )
        assert new_version.version == 2

    def test_ab_test_workflow(self):
        """Test complete A/B testing workflow."""
        optimizer = create_prompt_optimizer()
        tester = create_ab_tester(optimizer, min_samples=10)

        # Create prompts
        prompt_a = PromptVersion(
            prompt_id="test", version=1, template="Simple: {input}"
        )
        prompt_b = PromptVersion(
            prompt_id="test", version=2, template="Detailed: {input}. Be precise."
        )

        # Start test
        config = tester.create_test(prompt_a, prompt_b)

        # Simulate usage with B performing better
        for i in range(15):
            tester.record_outcome(
                config.test_id, "test_v1", success=(i < 8), confidence=0.6
            )
            tester.record_outcome(
                config.test_id, "test_v2", success=(i < 12), confidence=0.8
            )

        # Check completion
        assert tester.check_test_completion(config.test_id) is True

        # Analyze
        result = tester.analyze_test(config.test_id)
        assert (
            result.prompt_b_metrics.success_rate > result.prompt_a_metrics.success_rate
        )

        # Get active tests
        active = tester.get_active_tests()
        assert len(active) == 1

    def test_underperformer_identification_and_improvement(self):
        """Test identifying and improving underperforming prompts."""
        optimizer = create_prompt_optimizer(min_samples=5)

        # Register multiple prompts
        prompts = [
            PromptVersion(prompt_id="good", version=1, template="Good prompt"),
            PromptVersion(prompt_id="okay", version=1, template="Okay prompt"),
            PromptVersion(prompt_id="bad", version=1, template="Bad prompt"),
        ]

        for p in prompts:
            optimizer.register_prompt(p)

        # Simulate performance
        for _ in range(10):
            optimizer.track_prompt_performance("good_v1", success=True)
            optimizer.track_prompt_performance("okay_v1", success=(_ < 7))
            optimizer.track_prompt_performance("bad_v1", success=(_ < 3))

        # Find underperformers
        underperformers = optimizer.identify_underperforming_prompts(threshold=0.7)

        # Should identify bad prompt
        assert any(p[0] == "bad_v1" for p in underperformers)

        # Generate improvements for worst performer
        worst = underperformers[0][0]
        candidates = optimizer.generate_improvement_candidates(worst)
        assert len(candidates) > 0


class TestImprovementSuggestion:
    """Tests for ImprovementSuggestion dataclass."""

    def test_basic_creation(self):
        """Test basic creation of ImprovementSuggestion."""
        from loft.meta.prompt_optimizer import ImprovementSuggestion

        suggestion = ImprovementSuggestion(
            suggestion_id="sugg_123",
            prompt_id="test_v1",
            suggestion_type="wording",
            description="Add clearer instructions",
            rationale="Success rate is below threshold",
            expected_improvement_percentage=15.0,
            confidence=0.8,
            priority="high",
            implementation_hint="Use step-by-step format",
        )

        assert suggestion.suggestion_id == "sugg_123"
        assert suggestion.prompt_id == "test_v1"
        assert suggestion.suggestion_type == "wording"
        assert suggestion.expected_improvement_percentage == 15.0
        assert suggestion.confidence == 0.8
        assert suggestion.priority == "high"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from loft.meta.prompt_optimizer import ImprovementSuggestion

        suggestion = ImprovementSuggestion(
            suggestion_id="sugg_123",
            prompt_id="test_v1",
            suggestion_type="structure",
            description="Test",
            rationale="Test rationale",
            expected_improvement_percentage=10.0,
            confidence=0.7,
        )

        data = suggestion.to_dict()
        assert data["suggestion_id"] == "sugg_123"
        assert data["suggestion_type"] == "structure"
        assert data["expected_improvement_percentage"] == 10.0
        assert "generated_at" in data


class TestAggregateEffectivenessReport:
    """Tests for AggregateEffectivenessReport dataclass."""

    def test_basic_creation(self):
        """Test basic creation of AggregateEffectivenessReport."""
        from loft.meta.prompt_optimizer import AggregateEffectivenessReport

        report = AggregateEffectivenessReport(
            report_id="report_123",
            generated_at=datetime.now(),
            total_prompts_tracked=5,
            total_uses_recorded=100,
            overall_success_rate=0.85,
            overall_average_confidence=0.75,
            overall_average_latency_ms=500.0,
            prompt_reports={"prompt_v1": {"success_rate": 0.9}},
            top_performers=[("prompt_v1", 0.9)],
            bottom_performers=[("prompt_v2", 0.6)],
            category_breakdown={"general": {"success_rate": 0.8}},
            domain_breakdown={"contracts": {"success_rate": 0.85}},
            trends={"prompt_v1": "stable"},
            recommendations=["Keep monitoring"],
        )

        assert report.report_id == "report_123"
        assert report.total_prompts_tracked == 5
        assert report.overall_success_rate == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from loft.meta.prompt_optimizer import AggregateEffectivenessReport

        report = AggregateEffectivenessReport(
            report_id="report_123",
            generated_at=datetime.now(),
            total_prompts_tracked=3,
            total_uses_recorded=50,
            overall_success_rate=0.8,
            overall_average_confidence=0.7,
            overall_average_latency_ms=400.0,
            prompt_reports={},
            top_performers=[],
            bottom_performers=[],
            category_breakdown={},
            domain_breakdown={},
            trends={},
            recommendations=[],
        )

        data = report.to_dict()
        assert data["report_id"] == "report_123"
        assert data["total_prompts_tracked"] == 3
        assert "generated_at" in data


class TestGenerateEffectivenessReport:
    """Tests for generate_effectiveness_report method."""

    @pytest.fixture
    def optimizer_with_data(self):
        """Create optimizer with diverse test data."""
        optimizer = PromptOptimizer(min_samples_for_analysis=5)

        # Register multiple prompts with different categories
        prompt1 = PromptVersion(
            prompt_id="rule_gen",
            version=1,
            template="Generate rule: {input}",
            category=PromptCategory.RULE_GENERATION,
        )
        prompt2 = PromptVersion(
            prompt_id="validation",
            version=1,
            template="Validate: {input}",
            category=PromptCategory.VALIDATION,
        )
        prompt3 = PromptVersion(
            prompt_id="translation",
            version=1,
            template="Translate: {input}",
            category=PromptCategory.TRANSLATION,
        )

        optimizer.register_prompt(prompt1)
        optimizer.register_prompt(prompt2)
        optimizer.register_prompt(prompt3)

        # Add performance data
        for i in range(15):
            optimizer.track_prompt_performance(
                prompt1.full_id,
                success=i < 12,  # 80% success
                confidence=0.8,
                latency_ms=100,
                domain="contracts",
            )

        for i in range(10):
            optimizer.track_prompt_performance(
                prompt2.full_id,
                success=i < 9,  # 90% success
                confidence=0.9,
                latency_ms=50,
                domain="torts",
            )

        for i in range(20):
            optimizer.track_prompt_performance(
                prompt3.full_id,
                success=i < 10,  # 50% success
                confidence=0.5,
                latency_ms=200,
                domain="contracts",
            )

        return optimizer

    def test_generate_report_basic(self, optimizer_with_data):
        """Test basic report generation."""
        report = optimizer_with_data.generate_effectiveness_report()

        assert report.total_prompts_tracked == 3
        assert report.total_uses_recorded == 45  # 15 + 10 + 20
        assert len(report.prompt_reports) == 3
        assert len(report.recommendations) > 0

    def test_report_calculates_overall_metrics(self, optimizer_with_data):
        """Test that overall metrics are calculated correctly."""
        report = optimizer_with_data.generate_effectiveness_report()

        # Total: 12 + 9 + 10 = 31 successes out of 45
        expected_rate = 31 / 45
        assert abs(report.overall_success_rate - expected_rate) < 0.01

    def test_report_identifies_top_bottom_performers(self, optimizer_with_data):
        """Test identification of top and bottom performers."""
        report = optimizer_with_data.generate_effectiveness_report()

        # Should have top performers
        assert len(report.top_performers) > 0
        # Best performer should be validation (90%)
        top_prompt, top_rate = report.top_performers[0]
        assert top_rate >= 0.9

        # Should have bottom performers
        assert len(report.bottom_performers) > 0
        # Worst performer should be translation (50%)
        bottom_prompt, bottom_rate = report.bottom_performers[0]
        assert bottom_rate <= 0.5

    def test_report_includes_category_breakdown(self, optimizer_with_data):
        """Test category breakdown in report."""
        report = optimizer_with_data.generate_effectiveness_report()

        assert len(report.category_breakdown) > 0
        assert "rule_generation" in report.category_breakdown
        assert "validation" in report.category_breakdown

    def test_report_includes_domain_breakdown(self, optimizer_with_data):
        """Test domain breakdown in report."""
        report = optimizer_with_data.generate_effectiveness_report()

        assert len(report.domain_breakdown) > 0
        assert "contracts" in report.domain_breakdown
        assert "torts" in report.domain_breakdown

    def test_empty_optimizer_report(self):
        """Test report generation with no data."""
        optimizer = PromptOptimizer()
        report = optimizer.generate_effectiveness_report()

        assert report.total_prompts_tracked == 0
        assert report.total_uses_recorded == 0
        assert report.overall_success_rate == 0.0

    def test_report_to_dict(self, optimizer_with_data):
        """Test report serialization."""
        report = optimizer_with_data.generate_effectiveness_report()
        data = report.to_dict()

        assert "report_id" in data
        assert "total_prompts_tracked" in data
        assert "overall_success_rate" in data
        assert "recommendations" in data


class TestSuggestImprovements:
    """Tests for suggest_improvements method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for testing."""
        return PromptOptimizer(min_samples_for_analysis=10)

    def test_suggest_improvements_prompt_not_found(self, optimizer):
        """Test error when prompt not found."""
        with pytest.raises(ValueError, match="not found"):
            optimizer.suggest_improvements("nonexistent_v1")

    def test_suggest_improvements_insufficient_data(self, optimizer):
        """Test suggestion for insufficient data."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add only 5 samples (below threshold of 10)
        for i in range(5):
            optimizer.track_prompt_performance(prompt.full_id, success=True)

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        assert len(suggestions) == 1
        assert suggestions[0].suggestion_type == "data_collection"
        assert suggestions[0].priority == "high"

    def test_suggest_improvements_low_success_rate(self, optimizer):
        """Test suggestions for low success rate."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add 15 samples with 50% success rate
        for i in range(15):
            optimizer.track_prompt_performance(prompt.full_id, success=i < 7)

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        # Should have wording suggestion for low success
        wording_suggestions = [s for s in suggestions if s.suggestion_type == "wording"]
        assert len(wording_suggestions) > 0
        assert wording_suggestions[0].priority == "high"

    def test_suggest_improvements_syntax_issues(self, optimizer):
        """Test suggestions for syntax validity issues."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add samples with syntax issues
        for i in range(15):
            optimizer.track_prompt_performance(
                prompt.full_id,
                success=True,
                syntax_valid=i < 10,  # 66% syntax valid
            )

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        # Should have constraints suggestion for syntax issues
        constraint_suggestions = [
            s for s in suggestions if s.suggestion_type == "constraints"
        ]
        assert len(constraint_suggestions) > 0

    def test_suggest_improvements_domain_weakness(self, optimizer):
        """Test suggestions for domain-specific weaknesses."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add samples - good overall but bad for one domain
        for i in range(20):
            optimizer.track_prompt_performance(
                prompt.full_id, success=True, domain="contracts"
            )

        for i in range(10):
            optimizer.track_prompt_performance(
                prompt.full_id,
                success=i < 3,
                domain="torts",  # 30% for torts
            )

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        # Should have domain-specific suggestion
        domain_suggestions = [s for s in suggestions if s.suggestion_type == "domain"]
        assert len(domain_suggestions) > 0
        assert "torts" in domain_suggestions[0].description

    def test_suggest_improvements_low_confidence(self, optimizer):
        """Test suggestions for low confidence scores."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add samples with low confidence
        for i in range(15):
            optimizer.track_prompt_performance(
                prompt.full_id, success=True, confidence=0.4
            )

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        # Should have examples suggestion for low confidence
        examples_suggestions = [
            s for s in suggestions if s.suggestion_type == "examples"
        ]
        assert len(examples_suggestions) > 0

    def test_suggest_improvements_high_latency(self, optimizer):
        """Test suggestions for high latency."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add samples with high latency
        for i in range(15):
            optimizer.track_prompt_performance(
                prompt.full_id, success=True, latency_ms=10000, confidence=0.9
            )

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        # Should have structure suggestion for high latency
        structure_suggestions = [
            s for s in suggestions if s.suggestion_type == "structure"
        ]
        assert len(structure_suggestions) > 0

    def test_suggest_improvements_well_performing(self, optimizer):
        """Test suggestions for well-performing prompt."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add samples with excellent performance
        for i in range(15):
            optimizer.track_prompt_performance(
                prompt.full_id,
                success=True,
                confidence=0.95,
                latency_ms=100,
                syntax_valid=True,
            )

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        # Should suggest A/B testing for well-performing prompt
        assert len(suggestions) > 0
        assert suggestions[0].suggestion_type == "testing"
        assert suggestions[0].priority == "low"

    def test_suggestions_sorted_by_priority(self, optimizer):
        """Test that suggestions are sorted by priority."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        # Add samples that trigger multiple suggestions
        for i in range(15):
            optimizer.track_prompt_performance(
                prompt.full_id,
                success=i < 7,  # 47% success
                confidence=0.4,
                latency_ms=8000,
                syntax_valid=i < 10,
            )

        suggestions = optimizer.suggest_improvements(prompt.full_id)

        # Should have multiple suggestions sorted by priority
        assert len(suggestions) >= 2
        priorities = [s.priority for s in suggestions]
        priority_order = {"high": 0, "medium": 1, "low": 2}
        priority_values = [priority_order.get(p, 1) for p in priorities]
        assert priority_values == sorted(priority_values)

    def test_suggestions_to_dict(self, optimizer):
        """Test suggestion serialization."""
        prompt = PromptVersion(
            prompt_id="test",
            version=1,
            template="Test: {input}",
        )
        optimizer.register_prompt(prompt)

        for i in range(15):
            optimizer.track_prompt_performance(prompt.full_id, success=i < 7)

        suggestions = optimizer.suggest_improvements(prompt.full_id)
        assert len(suggestions) > 0

        data = suggestions[0].to_dict()
        assert "suggestion_id" in data
        assert "prompt_id" in data
        assert "suggestion_type" in data
        assert "expected_improvement_percentage" in data
