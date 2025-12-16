"""
Unit tests for baseline metrics.

Issue #257: Baseline Validation Benchmarks
"""

import tempfile
from pathlib import Path

from loft.benchmarks import (
    BaselineMetrics,
    BenchmarkConfig,
    BenchmarkSuite,
    EndToEndMetrics,
    MetaReasoningMetrics,
    PersistenceMetrics,
    RuleGenerationMetrics,
    TranslationMetrics,
    ValidationMetrics,
    compare_baselines,
)


class TestRuleGenerationMetrics:
    """Tests for RuleGenerationMetrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = RuleGenerationMetrics(
            rules_per_hour=150.0,
            avg_generation_time_ms=24000.0,
            avg_tokens_used=1500,
            success_rate=0.92,
            retry_rate=0.15,
            predicate_alignment_rate=0.88,
        )

        assert metrics.rules_per_hour == 150.0
        assert metrics.success_rate == 0.92

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = RuleGenerationMetrics(
            rules_per_hour=150.0,
            avg_generation_time_ms=24000.0,
            avg_tokens_used=1500,
            success_rate=0.92,
            retry_rate=0.15,
            predicate_alignment_rate=0.88,
        )

        data = metrics.to_dict()
        assert data["rules_per_hour"] == 150.0
        assert "success_rate" in data


class TestValidationMetrics:
    """Tests for ValidationMetrics."""

    def test_initialization(self):
        """Test validation metrics."""
        metrics = ValidationMetrics(
            syntax_pass_rate=0.98,
            semantic_pass_rate=0.95,
            empirical_pass_rate=0.87,
            consensus_pass_rate=0.91,
            dialectical_pass_rate=0.89,
            avg_validation_time_ms=850.0,
        )

        assert metrics.syntax_pass_rate == 0.98
        assert metrics.overall_pass_rate == 0.0  # Default


class TestBaselineMetrics:
    """Tests for BaselineMetrics."""

    def test_save_and_load(self):
        """Test baseline save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"

            # Create baseline
            baseline = BaselineMetrics(
                timestamp="2025-12-16T10:00:00",
                commit_hash="abc123",
                configuration={"test": True},
                rule_generation=RuleGenerationMetrics(
                    rules_per_hour=150.0,
                    avg_generation_time_ms=24000.0,
                    avg_tokens_used=1500,
                    success_rate=0.92,
                    retry_rate=0.15,
                    predicate_alignment_rate=0.88,
                ),
                validation=ValidationMetrics(
                    syntax_pass_rate=0.98,
                    semantic_pass_rate=0.95,
                    empirical_pass_rate=0.87,
                    consensus_pass_rate=0.91,
                    dialectical_pass_rate=0.89,
                    avg_validation_time_ms=850.0,
                ),
                meta_reasoning=MetaReasoningMetrics(
                    strategy_distribution={},
                    strategy_success_rates={},
                    adaptation_frequency=2.5,
                    failure_patterns_detected=12,
                    recommendations_generated=18,
                    recommendations_applied=14,
                    prompt_variants_tested=8,
                    avg_improvement_per_optimization=0.07,
                ),
                translation=TranslationMetrics(
                    asp_to_nl_fidelity=0.94,
                    asp_to_nl_avg_time_ms=120.0,
                    nl_to_asp_fidelity=0.91,
                    nl_to_asp_avg_time_ms=180.0,
                    roundtrip_fidelity=0.89,
                    semantic_preservation_rate=0.92,
                    grounding_success_rate=0.95,
                ),
                end_to_end=EndToEndMetrics(
                    cases_per_hour=126.3,
                    avg_case_time_ms=28500.0,
                    avg_gap_identification_ms=2000.0,
                    avg_rule_generation_ms=24000.0,
                    avg_validation_ms=850.0,
                    avg_incorporation_ms=500.0,
                    avg_persistence_ms=150.0,
                    cases_with_new_rules=0.78,
                    cases_with_improvement=0.65,
                ),
                persistence=PersistenceMetrics(
                    avg_save_time_ms=45.0,
                    avg_load_time_ms=75.0,
                    avg_snapshot_time_ms=120.0,
                    avg_file_size_bytes=8500.0,
                    storage_overhead_ratio=1.15,
                    save_success_rate=0.999,
                    load_success_rate=0.998,
                    corruption_recovery_rate=0.95,
                ),
            )

            # Save
            baseline.save(path)
            assert path.exists()

            # Load
            loaded = BaselineMetrics.load(path)
            assert loaded.commit_hash == "abc123"
            assert loaded.rule_generation.rules_per_hour == 150.0

    def test_to_markdown(self):
        """Test markdown report generation."""
        baseline = BaselineMetrics(
            timestamp="2025-12-16T10:00:00",
            commit_hash="abc123",
            configuration={"description": "Test baseline"},
            rule_generation=RuleGenerationMetrics(
                rules_per_hour=150.0,
                avg_generation_time_ms=24000.0,
                avg_tokens_used=1500,
                success_rate=0.92,
                retry_rate=0.15,
                predicate_alignment_rate=0.88,
            ),
            validation=ValidationMetrics(
                syntax_pass_rate=0.98,
                semantic_pass_rate=0.95,
                empirical_pass_rate=0.87,
                consensus_pass_rate=0.91,
                dialectical_pass_rate=0.89,
                avg_validation_time_ms=850.0,
            ),
            meta_reasoning=MetaReasoningMetrics(
                strategy_distribution={},
                strategy_success_rates={},
                adaptation_frequency=2.5,
                failure_patterns_detected=12,
                recommendations_generated=18,
                recommendations_applied=14,
                prompt_variants_tested=8,
                avg_improvement_per_optimization=0.07,
            ),
            translation=TranslationMetrics(
                asp_to_nl_fidelity=0.94,
                asp_to_nl_avg_time_ms=120.0,
                nl_to_asp_fidelity=0.91,
                nl_to_asp_avg_time_ms=180.0,
                roundtrip_fidelity=0.89,
                semantic_preservation_rate=0.92,
                grounding_success_rate=0.95,
            ),
            end_to_end=EndToEndMetrics(
                cases_per_hour=126.3,
                avg_case_time_ms=28500.0,
                avg_gap_identification_ms=2000.0,
                avg_rule_generation_ms=24000.0,
                avg_validation_ms=850.0,
                avg_incorporation_ms=500.0,
                avg_persistence_ms=150.0,
                cases_with_new_rules=0.78,
                cases_with_improvement=0.65,
            ),
            persistence=PersistenceMetrics(
                avg_save_time_ms=45.0,
                avg_load_time_ms=75.0,
                avg_snapshot_time_ms=120.0,
                avg_file_size_bytes=8500.0,
                storage_overhead_ratio=1.15,
                save_success_rate=0.999,
                load_success_rate=0.998,
                corruption_recovery_rate=0.95,
            ),
        )

        markdown = baseline.to_markdown()

        assert "LOFT Infrastructure Baseline Metrics" in markdown
        assert "abc123" in markdown
        assert "150.0" in markdown  # rules per hour


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_run_all(self):
        """Test running complete benchmark suite."""
        config = BenchmarkConfig(
            sample_size=10,
            enable_llm=False,
            commit_hash="test123",
        )

        suite = BenchmarkSuite(config)
        baseline = suite.run_all()

        assert baseline.commit_hash == "test123"
        assert baseline.rule_generation.rules_per_hour > 0
        assert baseline.validation.syntax_pass_rate > 0
        assert baseline.translation.roundtrip_fidelity > 0


class TestComparison:
    """Tests for baseline comparison."""

    def test_compare_baselines(self):
        """Test baseline comparison."""
        # Previous baseline
        previous = BaselineMetrics(
            timestamp="2025-12-15T10:00:00",
            commit_hash="old123",
            configuration={},
            rule_generation=RuleGenerationMetrics(
                rules_per_hour=100.0,
                avg_generation_time_ms=30000.0,
                avg_tokens_used=1500,
                success_rate=0.85,
                retry_rate=0.20,
                predicate_alignment_rate=0.80,
            ),
            validation=ValidationMetrics(
                syntax_pass_rate=0.95,
                semantic_pass_rate=0.90,
                empirical_pass_rate=0.85,
                consensus_pass_rate=0.88,
                dialectical_pass_rate=0.86,
                avg_validation_time_ms=900.0,
                overall_pass_rate=0.80,
            ),
            meta_reasoning=MetaReasoningMetrics(
                strategy_distribution={},
                strategy_success_rates={},
                adaptation_frequency=2.0,
                failure_patterns_detected=10,
                recommendations_generated=15,
                recommendations_applied=12,
                prompt_variants_tested=5,
                avg_improvement_per_optimization=0.05,
            ),
            translation=TranslationMetrics(
                asp_to_nl_fidelity=0.90,
                asp_to_nl_avg_time_ms=130.0,
                nl_to_asp_fidelity=0.88,
                nl_to_asp_avg_time_ms=190.0,
                roundtrip_fidelity=0.85,
                semantic_preservation_rate=0.90,
                grounding_success_rate=0.93,
            ),
            end_to_end=EndToEndMetrics(
                cases_per_hour=110.0,
                avg_case_time_ms=32000.0,
                avg_gap_identification_ms=2200.0,
                avg_rule_generation_ms=28000.0,
                avg_validation_ms=900.0,
                avg_incorporation_ms=600.0,
                avg_persistence_ms=180.0,
                cases_with_new_rules=0.75,
                cases_with_improvement=0.60,
            ),
            persistence=PersistenceMetrics(
                avg_save_time_ms=50.0,
                avg_load_time_ms=80.0,
                avg_snapshot_time_ms=130.0,
                avg_file_size_bytes=9000.0,
                storage_overhead_ratio=1.20,
                save_success_rate=0.998,
                load_success_rate=0.997,
                corruption_recovery_rate=0.94,
            ),
        )

        # Current baseline (improved)
        current = BaselineMetrics(
            timestamp="2025-12-16T10:00:00",
            commit_hash="new456",
            configuration={},
            rule_generation=RuleGenerationMetrics(
                rules_per_hour=150.0,  # +50% improvement
                avg_generation_time_ms=24000.0,  # Faster
                avg_tokens_used=1500,
                success_rate=0.92,  # Better
                retry_rate=0.15,
                predicate_alignment_rate=0.88,
            ),
            validation=ValidationMetrics(
                syntax_pass_rate=0.98,
                semantic_pass_rate=0.95,
                empirical_pass_rate=0.87,
                consensus_pass_rate=0.91,
                dialectical_pass_rate=0.89,
                avg_validation_time_ms=850.0,
                overall_pass_rate=0.85,
            ),
            meta_reasoning=MetaReasoningMetrics(
                strategy_distribution={},
                strategy_success_rates={},
                adaptation_frequency=2.5,
                failure_patterns_detected=12,
                recommendations_generated=18,
                recommendations_applied=14,
                prompt_variants_tested=8,
                avg_improvement_per_optimization=0.07,
            ),
            translation=TranslationMetrics(
                asp_to_nl_fidelity=0.94,
                asp_to_nl_avg_time_ms=120.0,
                nl_to_asp_fidelity=0.91,
                nl_to_asp_avg_time_ms=180.0,
                roundtrip_fidelity=0.89,
                semantic_preservation_rate=0.92,
                grounding_success_rate=0.95,
            ),
            end_to_end=EndToEndMetrics(
                cases_per_hour=126.3,
                avg_case_time_ms=28500.0,
                avg_gap_identification_ms=2000.0,
                avg_rule_generation_ms=24000.0,
                avg_validation_ms=850.0,
                avg_incorporation_ms=500.0,
                avg_persistence_ms=150.0,
                cases_with_new_rules=0.78,
                cases_with_improvement=0.65,
            ),
            persistence=PersistenceMetrics(
                avg_save_time_ms=45.0,
                avg_load_time_ms=75.0,
                avg_snapshot_time_ms=120.0,
                avg_file_size_bytes=8500.0,
                storage_overhead_ratio=1.15,
                save_success_rate=0.999,
                load_success_rate=0.998,
                corruption_recovery_rate=0.95,
            ),
        )

        comparison = compare_baselines(previous, current)

        assert comparison.improvement_count > 0
        assert comparison.total_metrics > 0

    def test_comparison_markdown(self):
        """Test comparison markdown output."""
        previous = BaselineMetrics(
            timestamp="2025-12-15T10:00:00",
            commit_hash="old123",
            configuration={},
            rule_generation=RuleGenerationMetrics(
                rules_per_hour=100.0,
                avg_generation_time_ms=30000.0,
                avg_tokens_used=1500,
                success_rate=0.85,
                retry_rate=0.20,
                predicate_alignment_rate=0.80,
            ),
            validation=ValidationMetrics(
                syntax_pass_rate=0.95,
                semantic_pass_rate=0.90,
                empirical_pass_rate=0.85,
                consensus_pass_rate=0.88,
                dialectical_pass_rate=0.86,
                avg_validation_time_ms=900.0,
                overall_pass_rate=0.80,
            ),
            meta_reasoning=MetaReasoningMetrics(
                strategy_distribution={},
                strategy_success_rates={},
                adaptation_frequency=2.0,
                failure_patterns_detected=10,
                recommendations_generated=15,
                recommendations_applied=12,
                prompt_variants_tested=5,
                avg_improvement_per_optimization=0.05,
            ),
            translation=TranslationMetrics(
                asp_to_nl_fidelity=0.90,
                asp_to_nl_avg_time_ms=130.0,
                nl_to_asp_fidelity=0.88,
                nl_to_asp_avg_time_ms=190.0,
                roundtrip_fidelity=0.85,
                semantic_preservation_rate=0.90,
                grounding_success_rate=0.93,
            ),
            end_to_end=EndToEndMetrics(
                cases_per_hour=110.0,
                avg_case_time_ms=32000.0,
                avg_gap_identification_ms=2200.0,
                avg_rule_generation_ms=28000.0,
                avg_validation_ms=900.0,
                avg_incorporation_ms=600.0,
                avg_persistence_ms=180.0,
                cases_with_new_rules=0.75,
                cases_with_improvement=0.60,
            ),
            persistence=PersistenceMetrics(
                avg_save_time_ms=50.0,
                avg_load_time_ms=80.0,
                avg_snapshot_time_ms=130.0,
                avg_file_size_bytes=9000.0,
                storage_overhead_ratio=1.20,
                save_success_rate=0.998,
                load_success_rate=0.997,
                corruption_recovery_rate=0.94,
            ),
        )

        current = BaselineMetrics(
            timestamp="2025-12-16T10:00:00",
            commit_hash="new456",
            configuration={},
            rule_generation=RuleGenerationMetrics(
                rules_per_hour=150.0,
                avg_generation_time_ms=24000.0,
                avg_tokens_used=1500,
                success_rate=0.92,
                retry_rate=0.15,
                predicate_alignment_rate=0.88,
            ),
            validation=ValidationMetrics(
                syntax_pass_rate=0.98,
                semantic_pass_rate=0.95,
                empirical_pass_rate=0.87,
                consensus_pass_rate=0.91,
                dialectical_pass_rate=0.89,
                avg_validation_time_ms=850.0,
                overall_pass_rate=0.85,
            ),
            meta_reasoning=MetaReasoningMetrics(
                strategy_distribution={},
                strategy_success_rates={},
                adaptation_frequency=2.5,
                failure_patterns_detected=12,
                recommendations_generated=18,
                recommendations_applied=14,
                prompt_variants_tested=8,
                avg_improvement_per_optimization=0.07,
            ),
            translation=TranslationMetrics(
                asp_to_nl_fidelity=0.94,
                asp_to_nl_avg_time_ms=120.0,
                nl_to_asp_fidelity=0.91,
                nl_to_asp_avg_time_ms=180.0,
                roundtrip_fidelity=0.89,
                semantic_preservation_rate=0.92,
                grounding_success_rate=0.95,
            ),
            end_to_end=EndToEndMetrics(
                cases_per_hour=126.3,
                avg_case_time_ms=28500.0,
                avg_gap_identification_ms=2000.0,
                avg_rule_generation_ms=24000.0,
                avg_validation_ms=850.0,
                avg_incorporation_ms=500.0,
                avg_persistence_ms=150.0,
                cases_with_new_rules=0.78,
                cases_with_improvement=0.65,
            ),
            persistence=PersistenceMetrics(
                avg_save_time_ms=45.0,
                avg_load_time_ms=75.0,
                avg_snapshot_time_ms=120.0,
                avg_file_size_bytes=8500.0,
                storage_overhead_ratio=1.15,
                save_success_rate=0.999,
                load_success_rate=0.998,
                corruption_recovery_rate=0.95,
            ),
        )

        comparison = compare_baselines(previous, current)
        markdown = comparison.to_markdown()

        assert "Baseline Comparison Report" in markdown
        assert "old123" in markdown
        assert "new456" in markdown
