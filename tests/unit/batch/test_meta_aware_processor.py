"""
Unit tests for meta-aware batch processor.

Tests MetaAwareBatchProcessor, MetaAwareBatchConfig, and related classes.

Issue #255: Meta-Reasoning Integration with Batch Processing.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from loft.batch.meta_aware_processor import (
    Adaptation,
    FailurePattern,
    MetaAwareBatchConfig,
    MetaAwareBatchProcessor,
    MetaProcessingResult,
    create_meta_aware_processor,
)
from loft.batch.schemas import CaseResult, CaseStatus


@pytest.fixture
def mock_pipeline_processor():
    """Create a mock pipeline processor."""
    processor = MagicMock()
    processor.process_case.return_value = CaseResult(
        case_id="test_001",
        status=CaseStatus.SUCCESS,
        processed_at=datetime.now(),
        processing_time_ms=100.0,
        rules_generated=2,
        rules_accepted=1,
        rules_rejected=1,
        prediction_correct=True,
        confidence=0.85,
    )
    return processor


@pytest.fixture
def sample_case():
    """Create a sample test case."""
    return {
        "id": "test_case_001",
        "asp_facts": "contract(c1). offer(c1). acceptance(c1).",
        "ground_truth": "valid(c1).",
        "_domain": "contracts",
    }


class TestMetaAwareBatchConfig:
    """Tests for MetaAwareBatchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MetaAwareBatchConfig()

        assert config.enable_strategy_selection is True
        assert config.enable_failure_analysis is True
        assert config.enable_prompt_optimization is True
        assert config.min_failures_for_adaptation == 5
        assert config.failure_pattern_threshold == 0.5

    def test_custom_values(self):
        """Test custom configuration."""
        config = MetaAwareBatchConfig(
            enable_strategy_selection=False,
            min_failures_for_adaptation=10,
            default_strategy="dialectical",
        )

        assert config.enable_strategy_selection is False
        assert config.min_failures_for_adaptation == 10
        assert config.default_strategy == "dialectical"

    def test_default_strategy_weights(self):
        """Test default strategy weights initialization."""
        config = MetaAwareBatchConfig()

        assert "checklist" in config.strategy_weights
        assert "dialectical" in config.strategy_weights
        assert config.strategy_weights["rule_based"] == 1.0


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""

    def test_creation(self):
        """Test failure pattern creation."""
        now = datetime.now()
        pattern = FailurePattern(
            failure_type="timeout",
            count=3,
            cases=["case1", "case2", "case3"],
            first_seen=now,
            last_seen=now,
        )

        assert pattern.failure_type == "timeout"
        assert pattern.count == 3
        assert len(pattern.cases) == 3

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now()
        pattern = FailurePattern(
            failure_type="validation_failure",
            count=2,
            cases=["c1", "c2"],
            first_seen=now,
            last_seen=now,
            root_cause="insufficient_predicates",
        )

        data = pattern.to_dict()

        assert data["failure_type"] == "validation_failure"
        assert data["count"] == 2
        assert data["root_cause"] == "insufficient_predicates"


class TestAdaptation:
    """Tests for Adaptation dataclass."""

    def test_creation(self):
        """Test adaptation creation."""
        adaptation = Adaptation(
            adaptation_type="strategy_change",
            timestamp=datetime.now(),
            trigger="failure_pattern:timeout",
            changes={"dialectical": 0.5},
            reason="Too many timeouts",
        )

        assert adaptation.adaptation_type == "strategy_change"
        assert adaptation.trigger == "failure_pattern:timeout"

    def test_to_dict(self):
        """Test serialization."""
        adaptation = Adaptation(
            adaptation_type="prompt_refinement",
            timestamp=datetime.now(),
            trigger="low_acceptance_rate",
            changes={"temperature": 0.7},
            reason="Improve generation quality",
        )

        data = adaptation.to_dict()

        assert data["adaptation_type"] == "prompt_refinement"
        assert "timestamp" in data


class TestMetaAwareBatchProcessor:
    """Tests for MetaAwareBatchProcessor."""

    def test_initialization(self, mock_pipeline_processor):
        """Test processor initialization."""
        config = MetaAwareBatchConfig()
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
            config=config,
        )

        assert processor.pipeline == mock_pipeline_processor
        assert processor.config == config
        assert processor.cases_processed == 0
        assert processor.failure_patterns == []

    def test_factory_function(self, mock_pipeline_processor):
        """Test factory function creates processor."""
        processor = create_meta_aware_processor(
            pipeline_processor=mock_pipeline_processor,
        )

        assert isinstance(processor, MetaAwareBatchProcessor)

    def test_process_case_success(self, mock_pipeline_processor, sample_case):
        """Test successful case processing."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        result = processor.process_case_with_meta(sample_case)

        assert isinstance(result, MetaProcessingResult)
        assert result.case_result.status == CaseStatus.SUCCESS
        assert processor.cases_processed == 1
        assert processor.successful_cases == 1

    def test_process_case_failure(self, mock_pipeline_processor, sample_case):
        """Test failed case processing."""
        mock_pipeline_processor.process_case.return_value = CaseResult(
            case_id="test_001",
            status=CaseStatus.FAILED,
            processed_at=datetime.now(),
            processing_time_ms=100.0,
            rules_generated=0,
            rules_accepted=0,
            rules_rejected=0,
            error_message="Timeout exceeded",
        )

        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
            config=MetaAwareBatchConfig(enable_failure_analysis=True),
        )

        result = processor.process_case_with_meta(sample_case)

        assert result.case_result.status == CaseStatus.FAILED
        assert processor.failed_cases == 1
        assert len(processor.failure_patterns) > 0

    def test_strategy_selection_disabled(self, mock_pipeline_processor, sample_case):
        """Test with strategy selection disabled."""
        config = MetaAwareBatchConfig(
            enable_strategy_selection=False,
            default_strategy="checklist",
        )
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
            config=config,
        )

        result = processor.process_case_with_meta(sample_case)

        assert result.strategy_used == "checklist"

    def test_detect_case_type_from_domain(self, mock_pipeline_processor):
        """Test case type detection from domain."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        case_type = processor._detect_case_type({"_domain": "contracts"})
        assert case_type == "contracts"

    def test_detect_case_type_from_facts(self, mock_pipeline_processor):
        """Test case type detection from facts."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        case_type = processor._detect_case_type(
            {"asp_facts": "tort(t1). negligence(t1)."}
        )
        assert case_type == "tort"

    def test_categorize_failure_timeout(self, mock_pipeline_processor):
        """Test timeout failure categorization."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        result = CaseResult(
            case_id="test",
            status=CaseStatus.FAILED,
            processed_at=datetime.now(),
            processing_time_ms=100.0,
            error_message="Request timeout exceeded",
        )

        category = processor._categorize_failure(result)
        assert category == "timeout"

    def test_categorize_failure_validation(self, mock_pipeline_processor):
        """Test validation failure categorization."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        result = CaseResult(
            case_id="test",
            status=CaseStatus.FAILED,
            processed_at=datetime.now(),
            processing_time_ms=100.0,
            error_message="Validation check failed",
        )

        category = processor._categorize_failure(result)
        assert category == "validation_failure"

    def test_categorize_failure_no_rules_generated(self, mock_pipeline_processor):
        """Test no rules generated categorization."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        result = CaseResult(
            case_id="test",
            status=CaseStatus.FAILED,
            processed_at=datetime.now(),
            processing_time_ms=100.0,
            rules_generated=0,
        )

        category = processor._categorize_failure(result)
        assert category == "no_rules_generated"

    def test_should_adapt_not_enough_failures(self, mock_pipeline_processor):
        """Test adaptation not triggered with few failures."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
            config=MetaAwareBatchConfig(min_failures_for_adaptation=5),
        )

        # Add only 2 failures
        processor.failure_patterns = [
            FailurePattern(
                failure_type="timeout",
                count=1,
                cases=["c1"],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            ),
            FailurePattern(
                failure_type="timeout",
                count=1,
                cases=["c2"],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            ),
        ]

        assert processor._should_adapt() is False

    def test_should_adapt_pattern_threshold(self, mock_pipeline_processor):
        """Test adaptation triggered when pattern threshold exceeded."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
            config=MetaAwareBatchConfig(
                min_failures_for_adaptation=5,
                adaptation_window_size=10,
                failure_pattern_threshold=0.5,
            ),
        )

        # Add 10 failures, 6 of same type (>50%)
        for i in range(6):
            processor.failure_patterns.append(
                FailurePattern(
                    failure_type="timeout",
                    count=1,
                    cases=[f"c{i}"],
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                )
            )
        for i in range(4):
            processor.failure_patterns.append(
                FailurePattern(
                    failure_type="other",
                    count=1,
                    cases=[f"c{i+6}"],
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                )
            )

        assert processor._should_adapt() is True

    def test_adapt_strategies_timeout(self, mock_pipeline_processor):
        """Test strategy adaptation for timeout failures."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        # Add timeout failures
        for i in range(6):
            processor.failure_patterns.append(
                FailurePattern(
                    failure_type="timeout",
                    count=1,
                    cases=[f"c{i}"],
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                )
            )

        initial_weights = processor.config.strategy_weights.copy()
        processor._adapt_strategies()

        # Weights should have changed
        assert processor.config.strategy_weights != initial_weights
        assert len(processor.adaptations) > 0

    def test_get_processor_function(self, mock_pipeline_processor, sample_case):
        """Test getting processor function for harness."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        process_fn = processor.get_processor_function()

        result = process_fn(sample_case, [])

        assert isinstance(result, CaseResult)
        assert result.status == CaseStatus.SUCCESS

    def test_get_failure_summary(self, mock_pipeline_processor):
        """Test failure summary generation."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        processor.failure_patterns = [
            FailurePattern(
                failure_type="timeout",
                count=2,
                cases=["c1", "c2"],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            ),
        ]

        summary = processor.get_failure_summary()

        assert summary["total_failures"] == 1
        assert "timeout" in summary["failure_types"]

    def test_get_strategy_summary(self, mock_pipeline_processor):
        """Test strategy summary generation."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        processor.strategy_performance = {
            "checklist": {"successes": 5, "failures": 2, "total": 7}
        }

        summary = processor.get_strategy_summary()

        assert "current_weights" in summary
        assert "performance" in summary
        assert summary["performance"]["checklist"]["successes"] == 5

    def test_update_strategy_performance(self, mock_pipeline_processor):
        """Test strategy performance tracking."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
        )

        processor._update_strategy_performance("checklist", success=True)
        processor._update_strategy_performance("checklist", success=True)
        processor._update_strategy_performance("checklist", success=False)

        perf = processor.strategy_performance["checklist"]
        assert perf["successes"] == 2
        assert perf["failures"] == 1
        assert perf["total"] == 3
        assert perf["success_rate"] == pytest.approx(2 / 3)


class TestMetaProcessingResult:
    """Tests for MetaProcessingResult."""

    def test_creation(self):
        """Test result creation."""
        case_result = CaseResult(
            case_id="test",
            status=CaseStatus.SUCCESS,
            processed_at=datetime.now(),
            processing_time_ms=100.0,
        )

        result = MetaProcessingResult(
            case_result=case_result,
            strategy_used="checklist",
            adaptation_triggered=False,
        )

        assert result.case_result == case_result
        assert result.strategy_used == "checklist"
        assert result.adaptation_triggered is False

    def test_with_insights(self):
        """Test result with processing insights."""
        case_result = CaseResult(
            case_id="test",
            status=CaseStatus.SUCCESS,
            processed_at=datetime.now(),
            processing_time_ms=100.0,
        )

        result = MetaProcessingResult(
            case_result=case_result,
            strategy_used="dialectical",
            processing_insights={
                "processing_time_ms": 250.0,
                "current_success_rate": 0.75,
            },
        )

        assert result.processing_insights["processing_time_ms"] == 250.0
        assert result.processing_insights["current_success_rate"] == 0.75


class TestWeightedStrategySelection:
    """Tests for weighted strategy selection."""

    def test_weighted_selection(self, mock_pipeline_processor):
        """Test that weighted selection returns valid strategy."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
            config=MetaAwareBatchConfig(
                strategy_weights={
                    "checklist": 2.0,
                    "dialectical": 1.0,
                    "rule_based": 1.0,
                }
            ),
        )

        # Run multiple times to ensure it always returns valid strategy
        for _ in range(10):
            strategy = processor._weighted_strategy_selection()
            assert strategy in ["checklist", "dialectical", "rule_based"]

    def test_weighted_selection_zero_weights(self, mock_pipeline_processor):
        """Test selection with zero total weight."""
        processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline_processor,
            config=MetaAwareBatchConfig(
                strategy_weights={
                    "checklist": 0.0,
                    "dialectical": 0.0,
                },
                default_strategy="checklist",
            ),
        )

        strategy = processor._weighted_strategy_selection()
        assert strategy == "checklist"
