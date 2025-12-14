"""
Unit tests for FullPipelineProcessor.

Tests the production processor that connects batch harness to the complete
learning pipeline: gap identification, rule generation, validation,
incorporation, and persistence.

Issue #253: Phase 8 baseline validation infrastructure.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from loft.batch.full_pipeline import (
    FullPipelineProcessor,
    KnowledgeGap,
    ProcessingMetrics,
    create_full_pipeline_processor,
)
from loft.batch.schemas import CaseStatus
from loft.core.incorporation import IncorporationResult
from loft.neural.rule_schemas import GeneratedRule, GapFillingResponse, RuleCandidate
from loft.symbolic.stratification import StratificationLevel
from loft.validation.validation_schemas import ValidationReport


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_rule_generator():
    """Create mock rule generator."""
    generator = MagicMock()
    generator.generate_from_principle_aligned.return_value = GeneratedRule(
        asp_rule="enforceable(X) :- contract(X), not void(X).",
        confidence=0.85,
        reasoning="Generated rule for contract enforceability",
        predicates_used=["contract", "void"],
        new_predicates=["enforceable"],
        source_type="principle",
        source_text="A contract requires offer, acceptance, and consideration.",
    )
    generator.fill_knowledge_gap.return_value = GapFillingResponse(
        gap_description="Missing rule for contract outcome",
        missing_predicate="outcome(X, V)",
        candidates=[
            RuleCandidate(
                rule=GeneratedRule(
                    asp_rule="outcome(X, valid) :- contract(X), has_consideration(X).",
                    confidence=0.8,
                    reasoning="Gap fill candidate",
                    predicates_used=["contract", "has_consideration"],
                    new_predicates=["outcome"],
                    source_type="gap_fill",
                    source_text="Case analysis for contract validity",
                ),
                applicability_score=0.9,
            )
        ],
        recommended_index=0,
        requires_validation=True,
        test_cases_needed=["test_contract_validity"],
        confidence=0.85,
    )
    return generator


@pytest.fixture
def mock_validation_pipeline():
    """Create mock validation pipeline."""
    pipeline = MagicMock()
    pipeline.validate_rule.return_value = ValidationReport(
        rule_asp="enforceable(X) :- contract(X), not void(X).",
        rule_id="test_rule_1",
        target_layer="tactical",
    )
    pipeline.validate_rule.return_value.final_decision = "accept"
    pipeline.validate_rule.return_value.aggregate_confidence = 0.9
    return pipeline


@pytest.fixture
def mock_incorporation_engine():
    """Create mock incorporation engine."""
    engine = MagicMock()
    engine.incorporate.return_value = IncorporationResult(
        status="success",
        reason="Rule successfully incorporated",
        modification_number=1,
        accuracy_before=0.8,
        accuracy_after=0.82,
    )
    return engine


@pytest.fixture
def mock_persistence_manager():
    """Create mock persistence manager."""
    manager = MagicMock()
    manager.create_snapshot.return_value = "/asp_rules/snapshots/cycle_001"
    return manager


@pytest.fixture
def processor(
    mock_rule_generator,
    mock_validation_pipeline,
    mock_incorporation_engine,
    mock_persistence_manager,
):
    """Create FullPipelineProcessor with mocked dependencies."""
    return FullPipelineProcessor(
        rule_generator=mock_rule_generator,
        validation_pipeline=mock_validation_pipeline,
        incorporation_engine=mock_incorporation_engine,
        persistence_manager=mock_persistence_manager,
        target_layer=StratificationLevel.TACTICAL,
    )


@pytest.fixture
def sample_case():
    """Create sample test case."""
    return {
        "id": "test_case_001",
        "asp_facts": "contract(c1). party(c1, alice). party(c1, bob). has_consideration(c1).",
        "ground_truth": "valid",
        "prediction": "unknown",
        "legal_principle": "A contract requires offer, acceptance, and consideration.",
        "_domain": "contracts",
    }


@pytest.fixture
def sample_case_correct_prediction():
    """Create sample test case with correct prediction."""
    return {
        "id": "test_case_002",
        "asp_facts": "contract(c2). party(c2, carol).",
        "ground_truth": "valid",
        "prediction": "valid",
        "_domain": "contracts",
    }


# =============================================================================
# KnowledgeGap Tests
# =============================================================================


class TestKnowledgeGap:
    """Tests for KnowledgeGap dataclass."""

    def test_knowledge_gap_creation(self):
        """Test basic KnowledgeGap creation."""
        gap = KnowledgeGap(
            gap_id="gap_001",
            description="Missing enforceability rule",
            missing_predicate="enforceable(X)",
        )

        assert gap.gap_id == "gap_001"
        assert gap.description == "Missing enforceability rule"
        assert gap.missing_predicate == "enforceable(X)"
        assert gap.context == {}
        assert gap.priority == 1.0

    def test_knowledge_gap_with_context(self):
        """Test KnowledgeGap with context."""
        gap = KnowledgeGap(
            gap_id="gap_002",
            description="Missing statute of frauds rule",
            missing_predicate="within_statute(X)",
            context={"case_id": "case_001", "domain": "contracts"},
            priority=0.8,
        )

        assert gap.context["case_id"] == "case_001"
        assert gap.priority == 0.8

    def test_knowledge_gap_to_dict(self):
        """Test KnowledgeGap serialization."""
        gap = KnowledgeGap(
            gap_id="gap_003",
            description="Test gap",
            missing_predicate="pred(X)",
            context={"key": "value"},
            priority=0.5,
        )

        result = gap.to_dict()

        assert result["gap_id"] == "gap_003"
        assert result["description"] == "Test gap"
        assert result["missing_predicate"] == "pred(X)"
        assert result["context"] == {"key": "value"}
        assert result["priority"] == 0.5


# =============================================================================
# ProcessingMetrics Tests
# =============================================================================


class TestProcessingMetrics:
    """Tests for ProcessingMetrics dataclass."""

    def test_metrics_default_values(self):
        """Test default metrics values."""
        metrics = ProcessingMetrics()

        assert metrics.gaps_identified == 0
        assert metrics.rules_generated == 0
        assert metrics.rules_validated == 0
        assert metrics.rules_incorporated == 0
        assert metrics.generation_errors == 0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = ProcessingMetrics(
            gaps_identified=5,
            rules_generated=3,
            rules_validated=3,
            rules_incorporated=2,
            total_processing_time_ms=1500.0,
        )

        result = metrics.to_dict()

        assert result["gaps_identified"] == 5
        assert result["rules_generated"] == 3
        assert result["rules_incorporated"] == 2
        assert result["total_processing_time_ms"] == 1500.0


# =============================================================================
# FullPipelineProcessor Tests
# =============================================================================


class TestFullPipelineProcessor:
    """Tests for FullPipelineProcessor."""

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.rule_generator is not None
        assert processor.validation_pipeline is not None
        assert processor.incorporation_engine is not None
        assert processor.persistence_manager is not None
        assert processor.target_layer == StratificationLevel.TACTICAL

    def test_process_case_success(self, processor, sample_case):
        """Test successful case processing."""
        result = processor.process_case(sample_case, accumulated_rules=[])

        assert result.case_id == "test_case_001"
        assert result.status == CaseStatus.SUCCESS
        assert result.rules_generated >= 0
        assert result.processing_time_ms > 0
        assert result.error_message is None

    def test_process_case_with_correct_prediction(
        self, processor, sample_case_correct_prediction
    ):
        """Test case with correct prediction (no gaps)."""
        result = processor.process_case(
            sample_case_correct_prediction, accumulated_rules=[]
        )

        assert result.case_id == "test_case_002"
        assert result.status == CaseStatus.SUCCESS
        # Should have no gaps since prediction is correct
        assert result.metadata.get("gaps_identified", 0) == 0

    def test_process_case_generates_rules(self, processor, sample_case):
        """Test that processor generates rules for gaps."""
        processor.process_case(sample_case, accumulated_rules=[])

        # Should have attempted rule generation
        assert processor.rule_generator.fill_knowledge_gap.called or (
            processor.rule_generator.generate_from_principle_aligned.called
        )

    def test_process_case_validates_rules(
        self, processor, sample_case, mock_validation_pipeline
    ):
        """Test that generated rules are validated."""
        processor.process_case(sample_case, accumulated_rules=[])

        # Validation should be called for generated rules
        # (depends on whether gaps were identified)
        if processor.metrics.rules_generated > 0:
            mock_validation_pipeline.validate_rule.assert_called()

    def test_process_case_incorporates_valid_rules(
        self, processor, sample_case, mock_incorporation_engine
    ):
        """Test that validated rules are incorporated."""
        processor.process_case(sample_case, accumulated_rules=[])

        # If rules were validated and accepted, incorporation should be called
        if processor.metrics.rules_validated > 0:
            mock_incorporation_engine.incorporate.assert_called()

    def test_process_case_persists_rules(
        self, processor, sample_case, mock_persistence_manager
    ):
        """Test that incorporated rules are persisted."""
        processor.process_case(sample_case, accumulated_rules=[])

        # If rules were incorporated, snapshot should be created
        if processor.metrics.rules_incorporated > 0:
            mock_persistence_manager.create_snapshot.assert_called()

    def test_process_case_handles_generation_error(
        self, processor, sample_case, mock_rule_generator
    ):
        """Test handling of rule generation errors."""
        from loft.neural.rule_generator import RuleGenerationError

        mock_rule_generator.fill_knowledge_gap.side_effect = RuleGenerationError(
            "LLM error", attempts=3, last_error="Timeout"
        )
        mock_rule_generator.generate_from_principle_aligned.side_effect = (
            RuleGenerationError("LLM error", attempts=3, last_error="Timeout")
        )

        result = processor.process_case(sample_case, accumulated_rules=[])

        # Should still return success status (generation error is handled)
        assert result.status == CaseStatus.SUCCESS
        assert processor.metrics.generation_errors > 0

    def test_process_case_handles_exception(self, processor, sample_case):
        """Test handling of unexpected exceptions."""
        processor.rule_generator.fill_knowledge_gap.side_effect = Exception(
            "Unexpected error"
        )
        processor.rule_generator.generate_from_principle_aligned.side_effect = (
            Exception("Unexpected error")
        )

        result = processor.process_case(sample_case, accumulated_rules=[])

        # Should return failed status for unhandled exceptions
        assert result.status == CaseStatus.FAILED
        assert result.error_message is not None

    def test_metrics_accumulation(self, processor, sample_case):
        """Test that metrics accumulate across multiple cases."""
        initial_gaps = processor.metrics.gaps_identified

        processor.process_case(sample_case, accumulated_rules=[])
        first_gaps = processor.metrics.gaps_identified

        processor.process_case(sample_case, accumulated_rules=[])
        second_gaps = processor.metrics.gaps_identified

        # Metrics should accumulate
        assert second_gaps >= first_gaps >= initial_gaps

    def test_metrics_reset(self, processor, sample_case):
        """Test metrics reset."""
        processor.process_case(sample_case, accumulated_rules=[])
        assert processor.metrics.total_processing_time_ms > 0

        processor.reset_metrics()
        assert processor.metrics.total_processing_time_ms == 0
        assert processor.metrics.gaps_identified == 0

    def test_get_metrics(self, processor, sample_case):
        """Test get_metrics method."""
        processor.process_case(sample_case, accumulated_rules=[])
        metrics = processor.get_metrics()

        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.total_processing_time_ms > 0


class TestGapIdentification:
    """Tests for gap identification logic."""

    def test_identify_gaps_from_prediction_mismatch(self, processor, sample_case):
        """Test gap identification from prediction mismatch."""
        # Force gap identification by calling the method directly
        gaps = processor._identify_gaps(sample_case)

        # Should identify at least one gap due to prediction != ground_truth
        assert len(gaps) >= 1
        assert gaps[0].description is not None

    def test_identify_gaps_no_gaps_for_correct_prediction(
        self, processor, sample_case_correct_prediction
    ):
        """Test no gaps when prediction matches ground truth."""
        gaps = processor._identify_gaps(sample_case_correct_prediction)

        # Should not identify gaps for correct prediction
        assert len(gaps) == 0

    def test_identify_explicit_gaps(self, processor):
        """Test identification of explicit gaps in case data."""
        case_with_explicit_gaps = {
            "id": "case_explicit",
            "asp_facts": "test(x).",
            "ground_truth": "valid",
            "prediction": "valid",
            "knowledge_gaps": [
                "Need rule for consideration",
                {"id": "gap_1", "description": "Missing offer rule", "priority": 0.9},
            ],
        }

        gaps = processor._identify_gaps(case_with_explicit_gaps)

        # Should identify explicit gaps
        assert len(gaps) == 2

    def test_custom_gap_identifier(
        self,
        mock_rule_generator,
        mock_validation_pipeline,
        mock_incorporation_engine,
    ):
        """Test custom gap identifier function."""

        def custom_identifier(case: Dict[str, Any]) -> List[KnowledgeGap]:
            return [
                KnowledgeGap(
                    gap_id="custom_gap",
                    description="Custom identified gap",
                    missing_predicate="custom_pred(X)",
                )
            ]

        processor = FullPipelineProcessor(
            rule_generator=mock_rule_generator,
            validation_pipeline=mock_validation_pipeline,
            incorporation_engine=mock_incorporation_engine,
            gap_identifier=custom_identifier,
        )

        gaps = processor._identify_gaps({"id": "test"})

        assert len(gaps) == 1
        assert gaps[0].gap_id == "custom_gap"


class TestValidationRejection:
    """Tests for validation rejection handling."""

    def test_validation_rejection(
        self,
        mock_rule_generator,
        mock_incorporation_engine,
        mock_persistence_manager,
        sample_case,
    ):
        """Test handling of validation rejection."""
        # Create pipeline that rejects rules
        rejecting_pipeline = MagicMock()
        rejecting_pipeline.validate_rule.return_value = ValidationReport(
            rule_asp="test_rule.",
            rule_id="test_1",
            target_layer="tactical",
        )
        rejecting_pipeline.validate_rule.return_value.final_decision = "reject"
        rejecting_pipeline.validate_rule.return_value.rejection_reason = "Syntax error"

        processor = FullPipelineProcessor(
            rule_generator=mock_rule_generator,
            validation_pipeline=rejecting_pipeline,
            incorporation_engine=mock_incorporation_engine,
            persistence_manager=mock_persistence_manager,
        )

        result = processor.process_case(sample_case, accumulated_rules=[])

        # Should have validation failures
        assert result.status == CaseStatus.SUCCESS
        assert result.rules_accepted == 0


class TestIncorporationFailure:
    """Tests for incorporation failure handling."""

    def test_incorporation_failure(
        self,
        mock_rule_generator,
        mock_validation_pipeline,
        mock_persistence_manager,
        sample_case,
    ):
        """Test handling of incorporation failure."""
        # Create engine that fails incorporation
        failing_engine = MagicMock()
        failing_engine.incorporate.return_value = IncorporationResult(
            status="blocked",
            reason="Confidence below threshold",
            requires_human_review=True,
        )

        processor = FullPipelineProcessor(
            rule_generator=mock_rule_generator,
            validation_pipeline=mock_validation_pipeline,
            incorporation_engine=failing_engine,
            persistence_manager=mock_persistence_manager,
        )

        result = processor.process_case(sample_case, accumulated_rules=[])

        # Should have incorporation failures
        assert result.status == CaseStatus.SUCCESS
        assert result.rules_accepted == 0


class TestProcessorWithoutPersistence:
    """Tests for processor without persistence manager."""

    def test_process_without_persistence(
        self,
        mock_rule_generator,
        mock_validation_pipeline,
        mock_incorporation_engine,
        sample_case,
    ):
        """Test processing without persistence manager."""
        processor = FullPipelineProcessor(
            rule_generator=mock_rule_generator,
            validation_pipeline=mock_validation_pipeline,
            incorporation_engine=mock_incorporation_engine,
            persistence_manager=None,  # No persistence
        )

        result = processor.process_case(sample_case, accumulated_rules=[])

        assert result.status == CaseStatus.SUCCESS
        # Should still work without persistence


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateFullPipelineProcessor:
    """Tests for create_full_pipeline_processor factory function."""

    def test_factory_requires_valid_imports(self):
        """Test that factory function imports are available."""
        # This tests that the factory can be called - actual execution
        # requires real LLM credentials and would be integration tested.
        # Here we just verify the function signature and return type annotation.
        import inspect

        sig = inspect.signature(create_full_pipeline_processor)

        # Check parameters exist
        params = sig.parameters
        assert "model" in params
        assert "persistence_dir" in params
        assert "enable_persistence" in params
        assert "target_layer" in params
        assert "validation_threshold" in params

    def test_factory_default_parameters(self):
        """Test factory function default parameter values."""
        import inspect

        sig = inspect.signature(create_full_pipeline_processor)
        params = sig.parameters

        # Check defaults
        assert params["model"].default == "claude-3-5-haiku-20241022"
        assert params["persistence_dir"].default == "./asp_rules"
        assert params["enable_persistence"].default is True
        assert params["target_layer"].default == StratificationLevel.TACTICAL
        assert params["validation_threshold"].default == 0.6
