"""Tests for meta-reasoning schemas."""

from datetime import datetime, timedelta

import pytest

from loft.meta.schemas import (
    Bottleneck,
    BottleneckReport,
    FailureDiagnosis,
    Improvement,
    ImprovementPriority,
    ImprovementType,
    ObservationSummary,
    PatternType,
    ReasoningChain,
    ReasoningPattern,
    ReasoningStep,
    ReasoningStepType,
)


class TestReasoningStepType:
    """Tests for ReasoningStepType enum."""

    def test_all_step_types_exist(self):
        """Verify all expected step types are defined."""
        expected_types = [
            "TRANSLATION",
            "RULE_APPLICATION",
            "RULE_GENERATION",
            "VALIDATION",
            "GROUNDING",
            "INFERENCE",
            "CONSENSUS",
            "DIALECTICAL",
        ]
        for type_name in expected_types:
            assert hasattr(ReasoningStepType, type_name)

    def test_step_type_values(self):
        """Test step type string values."""
        assert ReasoningStepType.TRANSLATION.value == "translation"
        assert ReasoningStepType.INFERENCE.value == "inference"
        assert ReasoningStepType.VALIDATION.value == "validation"


class TestPatternType:
    """Tests for PatternType enum."""

    def test_all_pattern_types_exist(self):
        """Verify all expected pattern types are defined."""
        expected_types = [
            "SUCCESS",
            "FAILURE",
            "BOTTLENECK",
            "RECURRING",
            "DOMAIN_SPECIFIC",
        ]
        for type_name in expected_types:
            assert hasattr(PatternType, type_name)


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""

    @pytest.fixture
    def sample_step(self):
        """Create a sample reasoning step."""
        started = datetime.now()
        completed = started + timedelta(milliseconds=150)
        return ReasoningStep(
            step_id="step_001",
            step_type=ReasoningStepType.TRANSLATION,
            description="Translate NL to ASP",
            input_data={"text": "Alice has a contract"},
            output_data={"asp": "contract(alice)."},
            started_at=started,
            completed_at=completed,
            success=True,
            confidence=0.85,
        )

    def test_step_creation(self, sample_step):
        """Test creating a reasoning step."""
        assert sample_step.step_id == "step_001"
        assert sample_step.step_type == ReasoningStepType.TRANSLATION
        assert sample_step.success is True
        assert sample_step.confidence == 0.85

    def test_duration_calculation(self, sample_step):
        """Test duration_ms property."""
        duration = sample_step.duration_ms
        assert 140 <= duration <= 160  # Allow for small timing variations

    def test_to_dict(self, sample_step):
        """Test conversion to dictionary."""
        data = sample_step.to_dict()
        assert data["step_id"] == "step_001"
        assert data["step_type"] == "translation"
        assert data["success"] is True
        assert "duration_ms" in data

    def test_from_dict(self, sample_step):
        """Test creation from dictionary."""
        data = sample_step.to_dict()
        restored = ReasoningStep.from_dict(data)
        assert restored.step_id == sample_step.step_id
        assert restored.step_type == sample_step.step_type
        assert restored.success == sample_step.success

    def test_step_with_error(self):
        """Test step with error message."""
        step = ReasoningStep(
            step_id="step_err",
            step_type=ReasoningStepType.VALIDATION,
            description="Validate rule",
            input_data={},
            output_data={},
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=False,
            error_message="Rule conflicts with existing rules",
        )
        assert step.success is False
        assert step.error_message == "Rule conflicts with existing rules"


class TestReasoningChain:
    """Tests for ReasoningChain dataclass."""

    @pytest.fixture
    def sample_chain(self):
        """Create a sample reasoning chain."""
        started = datetime.now()
        steps = []
        for i in range(3):
            step_started = started + timedelta(milliseconds=i * 100)
            step_completed = step_started + timedelta(milliseconds=50)
            steps.append(
                ReasoningStep(
                    step_id=f"step_{i}",
                    step_type=ReasoningStepType.TRANSLATION,
                    description=f"Step {i}",
                    input_data={},
                    output_data={},
                    started_at=step_started,
                    completed_at=step_completed,
                    success=True,
                    confidence=0.8,
                )
            )

        return ReasoningChain(
            chain_id="chain_001",
            case_id="case_001",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="enforceable",
            overall_success=True,
        )

    def test_chain_creation(self, sample_chain):
        """Test creating a reasoning chain."""
        assert sample_chain.chain_id == "chain_001"
        assert sample_chain.domain == "contracts"
        assert len(sample_chain.steps) == 3
        assert sample_chain.overall_success is True

    def test_total_duration(self, sample_chain):
        """Test total_duration_ms property."""
        duration = sample_chain.total_duration_ms
        # 3 steps * 50ms each = 150ms
        assert 140 <= duration <= 160

    def test_success_rate(self, sample_chain):
        """Test success_rate property."""
        assert sample_chain.success_rate == 1.0

    def test_success_rate_partial(self):
        """Test success_rate with some failed steps."""
        started = datetime.now()
        steps = [
            ReasoningStep(
                step_id="s1",
                step_type=ReasoningStepType.TRANSLATION,
                description="Step 1",
                input_data={},
                output_data={},
                started_at=started,
                completed_at=started,
                success=True,
            ),
            ReasoningStep(
                step_id="s2",
                step_type=ReasoningStepType.VALIDATION,
                description="Step 2",
                input_data={},
                output_data={},
                started_at=started,
                completed_at=started,
                success=False,
            ),
        ]
        chain = ReasoningChain(
            chain_id="c1",
            case_id="case1",
            domain="torts",
            steps=steps,
            overall_success=False,
        )
        assert chain.success_rate == 0.5

    def test_failed_steps(self):
        """Test failed_steps property."""
        started = datetime.now()
        steps = [
            ReasoningStep(
                step_id="s1",
                step_type=ReasoningStepType.TRANSLATION,
                description="Step 1",
                input_data={},
                output_data={},
                started_at=started,
                completed_at=started,
                success=True,
            ),
            ReasoningStep(
                step_id="s2",
                step_type=ReasoningStepType.VALIDATION,
                description="Step 2",
                input_data={},
                output_data={},
                started_at=started,
                completed_at=started,
                success=False,
            ),
        ]
        chain = ReasoningChain(
            chain_id="c1",
            case_id="case1",
            domain="torts",
            steps=steps,
            overall_success=False,
        )
        failed = chain.failed_steps
        assert len(failed) == 1
        assert failed[0].step_id == "s2"

    def test_to_dict(self, sample_chain):
        """Test conversion to dictionary."""
        data = sample_chain.to_dict()
        assert data["chain_id"] == "chain_001"
        assert data["domain"] == "contracts"
        assert len(data["steps"]) == 3
        assert "total_duration_ms" in data
        assert "success_rate" in data


class TestReasoningPattern:
    """Tests for ReasoningPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a reasoning pattern."""
        pattern = ReasoningPattern(
            pattern_id="pattern_001",
            pattern_type=PatternType.SUCCESS,
            name="translation_success",
            description="Translation steps have high success rate",
            frequency=50,
            associated_step_types=[ReasoningStepType.TRANSLATION],
            success_correlation=0.8,
            domains=["contracts", "torts"],
        )
        assert pattern.pattern_id == "pattern_001"
        assert pattern.pattern_type == PatternType.SUCCESS
        assert pattern.frequency == 50
        assert pattern.success_correlation == 0.8

    def test_pattern_to_dict(self):
        """Test pattern conversion to dictionary."""
        pattern = ReasoningPattern(
            pattern_id="pattern_002",
            pattern_type=PatternType.FAILURE,
            name="validation_failure",
            description="Validation frequently fails",
            frequency=25,
            associated_step_types=[ReasoningStepType.VALIDATION],
            success_correlation=-0.6,
        )
        data = pattern.to_dict()
        assert data["pattern_type"] == "failure"
        assert data["associated_step_types"] == ["validation"]


class TestBottleneck:
    """Tests for Bottleneck dataclass."""

    def test_bottleneck_creation(self):
        """Test creating a bottleneck."""
        bottleneck = Bottleneck(
            bottleneck_id="bn_001",
            step_type=ReasoningStepType.INFERENCE,
            description="ASP inference is slow",
            avg_duration_ms=2500.0,
            max_duration_ms=5000.0,
            occurrence_count=100,
            percentage_of_total_time=35.0,
            affected_domains=["contracts"],
            potential_causes=["Large rule base", "Complex queries"],
            severity="high",
        )
        assert bottleneck.avg_duration_ms == 2500.0
        assert bottleneck.severity == "high"

    def test_bottleneck_to_dict(self):
        """Test bottleneck conversion to dictionary."""
        bottleneck = Bottleneck(
            bottleneck_id="bn_002",
            step_type=ReasoningStepType.TRANSLATION,
            description="Translation is slow",
            avg_duration_ms=1000.0,
            max_duration_ms=2000.0,
            occurrence_count=50,
            percentage_of_total_time=20.0,
        )
        data = bottleneck.to_dict()
        assert data["step_type"] == "translation"
        assert data["avg_duration_ms"] == 1000.0


class TestBottleneckReport:
    """Tests for BottleneckReport dataclass."""

    def test_report_creation(self):
        """Test creating a bottleneck report."""
        bottleneck = Bottleneck(
            bottleneck_id="bn_001",
            step_type=ReasoningStepType.INFERENCE,
            description="Inference bottleneck",
            avg_duration_ms=2000.0,
            max_duration_ms=4000.0,
            occurrence_count=50,
            percentage_of_total_time=30.0,
        )
        report = BottleneckReport(
            report_id="report_001",
            generated_at=datetime.now(),
            total_chains_analyzed=100,
            total_steps_analyzed=500,
            bottlenecks=[bottleneck],
            recommendations=["Optimize inference"],
        )
        assert len(report.bottlenecks) == 1
        assert report.total_chains_analyzed == 100


class TestFailureDiagnosis:
    """Tests for FailureDiagnosis dataclass."""

    def test_diagnosis_creation(self):
        """Test creating a failure diagnosis."""
        diagnosis = FailureDiagnosis(
            diagnosis_id="diag_001",
            chain_id="chain_001",
            case_id="case_001",
            prediction="enforceable",
            ground_truth="unenforceable",
            primary_failure_step="step_003",
            failure_type="rule_gap",
            root_causes=["Missing rule for exception"],
            confidence=0.75,
            explanation="The system lacked rules for this exception case.",
        )
        assert diagnosis.prediction != diagnosis.ground_truth
        assert diagnosis.failure_type == "rule_gap"
        assert diagnosis.confidence == 0.75


class TestImprovement:
    """Tests for Improvement dataclass."""

    def test_improvement_creation(self):
        """Test creating an improvement suggestion."""
        improvement = Improvement(
            improvement_id="imp_001",
            improvement_type=ImprovementType.PROMPT_REFINEMENT,
            priority=ImprovementPriority.HIGH,
            title="Improve Translation Prompts",
            description="Refine prompts to reduce translation errors",
            expected_impact="Reduce translation failures by 20%",
            target_component="translation",
            estimated_effort="medium",
            implementation_steps=[
                "Analyze failed translations",
                "Identify common error patterns",
                "Create improved prompts",
            ],
        )
        assert improvement.priority == ImprovementPriority.HIGH
        assert len(improvement.implementation_steps) == 3

    def test_improvement_to_dict(self):
        """Test improvement conversion to dictionary."""
        improvement = Improvement(
            improvement_id="imp_002",
            improvement_type=ImprovementType.RULE_MODIFICATION,
            priority=ImprovementPriority.CRITICAL,
            title="Fix Rule Conflict",
            description="Resolve conflicting rules",
            expected_impact="Fix 5 failure cases",
            target_component="rule_base",
        )
        data = improvement.to_dict()
        assert data["improvement_type"] == "rule_modification"
        assert data["priority"] == "critical"


class TestObservationSummary:
    """Tests for ObservationSummary dataclass."""

    def test_summary_creation(self):
        """Test creating an observation summary."""
        now = datetime.now()
        summary = ObservationSummary(
            summary_id="sum_001",
            generated_at=now,
            observation_period_start=now - timedelta(hours=1),
            observation_period_end=now,
            total_chains_observed=100,
            total_steps_observed=500,
            success_rate=0.85,
            avg_chain_duration_ms=1500.0,
            patterns_identified=5,
            bottlenecks_identified=2,
            domains_observed=["contracts", "torts"],
            step_type_distribution={"translation": 200, "inference": 150},
            domain_success_rates={"contracts": 0.9, "torts": 0.8},
        )
        assert summary.success_rate == 0.85
        assert len(summary.domains_observed) == 2

    def test_summary_to_dict(self):
        """Test summary conversion to dictionary."""
        now = datetime.now()
        summary = ObservationSummary(
            summary_id="sum_002",
            generated_at=now,
            observation_period_start=now,
            observation_period_end=now,
            total_chains_observed=50,
            total_steps_observed=250,
            success_rate=0.75,
            avg_chain_duration_ms=1000.0,
            patterns_identified=3,
            bottlenecks_identified=1,
        )
        data = summary.to_dict()
        assert data["success_rate"] == 0.75
        assert data["patterns_identified"] == 3
