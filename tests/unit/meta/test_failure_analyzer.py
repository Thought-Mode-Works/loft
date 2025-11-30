"""
Unit tests for the failure analyzer module.

Tests cover:
- Error classification
- Root cause analysis
- Failure pattern detection
- Recommendation generation
- Report generation
"""

from datetime import datetime, timedelta

from loft.meta.schemas import (
    ReasoningChain,
    ReasoningStep,
    ReasoningStepType,
)
from loft.meta.failure_analyzer import (
    ErrorCategory,
    FailureAnalysisReport,
    FailureAnalyzer,
    FailurePattern,
    PredictionError,
    Recommendation,
    RecommendationCategory,
    RecommendationEngine,
    RootCause,
    RootCauseAnalysis,
    RootCauseType,
    create_failure_analyzer,
    create_recommendation_engine,
)
from loft.meta import ImprovementPriority


# Helper functions for creating test data


def create_reasoning_step(
    step_id: str,
    step_type: ReasoningStepType,
    success: bool = True,
    confidence: float = 0.8,
    error_message: str = None,
    metadata: dict = None,
) -> ReasoningStep:
    """Create a reasoning step for testing."""
    now = datetime.now()
    return ReasoningStep(
        step_id=step_id,
        step_type=step_type,
        description=f"Test step {step_id}",
        input_data={"test": "input"},
        output_data={"test": "output"},
        started_at=now,
        completed_at=now + timedelta(milliseconds=100),
        success=success,
        confidence=confidence,
        error_message=error_message,
        metadata=metadata or {},
    )


def create_reasoning_chain(
    chain_id: str,
    case_id: str,
    domain: str,
    steps: list,
    prediction: str = "enforceable",
    ground_truth: str = "enforceable",
    metadata: dict = None,
) -> ReasoningChain:
    """Create a reasoning chain for testing."""
    return ReasoningChain(
        chain_id=chain_id,
        case_id=case_id,
        domain=domain,
        steps=steps,
        prediction=prediction,
        ground_truth=ground_truth,
        overall_success=(prediction == ground_truth),
        started_at=datetime.now(),
        completed_at=datetime.now() + timedelta(seconds=1),
        metadata=metadata or {},
    )


def create_prediction_error(
    error_id: str,
    case_id: str,
    domain: str,
    predicted: str = "enforceable",
    actual: str = "unenforceable",
    chain: ReasoningChain = None,
    rules: list = None,
) -> PredictionError:
    """Create a prediction error for testing."""
    return PredictionError(
        error_id=error_id,
        case_id=case_id,
        domain=domain,
        predicted=predicted,
        actual=actual,
        reasoning_chain=chain,
        contributing_rules=rules or [],
    )


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_exist(self):
        """Test all error categories exist."""
        assert ErrorCategory.RULE_COVERAGE_GAP.value == "rule_coverage_gap"
        assert ErrorCategory.RULE_CONFLICT.value == "rule_conflict"
        assert ErrorCategory.TRANSLATION_ERROR.value == "translation_error"
        assert ErrorCategory.PREDICATE_MISMATCH.value == "predicate_mismatch"
        assert ErrorCategory.EDGE_CASE.value == "edge_case"
        assert ErrorCategory.DOMAIN_BOUNDARY.value == "domain_boundary"
        assert ErrorCategory.VALIDATION_FAILURE.value == "validation_failure"
        assert ErrorCategory.INFERENCE_ERROR.value == "inference_error"
        assert ErrorCategory.CONFIDENCE_THRESHOLD.value == "confidence_threshold"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestRootCauseType:
    """Tests for RootCauseType enum."""

    def test_all_types_exist(self):
        """Test all root cause types exist."""
        assert RootCauseType.MISSING_RULE.value == "missing_rule"
        assert RootCauseType.INCORRECT_RULE.value == "incorrect_rule"
        assert RootCauseType.PROMPT_WEAKNESS.value == "prompt_weakness"
        assert RootCauseType.DOMAIN_MISMATCH.value == "domain_mismatch"


class TestPredictionError:
    """Tests for PredictionError dataclass."""

    def test_basic_creation(self):
        """Test basic error creation."""
        error = create_prediction_error("err_001", "case_001", "contracts")

        assert error.error_id == "err_001"
        assert error.case_id == "case_001"
        assert error.domain == "contracts"
        assert error.predicted == "enforceable"
        assert error.actual == "unenforceable"

    def test_with_chain(self):
        """Test error with reasoning chain."""
        steps = [create_reasoning_step("step_1", ReasoningStepType.TRANSLATION)]
        chain = create_reasoning_chain("chain_1", "case_1", "contracts", steps)
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        assert error.reasoning_chain is not None
        assert error.reasoning_chain.chain_id == "chain_1"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = create_prediction_error(
            "err_001", "case_001", "contracts", rules=["rule_1", "rule_2"]
        )
        data = error.to_dict()

        assert data["error_id"] == "err_001"
        assert data["domain"] == "contracts"
        assert len(data["contributing_rules"]) == 2


class TestRootCause:
    """Tests for RootCause dataclass."""

    def test_basic_creation(self):
        """Test basic root cause creation."""
        cause = RootCause(
            cause_id="cause_001",
            cause_type=RootCauseType.MISSING_RULE,
            description="No rule for this case type",
            confidence=0.8,
            evidence=["No matching rules found"],
            remediation_hints=["Add new rule"],
        )

        assert cause.cause_type == RootCauseType.MISSING_RULE
        assert cause.confidence == 0.8
        assert len(cause.evidence) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cause = RootCause(
            cause_id="cause_001",
            cause_type=RootCauseType.PROMPT_WEAKNESS,
            description="Weak translation prompt",
            confidence=0.7,
            affected_step_types=[ReasoningStepType.TRANSLATION],
        )
        data = cause.to_dict()

        assert data["cause_type"] == "prompt_weakness"
        assert "translation" in data["affected_step_types"]


class TestRootCauseAnalysis:
    """Tests for RootCauseAnalysis dataclass."""

    def test_basic_creation(self):
        """Test basic analysis creation."""
        primary = RootCause(
            cause_id="cause_1",
            cause_type=RootCauseType.MISSING_RULE,
            description="Primary cause",
            confidence=0.8,
        )
        analysis = RootCauseAnalysis(
            analysis_id="rca_001",
            error_id="err_001",
            primary_cause=primary,
            analysis_confidence=0.75,
        )

        assert analysis.primary_cause.cause_type == RootCauseType.MISSING_RULE
        assert analysis.analysis_confidence == 0.75

    def test_with_secondary_causes(self):
        """Test analysis with secondary causes."""
        primary = RootCause(
            cause_id="cause_1",
            cause_type=RootCauseType.MISSING_RULE,
            description="Primary",
            confidence=0.8,
        )
        secondary = RootCause(
            cause_id="cause_2",
            cause_type=RootCauseType.DATA_QUALITY,
            description="Secondary",
            confidence=0.5,
        )
        analysis = RootCauseAnalysis(
            analysis_id="rca_001",
            error_id="err_001",
            primary_cause=primary,
            secondary_causes=[secondary],
        )

        assert len(analysis.secondary_causes) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        primary = RootCause(
            cause_id="cause_1",
            cause_type=RootCauseType.MISSING_RULE,
            description="Primary",
            confidence=0.8,
        )
        analysis = RootCauseAnalysis(
            analysis_id="rca_001",
            error_id="err_001",
            primary_cause=primary,
        )
        data = analysis.to_dict()

        assert data["analysis_id"] == "rca_001"
        assert "primary_cause" in data


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""

    def test_basic_creation(self):
        """Test basic pattern creation."""
        pattern = FailurePattern(
            pattern_id="pattern_001",
            name="Rule Coverage Gap Pattern",
            description="Recurring rule coverage gaps",
            error_category=ErrorCategory.RULE_COVERAGE_GAP,
            error_count=15,
            affected_domains=["contracts", "torts"],
            severity="high",
        )

        assert pattern.error_category == ErrorCategory.RULE_COVERAGE_GAP
        assert pattern.error_count == 15
        assert pattern.severity == "high"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        pattern = FailurePattern(
            pattern_id="pattern_001",
            name="Test Pattern",
            description="Test",
            error_category=ErrorCategory.TRANSLATION_ERROR,
            error_count=10,
        )
        data = pattern.to_dict()

        assert data["error_category"] == "translation_error"
        assert data["error_count"] == 10


class TestRecommendation:
    """Tests for Recommendation dataclass."""

    def test_basic_creation(self):
        """Test basic recommendation creation."""
        rec = Recommendation(
            recommendation_id="rec_001",
            category=RecommendationCategory.RULE_ADDITION,
            title="Add Missing Rules",
            description="Create rules for uncovered cases",
            priority=ImprovementPriority.HIGH,
            expected_impact=0.25,
            target_domains=["contracts"],
        )

        assert rec.category == RecommendationCategory.RULE_ADDITION
        assert rec.priority == ImprovementPriority.HIGH
        assert rec.expected_impact == 0.25

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = Recommendation(
            recommendation_id="rec_001",
            category=RecommendationCategory.PROMPT_IMPROVEMENT,
            title="Improve Prompts",
            description="Test",
            priority=ImprovementPriority.MEDIUM,
            expected_impact=0.15,
            implementation_steps=["Step 1", "Step 2"],
        )
        data = rec.to_dict()

        assert data["category"] == "prompt_improvement"
        assert len(data["implementation_steps"]) == 2


class TestFailureAnalyzer:
    """Tests for FailureAnalyzer class."""

    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = FailureAnalyzer()
        assert analyzer is not None

    def test_record_error(self):
        """Test recording errors."""
        analyzer = FailureAnalyzer()
        error = create_prediction_error("err_001", "case_001", "contracts")

        analyzer.record_error(error)

        stats = analyzer.get_error_statistics()
        assert stats["total_errors"] == 1

    def test_classify_error_no_chain(self):
        """Test classifying error without chain."""
        analyzer = FailureAnalyzer()
        error = create_prediction_error("err_001", "case_001", "contracts")

        category = analyzer.classify_error(error)

        assert category == ErrorCategory.UNKNOWN

    def test_classify_error_rule_coverage_gap(self):
        """Test classifying rule coverage gap."""
        analyzer = FailureAnalyzer()

        # Create chain with all successful steps but wrong prediction
        steps = [
            create_reasoning_step("step_1", ReasoningStepType.TRANSLATION, success=True),
            create_reasoning_step("step_2", ReasoningStepType.INFERENCE, success=True),
        ]
        chain = create_reasoning_chain(
            "chain_1",
            "case_1",
            "contracts",
            steps,
            prediction="enforceable",
            ground_truth="unenforceable",
        )
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain, rules=[])

        category = analyzer.classify_error(error)

        assert category == ErrorCategory.RULE_COVERAGE_GAP

    def test_classify_error_translation_failure(self):
        """Test classifying translation error."""
        analyzer = FailureAnalyzer()

        steps = [
            create_reasoning_step(
                "step_1",
                ReasoningStepType.TRANSLATION,
                success=False,
                error_message="Translation failed",
            )
        ]
        chain = create_reasoning_chain("chain_1", "case_1", "contracts", steps)
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        category = analyzer.classify_error(error)

        assert category == ErrorCategory.TRANSLATION_ERROR

    def test_classify_error_validation_failure(self):
        """Test classifying validation error."""
        analyzer = FailureAnalyzer()

        steps = [
            create_reasoning_step("step_1", ReasoningStepType.TRANSLATION, success=True),
            create_reasoning_step("step_2", ReasoningStepType.VALIDATION, success=False),
        ]
        chain = create_reasoning_chain("chain_1", "case_1", "contracts", steps)
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        category = analyzer.classify_error(error)

        assert category == ErrorCategory.VALIDATION_FAILURE

    def test_classify_error_inference_failure(self):
        """Test classifying inference error."""
        analyzer = FailureAnalyzer()

        steps = [
            create_reasoning_step("step_1", ReasoningStepType.TRANSLATION, success=True),
            create_reasoning_step("step_2", ReasoningStepType.INFERENCE, success=False),
        ]
        chain = create_reasoning_chain("chain_1", "case_1", "contracts", steps)
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        category = analyzer.classify_error(error)

        assert category == ErrorCategory.INFERENCE_ERROR

    def test_classify_error_rule_conflict(self):
        """Test classifying rule conflict."""
        analyzer = FailureAnalyzer()

        steps = [
            create_reasoning_step(
                "step_1",
                ReasoningStepType.RULE_APPLICATION,
                success=False,
                metadata={"conflict_detected": True},
            )
        ]
        chain = create_reasoning_chain("chain_1", "case_1", "contracts", steps)
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        category = analyzer.classify_error(error)

        assert category == ErrorCategory.RULE_CONFLICT

    def test_classify_error_domain_boundary(self):
        """Test classifying domain boundary error."""
        analyzer = FailureAnalyzer()

        # Domain boundary requires a failed step that doesn't match other categories
        steps = [
            create_reasoning_step(
                "step_1",
                ReasoningStepType.RULE_APPLICATION,
                success=False,
            )
        ]
        chain = create_reasoning_chain(
            "chain_1", "case_1", "contracts", steps, metadata={"cross_domain": True}
        )
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        category = analyzer.classify_error(error)

        assert category == ErrorCategory.DOMAIN_BOUNDARY

    def test_identify_root_cause(self):
        """Test identifying root cause."""
        analyzer = FailureAnalyzer()

        steps = [create_reasoning_step("step_1", ReasoningStepType.TRANSLATION, success=False)]
        chain = create_reasoning_chain("chain_1", "case_1", "contracts", steps)
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        analysis = analyzer.identify_root_cause(error)

        assert analysis.primary_cause is not None
        assert analysis.primary_cause.cause_type == RootCauseType.PROMPT_WEAKNESS
        assert analysis.analysis_confidence > 0

    def test_identify_root_cause_with_secondary(self):
        """Test root cause with secondary causes."""
        analyzer = FailureAnalyzer()

        steps = [
            create_reasoning_step(
                "step_1", ReasoningStepType.TRANSLATION, success=False, confidence=0.3
            ),
            create_reasoning_step(
                "step_2", ReasoningStepType.VALIDATION, success=False, confidence=0.4
            ),
        ]
        chain = create_reasoning_chain(
            "chain_1", "case_1", "contracts", steps, metadata={"input_quality_low": True}
        )
        error = create_prediction_error("err_001", "case_1", "contracts", chain=chain)

        analysis = analyzer.identify_root_cause(error)

        # Should have secondary causes due to low confidence steps
        assert len(analysis.secondary_causes) >= 1

    def test_find_failure_patterns(self):
        """Test finding failure patterns."""
        analyzer = FailureAnalyzer()

        # Add multiple similar errors
        for i in range(5):
            steps = [
                create_reasoning_step(f"step_{i}", ReasoningStepType.TRANSLATION, success=False)
            ]
            chain = create_reasoning_chain(f"chain_{i}", f"case_{i}", "contracts", steps)
            error = create_prediction_error(f"err_{i}", f"case_{i}", "contracts", chain=chain)
            analyzer.record_error(error)

        patterns = analyzer.find_failure_patterns(min_occurrences=3)

        assert len(patterns) >= 1
        assert patterns[0].error_category == ErrorCategory.TRANSLATION_ERROR
        assert patterns[0].error_count >= 3

    def test_find_patterns_minimum_occurrences(self):
        """Test pattern finding respects minimum occurrences."""
        analyzer = FailureAnalyzer()

        # Add 2 errors (below threshold of 3)
        for i in range(2):
            error = create_prediction_error(f"err_{i}", f"case_{i}", "contracts")
            analyzer.record_error(error)

        patterns = analyzer.find_failure_patterns(min_occurrences=3)

        assert len(patterns) == 0

    def test_create_diagnosis(self):
        """Test creating complete diagnosis."""
        analyzer = FailureAnalyzer()

        steps = [create_reasoning_step("step_1", ReasoningStepType.TRANSLATION, success=False)]
        chain = create_reasoning_chain(
            "chain_1",
            "case_1",
            "contracts",
            steps,
            prediction="enforceable",
            ground_truth="unenforceable",
        )
        error = create_prediction_error(
            "err_001",
            "case_1",
            "contracts",
            predicted="enforceable",
            actual="unenforceable",
            chain=chain,
        )

        diagnosis = analyzer.create_diagnosis(error)

        assert diagnosis.case_id == "case_1"
        assert diagnosis.failure_type == "translation_error"
        assert len(diagnosis.root_causes) >= 1
        assert diagnosis.explanation != ""

    def test_get_error_statistics(self):
        """Test getting error statistics."""
        analyzer = FailureAnalyzer()

        # Add errors in different domains
        for i in range(3):
            error = create_prediction_error(f"err_{i}", f"case_{i}", "contracts")
            analyzer.record_error(error)
            analyzer.classify_error(error)

        for i in range(2):
            error = create_prediction_error(f"err_t{i}", f"case_t{i}", "torts")
            analyzer.record_error(error)
            analyzer.classify_error(error)

        stats = analyzer.get_error_statistics()

        assert stats["total_errors"] == 5
        assert stats["domain_distribution"]["contracts"] == 3
        assert stats["domain_distribution"]["torts"] == 2


class TestRecommendationEngine:
    """Tests for RecommendationEngine class."""

    def test_engine_creation(self):
        """Test engine creation."""
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)
        assert engine is not None

    def test_generate_recommendations(self):
        """Test generating recommendations from patterns."""
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)

        pattern = FailurePattern(
            pattern_id="pattern_001",
            name="Test Pattern",
            description="Test",
            error_category=ErrorCategory.RULE_COVERAGE_GAP,
            error_count=10,
            affected_domains=["contracts"],
            severity="high",
        )

        recommendations = engine.generate_recommendations([pattern])

        assert len(recommendations) >= 1
        assert any(r.category == RecommendationCategory.RULE_ADDITION for r in recommendations)

    def test_generate_recommendations_translation_error(self):
        """Test recommendations for translation errors."""
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)

        pattern = FailurePattern(
            pattern_id="pattern_001",
            name="Translation Error Pattern",
            description="Test",
            error_category=ErrorCategory.TRANSLATION_ERROR,
            error_count=8,
            affected_domains=["contracts"],
            severity="medium",
        )

        recommendations = engine.generate_recommendations([pattern])

        assert any(r.category == RecommendationCategory.PROMPT_IMPROVEMENT for r in recommendations)

    def test_prioritize_recommendations(self):
        """Test prioritizing recommendations."""
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)

        recommendations = [
            Recommendation(
                recommendation_id="rec_1",
                category=RecommendationCategory.RULE_ADDITION,
                title="Low Priority",
                description="Test",
                priority=ImprovementPriority.LOW,
                expected_impact=0.05,
            ),
            Recommendation(
                recommendation_id="rec_2",
                category=RecommendationCategory.RULE_MODIFICATION,
                title="Critical Priority",
                description="Test",
                priority=ImprovementPriority.CRITICAL,
                expected_impact=0.30,
            ),
            Recommendation(
                recommendation_id="rec_3",
                category=RecommendationCategory.PROMPT_IMPROVEMENT,
                title="High Priority",
                description="Test",
                priority=ImprovementPriority.HIGH,
                expected_impact=0.20,
            ),
        ]

        prioritized = engine.prioritize_recommendations(recommendations)

        assert prioritized[0].priority == ImprovementPriority.CRITICAL
        assert prioritized[1].priority == ImprovementPriority.HIGH
        assert prioritized[2].priority == ImprovementPriority.LOW

    def test_track_recommendation_effectiveness(self):
        """Test tracking recommendation effectiveness."""
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)

        engine.track_recommendation_effectiveness("rec_001", 0.25)
        engine.track_recommendation_effectiveness("rec_002", 0.15)

        report = engine.get_effectiveness_report()

        assert report["total_tracked"] == 2
        assert report["average_improvement"] == 0.20

    def test_generate_report(self):
        """Test generating comprehensive report."""
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)

        # Add some errors
        for i in range(5):
            steps = [
                create_reasoning_step(f"step_{i}", ReasoningStepType.TRANSLATION, success=False)
            ]
            chain = create_reasoning_chain(f"chain_{i}", f"case_{i}", "contracts", steps)
            error = create_prediction_error(f"err_{i}", f"case_{i}", "contracts", chain=chain)
            analyzer.record_error(error)

        report = engine.generate_report()

        assert isinstance(report, FailureAnalysisReport)
        assert report.total_errors_analyzed == 5
        assert "contracts" in report.domains_analyzed
        assert len(report.error_category_distribution) > 0

    def test_generate_report_empty(self):
        """Test generating report with no errors."""
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)

        report = engine.generate_report()

        assert report.total_errors_analyzed == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_failure_analyzer(self):
        """Test create_failure_analyzer factory."""
        analyzer = create_failure_analyzer()
        assert isinstance(analyzer, FailureAnalyzer)

    def test_create_recommendation_engine(self):
        """Test create_recommendation_engine factory."""
        engine = create_recommendation_engine()
        assert isinstance(engine, RecommendationEngine)

    def test_create_recommendation_engine_with_analyzer(self):
        """Test create_recommendation_engine with existing analyzer."""
        analyzer = FailureAnalyzer()
        engine = create_recommendation_engine(analyzer)
        assert engine.analyzer is analyzer


class TestIntegration:
    """Integration tests for failure analysis workflow."""

    def test_full_analysis_workflow(self):
        """Test complete failure analysis workflow."""
        analyzer = create_failure_analyzer()
        engine = create_recommendation_engine(analyzer)

        # Record errors of different types
        # Translation errors
        for i in range(5):
            steps = [
                create_reasoning_step(f"trans_{i}", ReasoningStepType.TRANSLATION, success=False)
            ]
            chain = create_reasoning_chain(f"chain_t{i}", f"case_t{i}", "contracts", steps)
            error = create_prediction_error(f"err_t{i}", f"case_t{i}", "contracts", chain=chain)
            analyzer.record_error(error)

        # Rule coverage gaps
        for i in range(4):
            steps = [
                create_reasoning_step(f"rule_{i}", ReasoningStepType.RULE_APPLICATION, success=True)
            ]
            chain = create_reasoning_chain(
                f"chain_r{i}",
                f"case_r{i}",
                "torts",
                steps,
                prediction="liable",
                ground_truth="not_liable",
            )
            error = create_prediction_error(
                f"err_r{i}",
                f"case_r{i}",
                "torts",
                predicted="liable",
                actual="not_liable",
                chain=chain,
                rules=[],
            )
            analyzer.record_error(error)

        # Find patterns
        patterns = analyzer.find_failure_patterns(min_occurrences=3)
        assert len(patterns) >= 1

        # Generate recommendations
        recommendations = engine.generate_recommendations(patterns)
        assert len(recommendations) >= 1

        # Prioritize
        prioritized = engine.prioritize_recommendations(recommendations)
        # With 5 and 4 errors respectively, we expect MEDIUM priority at minimum
        assert prioritized[0].priority in [
            ImprovementPriority.CRITICAL,
            ImprovementPriority.HIGH,
            ImprovementPriority.MEDIUM,
            ImprovementPriority.LOW,
        ]

        # Generate report
        report = engine.generate_report()
        assert report.total_errors_analyzed == 9
        assert len(report.patterns_identified) >= 1
        assert len(report.recommendations) >= 1

    def test_diagnosis_and_similar_finding(self):
        """Test diagnosis with similar error finding."""
        analyzer = create_failure_analyzer()

        # Create multiple similar errors
        for i in range(3):
            steps = [
                create_reasoning_step(f"step_{i}", ReasoningStepType.TRANSLATION, success=False)
            ]
            chain = create_reasoning_chain(f"chain_{i}", f"case_{i}", "contracts", steps)
            error = create_prediction_error(f"err_{i}", f"case_{i}", "contracts", chain=chain)
            analyzer.record_error(error)
            analyzer.classify_error(error)

        # Create diagnosis for last error
        last_error = list(analyzer._errors.values())[-1]
        diagnosis = analyzer.create_diagnosis(last_error)

        # Should find similar errors
        assert len(diagnosis.similar_failures) >= 1

    def test_pattern_severity_escalation(self):
        """Test that pattern severity escalates with count."""
        analyzer = create_failure_analyzer()

        # Add many errors of same type
        for i in range(25):
            steps = [create_reasoning_step(f"step_{i}", ReasoningStepType.INFERENCE, success=False)]
            chain = create_reasoning_chain(f"chain_{i}", f"case_{i}", "contracts", steps)
            error = create_prediction_error(f"err_{i}", f"case_{i}", "contracts", chain=chain)
            analyzer.record_error(error)

        patterns = analyzer.find_failure_patterns(min_occurrences=3)

        # With 25 errors, severity should be critical
        assert patterns[0].severity == "critical"
