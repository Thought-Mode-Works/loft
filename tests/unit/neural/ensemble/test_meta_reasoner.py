"""Unit tests for MetaReasonerLLM - Reasoning about reasoning.

Tests for the Meta-Reasoner LLM implementation (Issue #191, Phase 6 Neural Ensemble).
"""

import json
import pytest
from unittest.mock import Mock

from loft.neural.ensemble.meta_reasoner import (
    # Data classes
    FailureRecord,
    Insight,
    StrategyChange,
    ImprovementReport,
    MetaReasoningResult,
    # Config
    MetaReasonerConfig,
    # Main class
    MetaReasonerLLM,
    MetaReasoner,
    # Strategy classes
    MetaReasoningStrategyType,
    AnalyticalStrategy,
    ReflectiveStrategy,
    ComparativeStrategy,
    DiagnosticStrategy,
    create_meta_reasoning_strategy,
    # Enums
    InsightType,
    FailureCategory,
    # Exceptions
    MetaReasoningError,
)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestFailureRecord:
    """Tests for FailureRecord dataclass."""

    def test_failure_record_creation(self):
        """Test basic FailureRecord creation."""
        failure = FailureRecord(
            failure_id="F001",
            category=FailureCategory.SYNTAX_ERROR,
            error_message="Invalid ASP syntax",
            domain="contracts",
            strategy_used="few_shot",
        )

        assert failure.failure_id == "F001"
        assert failure.category == FailureCategory.SYNTAX_ERROR
        assert failure.domain == "contracts"

    def test_failure_record_defaults(self):
        """Test FailureRecord default values."""
        failure = FailureRecord(failure_id="F002")

        assert failure.category == FailureCategory.UNKNOWN
        assert failure.error_message is None
        assert failure.domain == "unknown"
        assert failure.strategy_used == "unknown"
        assert failure.confidence_before == 0.0

    def test_failure_record_with_context(self):
        """Test FailureRecord with context dictionary."""
        failure = FailureRecord(
            failure_id="F003",
            context={"input_length": 500, "rule_type": "constraint"},
            input_summary="Long contract case",
        )

        assert failure.context["input_length"] == 500
        assert failure.input_summary == "Long contract case"


class TestInsight:
    """Tests for Insight dataclass."""

    def test_insight_creation(self):
        """Test basic Insight creation."""
        insight = Insight(
            insight_id="INS001",
            insight_type=InsightType.PATTERN_IDENTIFIED,
            title="Recurring syntax errors",
            description="Syntax errors occur frequently in constraint rules",
            confidence=0.85,
        )

        assert insight.insight_id == "INS001"
        assert insight.insight_type == InsightType.PATTERN_IDENTIFIED
        assert insight.confidence == 0.85

    def test_insight_defaults(self):
        """Test Insight default values."""
        insight = Insight(
            insight_id="INS002",
            insight_type=InsightType.ROOT_CAUSE,
            title="Root cause",
            description="Description",
        )

        assert insight.confidence == 0.0
        assert insight.actionable is True
        assert insight.priority == "medium"
        assert insight.evidence == []
        assert insight.related_failures == []

    def test_insight_to_dict(self):
        """Test Insight serialization."""
        insight = Insight(
            insight_id="INS003",
            insight_type=InsightType.STRATEGY_RECOMMENDATION,
            title="Use chain-of-thought",
            description="CoT improves reasoning",
            confidence=0.9,
            priority="high",
        )

        data = insight.to_dict()

        assert data["insight_id"] == "INS003"
        assert data["insight_type"] == "strategy_recommendation"
        assert data["priority"] == "high"
        assert data["confidence"] == 0.9

    def test_insight_with_evidence(self):
        """Test Insight with evidence list."""
        insight = Insight(
            insight_id="INS004",
            insight_type=InsightType.KNOWLEDGE_GAP,
            title="Missing domain knowledge",
            description="Gap in tort law rules",
            evidence=["Failed on 10 tort cases", "No rules for negligence"],
            related_failures=["F001", "F002"],
        )

        assert len(insight.evidence) == 2
        assert len(insight.related_failures) == 2


class TestStrategyChange:
    """Tests for StrategyChange dataclass."""

    def test_strategy_change_creation(self):
        """Test basic StrategyChange creation."""
        change = StrategyChange(
            change_id="SC001",
            current_strategy="few_shot",
            recommended_strategy="chain_of_thought",
            confidence=0.8,
            rationale="CoT handles complex cases better",
        )

        assert change.current_strategy == "few_shot"
        assert change.recommended_strategy == "chain_of_thought"
        assert change.confidence == 0.8

    def test_strategy_change_with_conditions(self):
        """Test StrategyChange with context conditions."""
        change = StrategyChange(
            change_id="SC002",
            current_strategy="prompt_optimization",
            recommended_strategy="self_consistency",
            context_conditions={"domain": "contracts", "complexity": "high"},
            expected_improvement="15% accuracy gain",
        )

        assert change.context_conditions["domain"] == "contracts"
        assert "accuracy" in change.expected_improvement.lower()

    def test_strategy_change_to_dict(self):
        """Test StrategyChange serialization."""
        change = StrategyChange(
            change_id="SC003",
            current_strategy="analytical",
            recommended_strategy="diagnostic",
            confidence=0.75,
        )

        data = change.to_dict()

        assert data["change_id"] == "SC003"
        assert data["confidence"] == 0.75


class TestImprovementReport:
    """Tests for ImprovementReport dataclass."""

    def test_improvement_report_creation(self):
        """Test basic ImprovementReport creation."""
        report = ImprovementReport(
            report_id="IR001",
            overall_trajectory="improving",
            next_goals=["Improve constraint handling", "Add more examples"],
        )

        assert report.report_id == "IR001"
        assert report.overall_trajectory == "improving"
        assert len(report.next_goals) == 2

    def test_improvement_report_defaults(self):
        """Test ImprovementReport default values."""
        report = ImprovementReport(report_id="IR002")

        assert report.overall_trajectory == "stable"
        assert report.insights == []
        assert report.strategy_changes == []
        assert report.prompt_improvements == []

    def test_improvement_report_to_dict(self):
        """Test ImprovementReport serialization."""
        insight = Insight(
            insight_id="INS001",
            insight_type=InsightType.PATTERN_IDENTIFIED,
            title="Test",
            description="Test description",
        )

        report = ImprovementReport(
            report_id="IR003",
            overall_trajectory="declining",
            insights=[insight],
            metrics_summary={"accuracy": 0.75, "consistency": 0.8},
        )

        data = report.to_dict()

        assert data["overall_trajectory"] == "declining"
        assert len(data["insights"]) == 1
        assert data["metrics_summary"]["accuracy"] == 0.75


class TestMetaReasoningResult:
    """Tests for MetaReasoningResult dataclass."""

    def test_meta_reasoning_result_creation(self):
        """Test basic MetaReasoningResult creation."""
        result = MetaReasoningResult(
            result_id="MR001",
            analysis_type="failure_analysis",
            confidence=0.85,
            processing_time_ms=1500.0,
        )

        assert result.result_id == "MR001"
        assert result.confidence == 0.85
        assert result.processing_time_ms == 1500.0

    def test_meta_reasoning_result_with_insights(self):
        """Test MetaReasoningResult with insights."""
        insight = Insight(
            insight_id="INS001",
            insight_type=InsightType.ROOT_CAUSE,
            title="Test",
            description="Test",
        )

        result = MetaReasoningResult(
            result_id="MR002",
            insights=[insight],
            reasoning_trace=["Step 1", "Step 2"],
        )

        assert len(result.insights) == 1
        assert len(result.reasoning_trace) == 2

    def test_meta_reasoning_result_to_dict(self):
        """Test MetaReasoningResult serialization."""
        result = MetaReasoningResult(
            result_id="MR003",
            analysis_type="strategy_evaluation",
            confidence=0.9,
        )

        data = result.to_dict()

        assert data["result_id"] == "MR003"
        assert data["analysis_type"] == "strategy_evaluation"


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_meta_reasoning_strategy_type_values(self):
        """Test MetaReasoningStrategyType enum values."""
        assert MetaReasoningStrategyType.ANALYTICAL.value == "analytical"
        assert MetaReasoningStrategyType.REFLECTIVE.value == "reflective"
        assert MetaReasoningStrategyType.COMPARATIVE.value == "comparative"
        assert MetaReasoningStrategyType.DIAGNOSTIC.value == "diagnostic"

    def test_insight_type_values(self):
        """Test InsightType enum values."""
        assert InsightType.PATTERN_IDENTIFIED.value == "pattern_identified"
        assert InsightType.ROOT_CAUSE.value == "root_cause"
        assert InsightType.STRATEGY_RECOMMENDATION.value == "strategy_recommendation"
        assert InsightType.PROMPT_IMPROVEMENT.value == "prompt_improvement"
        assert InsightType.CONFIDENCE_CALIBRATION.value == "confidence_calibration"
        assert InsightType.KNOWLEDGE_GAP.value == "knowledge_gap"

    def test_failure_category_values(self):
        """Test FailureCategory enum values."""
        assert FailureCategory.SYNTAX_ERROR.value == "syntax_error"
        assert FailureCategory.SEMANTIC_ERROR.value == "semantic_error"
        assert FailureCategory.GROUNDING_ERROR.value == "grounding_error"
        assert FailureCategory.VALIDATION_ERROR.value == "validation_error"
        assert FailureCategory.TIMEOUT_ERROR.value == "timeout_error"
        assert FailureCategory.CONFIDENCE_ERROR.value == "confidence_error"
        assert FailureCategory.UNKNOWN.value == "unknown"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestMetaReasonerConfig:
    """Tests for MetaReasonerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetaReasonerConfig()

        assert config.model == "claude-3-5-haiku-20241022"
        assert config.temperature == 0.3
        assert config.strategy == MetaReasoningStrategyType.ANALYTICAL
        assert config.max_retries == 3
        assert config.enable_cache is True
        assert config.cache_max_size == 100

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetaReasonerConfig(
            model="claude-3-opus",
            temperature=0.5,
            strategy=MetaReasoningStrategyType.REFLECTIVE,
            max_retries=5,
            enable_cache=False,
        )

        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.strategy == MetaReasoningStrategyType.REFLECTIVE
        assert config.max_retries == 5
        assert config.enable_cache is False


# =============================================================================
# Strategy Tests
# =============================================================================


class TestAnalyticalStrategy:
    """Tests for AnalyticalStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = AnalyticalStrategy()
        assert strategy.strategy_type == MetaReasoningStrategyType.ANALYTICAL

    def test_prepare_failure_analysis_prompt(self):
        """Test failure analysis prompt preparation."""
        strategy = AnalyticalStrategy()
        failures = [
            FailureRecord(
                failure_id="F001",
                category=FailureCategory.SYNTAX_ERROR,
                error_message="Invalid syntax",
                domain="contracts",
                strategy_used="few_shot",
            )
        ]

        prompt = strategy.prepare_failure_analysis_prompt(failures)

        assert "syntax_error" in prompt.lower()
        assert "contracts" in prompt.lower()
        assert "pattern" in prompt.lower()

    def test_prepare_strategy_evaluation_prompt(self):
        """Test strategy evaluation prompt preparation."""
        strategy = AnalyticalStrategy()
        performance_data = {
            "few_shot": {"accuracy": 0.8, "cases": 100},
            "chain_of_thought": {"accuracy": 0.85, "cases": 100},
        }

        prompt = strategy.prepare_strategy_evaluation_prompt(performance_data)

        assert "few_shot" in prompt
        assert "chain_of_thought" in prompt
        assert "performance" in prompt.lower()

    def test_prepare_prompt_optimization_prompt(self):
        """Test prompt optimization prompt preparation."""
        strategy = AnalyticalStrategy()
        test_prompt = "Generate ASP rules for the given case."
        failures = [
            FailureRecord(
                failure_id="F001",
                error_message="Rule was too general",
            )
        ]

        prompt = strategy.prepare_prompt_optimization_prompt(test_prompt, failures)

        assert "Generate ASP rules" in prompt
        assert "improve" in prompt.lower()

    def test_prepare_self_assessment_prompt(self):
        """Test self-assessment prompt preparation."""
        strategy = AnalyticalStrategy()
        metrics = {"accuracy": 0.75, "consistency": 0.8, "coverage": 0.6}

        prompt = strategy.prepare_self_assessment_prompt(metrics)

        assert "accuracy" in prompt
        assert "trajectory" in prompt.lower()


class TestReflectiveStrategy:
    """Tests for ReflectiveStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = ReflectiveStrategy()
        assert strategy.strategy_type == MetaReasoningStrategyType.REFLECTIVE

    def test_failure_analysis_includes_second_order(self):
        """Test that failure analysis includes second-order reasoning."""
        strategy = ReflectiveStrategy()
        failures = [FailureRecord(failure_id="F001")]

        prompt = strategy.prepare_failure_analysis_prompt(failures)

        assert "second" in prompt.lower() or "reflect" in prompt.lower()
        assert "bias" in prompt.lower()

    def test_self_assessment_includes_meta_reflection(self):
        """Test that self-assessment includes meta-reflection."""
        strategy = ReflectiveStrategy()
        metrics = {"accuracy": 0.8}

        prompt = strategy.prepare_self_assessment_prompt(metrics)

        assert "blind" in prompt.lower() or "miss" in prompt.lower()


class TestComparativeStrategy:
    """Tests for ComparativeStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = ComparativeStrategy()
        assert strategy.strategy_type == MetaReasoningStrategyType.COMPARATIVE

    def test_failure_analysis_groups_by_domain(self):
        """Test that failure analysis groups failures by domain."""
        strategy = ComparativeStrategy()
        failures = [
            FailureRecord(failure_id="F001", domain="contracts"),
            FailureRecord(failure_id="F002", domain="torts"),
            FailureRecord(failure_id="F003", domain="contracts"),
        ]

        prompt = strategy.prepare_failure_analysis_prompt(failures)

        assert "contracts" in prompt.lower()
        assert "torts" in prompt.lower()
        assert "cross" in prompt.lower() or "compare" in prompt.lower()


class TestDiagnosticStrategy:
    """Tests for DiagnosticStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = DiagnosticStrategy()
        assert strategy.strategy_type == MetaReasoningStrategyType.DIAGNOSTIC

    def test_failure_analysis_includes_diagnosis(self):
        """Test that failure analysis includes diagnosis terminology."""
        strategy = DiagnosticStrategy()
        failures = [
            FailureRecord(
                failure_id="F001",
                category=FailureCategory.SYNTAX_ERROR,
                error_message="Invalid syntax",
            )
        ]

        prompt = strategy.prepare_failure_analysis_prompt(failures)

        assert "diagnos" in prompt.lower() or "symptom" in prompt.lower()
        assert "root cause" in prompt.lower() or "treatment" in prompt.lower()


class TestCreateMetaReasoningStrategy:
    """Tests for strategy factory function."""

    def test_create_analytical_strategy(self):
        """Test creating analytical strategy."""
        strategy = create_meta_reasoning_strategy(MetaReasoningStrategyType.ANALYTICAL)
        assert isinstance(strategy, AnalyticalStrategy)

    def test_create_reflective_strategy(self):
        """Test creating reflective strategy."""
        strategy = create_meta_reasoning_strategy(MetaReasoningStrategyType.REFLECTIVE)
        assert isinstance(strategy, ReflectiveStrategy)

    def test_create_comparative_strategy(self):
        """Test creating comparative strategy."""
        strategy = create_meta_reasoning_strategy(MetaReasoningStrategyType.COMPARATIVE)
        assert isinstance(strategy, ComparativeStrategy)

    def test_create_diagnostic_strategy(self):
        """Test creating diagnostic strategy."""
        strategy = create_meta_reasoning_strategy(MetaReasoningStrategyType.DIAGNOSTIC)
        assert isinstance(strategy, DiagnosticStrategy)


# =============================================================================
# MetaReasonerLLM Tests
# =============================================================================


class TestMetaReasonerLLMInitialization:
    """Tests for MetaReasonerLLM initialization."""

    def test_default_initialization(self):
        """Test initialization with default config."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        assert reasoner.config.strategy == MetaReasoningStrategyType.ANALYTICAL
        assert reasoner._total_analyses == 0

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = MetaReasonerConfig(
            strategy=MetaReasoningStrategyType.REFLECTIVE,
            temperature=0.5,
        )
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm, config=config)

        assert reasoner.config.strategy == MetaReasoningStrategyType.REFLECTIVE
        assert reasoner.config.temperature == 0.5

    def test_custom_llm_interface(self):
        """Test initialization with custom LLM interface."""
        mock_llm = Mock()

        reasoner = MetaReasonerLLM(mock_llm)

        assert reasoner._llm == mock_llm


class TestMetaReasonerLLMInputValidation:
    """Tests for input validation."""

    def test_validate_input_exceeds_max_length(self):
        """Test that input exceeding max length raises error."""
        config = MetaReasonerConfig(max_input_length=100)
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm, config=config)

        with pytest.raises(ValueError, match="exceeds maximum length"):
            reasoner._validate_input("x" * 150, "test_input")

    def test_sanitize_input_removes_control_chars(self):
        """Test that sanitization removes control characters."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        sanitized = reasoner._sanitize_input("Hello\x00World\x1f!")

        assert "\x00" not in sanitized
        assert "\x1f" not in sanitized
        assert "HelloWorld!" in sanitized

    def test_sanitize_input_preserves_newlines(self):
        """Test that sanitization preserves newlines and tabs."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        sanitized = reasoner._sanitize_input("Hello\nWorld\tTest")

        assert "\n" in sanitized
        assert "\t" in sanitized


class TestMetaReasonerLLMAnalyzeFailures:
    """Tests for analyze_failure_patterns method."""

    def test_analyze_empty_failures(self):
        """Test analyzing empty failure list."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        result = reasoner.analyze_failure_patterns([])

        assert result.insights == []
        assert result.confidence == 1.0

    def test_analyze_failures_calls_llm(self):
        """Test that analyzing failures calls LLM."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "patterns": [
                    {
                        "pattern_id": "P1",
                        "description": "Syntax errors in constraints",
                        "frequency": 0.3,
                        "severity": "high",
                        "examples": ["example1"],
                    }
                ],
                "root_causes": [],
                "recommendations": [],
                "confidence": 0.8,
                "reasoning_summary": "Analysis complete",
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)
        failures = [
            FailureRecord(
                failure_id="F001",
                category=FailureCategory.SYNTAX_ERROR,
                error_message="Invalid syntax",
            )
        ]

        result = reasoner.analyze_failure_patterns(failures)

        assert mock_llm.generate.called
        assert len(result.insights) >= 1
        assert result.confidence == 0.8

    def test_analyze_failures_caching(self):
        """Test that results are cached."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "patterns": [],
                "root_causes": [],
                "recommendations": [],
                "confidence": 0.7,
                "reasoning_summary": "",
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)
        failures = [FailureRecord(failure_id="F001")]

        # First call
        _ = reasoner.analyze_failure_patterns(failures)
        call_count_1 = mock_llm.generate.call_count

        # Second call should hit cache
        _ = reasoner.analyze_failure_patterns(failures)
        call_count_2 = mock_llm.generate.call_count

        assert call_count_1 == call_count_2  # No additional LLM call
        assert reasoner._cache_hits == 1


class TestMetaReasonerLLMSuggestStrategyChanges:
    """Tests for suggest_strategy_changes method."""

    def test_suggest_with_empty_data(self):
        """Test suggesting changes with empty data."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        changes = reasoner.suggest_strategy_changes({})

        assert changes == []

    def test_suggest_strategy_changes(self):
        """Test suggesting strategy changes."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "strategy_rankings": [],
                "recommendations": [
                    {
                        "current_strategy": "few_shot",
                        "recommended_strategy": "chain_of_thought",
                        "conditions": {"domain": "contracts"},
                        "expected_improvement": "10% accuracy gain",
                        "confidence": 0.85,
                        "rationale": "Better for complex reasoning",
                    }
                ],
                "overall_assessment": "Strategy change recommended",
                "confidence": 0.8,
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)
        performance_data = {
            "few_shot": {"accuracy": 0.7},
            "chain_of_thought": {"accuracy": 0.8},
        }

        changes = reasoner.suggest_strategy_changes(performance_data)

        assert len(changes) == 1
        assert changes[0].current_strategy == "few_shot"
        assert changes[0].recommended_strategy == "chain_of_thought"


class TestMetaReasonerLLMOptimizePrompts:
    """Tests for optimize_prompts method."""

    def test_optimize_empty_prompt(self):
        """Test optimizing empty prompt."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        improvements = reasoner.optimize_prompts("", [])

        assert improvements == []

    def test_optimize_prompts(self):
        """Test prompt optimization."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "issues_identified": [
                    {
                        "issue_id": "I1",
                        "description": "Missing examples",
                        "severity": "high",
                        "location": "Start of prompt",
                    }
                ],
                "improvements": [
                    {
                        "improvement_id": "IMP1",
                        "description": "Add few-shot examples",
                        "rationale": "Improves pattern matching",
                        "before": None,
                        "after": "Example: contract(c1) :- signed(c1).",
                    }
                ],
                "optimized_prompt_summary": "Added examples",
                "expected_improvement": "15% fewer errors",
                "confidence": 0.75,
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)

        improvements = reasoner.optimize_prompts(
            "Generate ASP rules.",
            [FailureRecord(failure_id="F001", error_message="Too general")],
        )

        assert len(improvements) >= 1
        assert any("example" in imp.lower() for imp in improvements)


class TestMetaReasonerLLMAssessSelfImprovement:
    """Tests for assess_self_improvement method."""

    def test_assess_empty_metrics(self):
        """Test assessing with empty metrics."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        report = reasoner.assess_self_improvement({})

        assert report.overall_trajectory == "unknown"
        assert "Collect more metrics" in report.next_goals[0]

    def test_assess_self_improvement(self):
        """Test self-improvement assessment."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "overall_trajectory": "improving",
                "trajectory_confidence": 0.8,
                "strengths": [
                    {
                        "area": "Contract rules",
                        "evidence": "85% accuracy",
                        "confidence": 0.9,
                    }
                ],
                "gaps": [
                    {
                        "area": "Tort rules",
                        "severity": "high",
                        "evidence": "60% accuracy",
                    }
                ],
                "confidence_calibration": {
                    "is_well_calibrated": True,
                    "calibration_issues": [],
                    "recommendations": [],
                },
                "next_goals": [
                    {
                        "goal": "Improve tort coverage",
                        "priority": "high",
                        "rationale": "Low accuracy",
                        "success_criteria": "80% accuracy",
                    }
                ],
                "epistemic_humility_note": "Assessment may be optimistic",
                "overall_confidence": 0.75,
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)
        metrics = {"accuracy": 0.75, "consistency": 0.8}

        report = reasoner.assess_self_improvement(metrics)

        assert report.overall_trajectory == "improving"
        assert len(report.insights) >= 1
        assert len(report.next_goals) >= 1


class TestMetaReasonerLLMStrategyManagement:
    """Tests for strategy management."""

    def test_set_strategy(self):
        """Test changing strategy."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        assert reasoner._strategy.strategy_type == MetaReasoningStrategyType.ANALYTICAL

        reasoner.set_strategy(MetaReasoningStrategyType.DIAGNOSTIC)

        assert reasoner._strategy.strategy_type == MetaReasoningStrategyType.DIAGNOSTIC


class TestMetaReasonerLLMStatistics:
    """Tests for statistics tracking."""

    def test_get_statistics(self):
        """Test getting statistics."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        stats = reasoner.get_statistics()

        assert stats["total_analyses"] == 0
        assert stats["successful_analyses"] == 0
        assert stats["failed_analyses"] == 0
        assert stats["success_rate"] == 0.0
        assert "current_strategy" in stats

    def test_statistics_update_on_analysis(self):
        """Test that statistics update after analysis."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "patterns": [],
                "root_causes": [],
                "recommendations": [],
                "confidence": 0.7,
                "reasoning_summary": "",
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)
        failures = [FailureRecord(failure_id="F001")]

        reasoner.analyze_failure_patterns(failures)

        stats = reasoner.get_statistics()
        assert stats["total_analyses"] == 1
        assert stats["successful_analyses"] == 1

    def test_reset_statistics(self):
        """Test resetting statistics."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "patterns": [],
                "root_causes": [],
                "recommendations": [],
                "confidence": 0.7,
                "reasoning_summary": "",
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)
        reasoner.analyze_failure_patterns([FailureRecord(failure_id="F001")])

        reasoner.reset_statistics()

        stats = reasoner.get_statistics()
        assert stats["total_analyses"] == 0


class TestMetaReasonerLLMCaching:
    """Tests for caching functionality."""

    def test_get_cache_statistics(self):
        """Test getting cache statistics."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        cache_stats = reasoner.get_cache_statistics()

        assert cache_stats["cache_size"] == 0
        assert cache_stats["cache_hits"] == 0
        assert cache_stats["cache_enabled"] is True

    def test_clear_cache(self):
        """Test clearing cache."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "patterns": [],
                "root_causes": [],
                "recommendations": [],
                "confidence": 0.7,
                "reasoning_summary": "",
            }
        )

        reasoner = MetaReasonerLLM(mock_llm)
        failures = [FailureRecord(failure_id="F001")]

        # Populate cache
        reasoner.analyze_failure_patterns(failures)
        assert reasoner.get_cache_statistics()["cache_size"] == 1

        # Clear cache
        reasoner.clear_cache()
        assert reasoner.get_cache_statistics()["cache_size"] == 0

    def test_cache_disabled(self):
        """Test that caching can be disabled."""
        config = MetaReasonerConfig(enable_cache=False)
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps(
            {
                "patterns": [],
                "root_causes": [],
                "recommendations": [],
                "confidence": 0.7,
                "reasoning_summary": "",
            }
        )

        reasoner = MetaReasonerLLM(mock_llm, config=config)
        failures = [FailureRecord(failure_id="F001")]

        # First call
        reasoner.analyze_failure_patterns(failures)
        # Second call should not hit cache
        reasoner.analyze_failure_patterns(failures)

        assert mock_llm.generate.call_count == 2


class TestMetaReasonerLLMRetryLogic:
    """Tests for retry logic."""

    def test_retry_on_failure(self):
        """Test retry behavior on LLM failure."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            json.dumps(
                {
                    "patterns": [],
                    "root_causes": [],
                    "recommendations": [],
                    "confidence": 0.7,
                    "reasoning_summary": "",
                }
            ),
        ]

        config = MetaReasonerConfig(
            max_retries=3,
            retry_base_delay_seconds=0.01,  # Fast for testing
            retry_jitter_max_seconds=0.001,
        )

        reasoner = MetaReasonerLLM(mock_llm, config=config)
        failures = [FailureRecord(failure_id="F001")]

        result = reasoner.analyze_failure_patterns(failures)

        assert mock_llm.generate.call_count == 3
        assert result is not None

    def test_raises_after_max_retries(self):
        """Test that MetaReasoningError is raised after max retries."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("Always fails")

        config = MetaReasonerConfig(
            max_retries=2,
            retry_base_delay_seconds=0.01,
            retry_jitter_max_seconds=0.001,
        )

        reasoner = MetaReasonerLLM(mock_llm, config=config)
        failures = [FailureRecord(failure_id="F001")]

        with pytest.raises(MetaReasoningError) as exc_info:
            reasoner.analyze_failure_patterns(failures)

        assert exc_info.value.attempts == 2


class TestMetaReasonerLLMJSONParsing:
    """Tests for JSON response parsing."""

    def test_parse_json_with_markdown_blocks(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        response = '```json\n{"key": "value"}\n```'
        parsed = reasoner._parse_json_response(response)

        assert parsed["key"] == "value"

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding text."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        response = 'Here is the result: {"key": "value"} End of result.'
        parsed = reasoner._parse_json_response(response)

        assert parsed["key"] == "value"

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)

        with pytest.raises(ValueError, match="Failed to parse JSON"):
            reasoner._parse_json_response("not valid json at all")


# =============================================================================
# Exception Tests
# =============================================================================


class TestMetaReasoningError:
    """Tests for MetaReasoningError exception."""

    def test_exception_creation(self):
        """Test exception creation."""
        error = MetaReasoningError(
            "Analysis failed",
            attempts=3,
            last_error="Connection timeout",
        )

        assert str(error) == "Analysis failed"
        assert error.attempts == 3
        assert error.last_error == "Connection timeout"

    def test_exception_defaults(self):
        """Test exception default values."""
        error = MetaReasoningError("Simple error")

        assert error.attempts == 0
        assert error.last_error is None


# =============================================================================
# Integration Tests (Abstract Base Class)
# =============================================================================


class TestMetaReasonerAbstractClass:
    """Tests for MetaReasoner abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that MetaReasoner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetaReasoner()

    def test_metareasonerllm_is_metareasoner(self):
        """Test that MetaReasonerLLM is a MetaReasoner."""
        mock_llm = Mock()
        reasoner = MetaReasonerLLM(mock_llm)
        assert isinstance(reasoner, MetaReasoner)
