"""Unit tests for CriticLLM - Edge case and contradiction detection.

Tests for the Critic LLM implementation (Issue #189, Phase 6 Neural Ensemble).
"""

import pytest
from unittest.mock import Mock, patch

from loft.neural.ensemble.critic import (
    # Data classes
    EdgeCase,
    Contradiction,
    GeneralizationAssessment,
    CriticResult,
    # Config
    CriticConfig,
    # Main class
    CriticLLM,
    # Strategy classes
    CriticStrategyType,
    AdversarialStrategy,
    CooperativeStrategy,
    SystematicStrategy,
    DialecticalStrategy,
    create_critic_strategy,
    # Exceptions
    CriticAnalysisError,
    # Schemas
    EdgeCaseSchema,
    ContradictionSchema,
    EdgeCaseAnalysisResult,
    ContradictionAnalysisResult,
    GeneralizationSchema,
)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestEdgeCase:
    """Tests for EdgeCase dataclass."""

    def test_edge_case_creation(self):
        """Test basic EdgeCase creation."""
        edge_case = EdgeCase(
            description="Rule fails when contract has no parties",
            scenario="A contract object exists but parties list is empty",
            failure_mode="false_positive",
            severity="high",
            confidence=0.85,
        )

        assert edge_case.description == "Rule fails when contract has no parties"
        assert edge_case.severity == "high"
        assert edge_case.confidence == 0.85

    def test_edge_case_severity_normalization(self):
        """Test that severity is normalized to lowercase."""
        edge_case = EdgeCase(
            description="Test",
            scenario="Test scenario",
            failure_mode="false_negative",
            severity="HIGH",
            confidence=0.5,
        )
        assert edge_case.severity == "high"

    def test_edge_case_invalid_severity_defaults(self):
        """Test that invalid severity defaults to medium."""
        edge_case = EdgeCase(
            description="Test",
            scenario="Test scenario",
            failure_mode="false_negative",
            severity="invalid_severity",
            confidence=0.5,
        )
        assert edge_case.severity == "medium"

    def test_edge_case_with_optional_fields(self):
        """Test EdgeCase with optional fields."""
        edge_case = EdgeCase(
            description="Test",
            scenario="Test scenario",
            failure_mode="false_negative",
            severity="low",
            confidence=0.5,
            suggested_fix="Add null check",
            related_predicates=["contract", "party"],
        )

        assert edge_case.suggested_fix == "Add null check"
        assert edge_case.related_predicates == ["contract", "party"]


class TestContradiction:
    """Tests for Contradiction dataclass."""

    def test_contradiction_creation(self):
        """Test basic Contradiction creation."""
        contradiction = Contradiction(
            description="Rules contradict on contract validity",
            rule1="valid(C) :- contract(C), signed(C).",
            rule2="invalid(C) :- contract(C), not written(C).",
            conflict_type="direct",
            confidence=0.9,
        )

        assert "contradict" in contradiction.description.lower()
        assert contradiction.conflict_type == "direct"
        assert contradiction.confidence == 0.9

    def test_contradiction_conflict_type_normalization(self):
        """Test that conflict type is normalized to lowercase."""
        contradiction = Contradiction(
            description="Test",
            rule1="rule1",
            rule2="rule2",
            conflict_type="IMPLICIT",
        )
        assert contradiction.conflict_type == "implicit"

    def test_contradiction_invalid_conflict_type_defaults(self):
        """Test that invalid conflict type defaults to implicit."""
        contradiction = Contradiction(
            description="Test",
            rule1="rule1",
            rule2="rule2",
            conflict_type="unknown_type",
        )
        assert contradiction.conflict_type == "implicit"

    def test_contradiction_with_optional_fields(self):
        """Test Contradiction with optional fields."""
        contradiction = Contradiction(
            description="Test",
            rule1="rule1",
            rule2="rule2",
            conflict_type="semantic",
            example_trigger="contract(c1). signed(c1). not written(c1).",
            resolution_suggestion="Use explicit precedence",
            confidence=0.75,
            affected_predicates=["valid", "invalid"],
        )

        assert contradiction.example_trigger is not None
        assert contradiction.resolution_suggestion is not None
        assert len(contradiction.affected_predicates) == 2


class TestGeneralizationAssessment:
    """Tests for GeneralizationAssessment dataclass."""

    def test_generalization_assessment_creation(self):
        """Test basic GeneralizationAssessment creation."""
        assessment = GeneralizationAssessment(
            generalization_score=0.75,
            coverage_estimate=0.8,
            overfitting_risk=0.2,
            underfitting_risk=0.1,
        )

        assert assessment.generalization_score == 0.75
        assert assessment.coverage_estimate == 0.8

    def test_generalization_assessment_with_test_cases(self):
        """Test GeneralizationAssessment with test cases and edge cases."""
        edge_case = EdgeCase(
            description="Edge case",
            scenario="Scenario",
            failure_mode="false_positive",
            severity="medium",
            confidence=0.6,
        )

        assessment = GeneralizationAssessment(
            generalization_score=0.6,
            coverage_estimate=0.7,
            overfitting_risk=0.3,
            underfitting_risk=0.15,
            test_cases_needed=["Test unsigned contract", "Test oral agreement"],
            edge_cases_found=[edge_case],
            confidence=0.8,
        )

        assert len(assessment.test_cases_needed) == 2
        assert len(assessment.edge_cases_found) == 1


class TestCriticResult:
    """Tests for CriticResult dataclass."""

    def test_critic_result_creation(self):
        """Test basic CriticResult creation."""
        result = CriticResult(
            rule="valid(C) :- contract(C).",
            overall_quality_score=0.7,
            recommendation="revise",
        )

        assert result.rule == "valid(C) :- contract(C)."
        assert result.recommendation == "revise"
        assert result.edge_cases == []
        assert result.contradictions == []


# =============================================================================
# Config Tests
# =============================================================================


class TestCriticConfig:
    """Tests for CriticConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CriticConfig()

        assert config.model == "claude-3-5-haiku-20241022"
        assert config.temperature == 0.4
        assert config.max_tokens == 4096
        assert config.strategy == CriticStrategyType.ADVERSARIAL
        assert config.max_edge_cases == 10
        assert config.max_contradictions == 10
        assert config.confidence_threshold == 0.3
        assert config.enable_cache is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CriticConfig(
            model="claude-3-opus-20240229",
            temperature=0.6,
            strategy=CriticStrategyType.DIALECTICAL,
            max_edge_cases=5,
            confidence_threshold=0.5,
        )

        assert config.model == "claude-3-opus-20240229"
        assert config.temperature == 0.6
        assert config.strategy == CriticStrategyType.DIALECTICAL
        assert config.max_edge_cases == 5
        assert config.confidence_threshold == 0.5


# =============================================================================
# Strategy Tests
# =============================================================================


class TestCriticStrategyType:
    """Tests for CriticStrategyType enum."""

    def test_strategy_types_exist(self):
        """Test that all strategy types are defined."""
        assert CriticStrategyType.ADVERSARIAL.value == "adversarial"
        assert CriticStrategyType.COOPERATIVE.value == "cooperative"
        assert CriticStrategyType.SYSTEMATIC.value == "systematic"
        assert CriticStrategyType.DIALECTICAL.value == "dialectical"


class TestAdversarialStrategy:
    """Tests for AdversarialStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = AdversarialStrategy()
        assert strategy.strategy_type == CriticStrategyType.ADVERSARIAL

    def test_prepare_edge_case_prompt(self):
        """Test edge case prompt preparation."""
        strategy = AdversarialStrategy()
        base_prompt = "Analyze this rule:"
        rule = "valid(C) :- contract(C)."
        context = {"domain": "contracts"}

        prompt = strategy.prepare_edge_case_prompt(base_prompt, rule, context)

        assert "Analyze this rule:" in prompt
        assert "Adversarial" in prompt
        assert "every possible way" in prompt.lower()

    def test_prepare_contradiction_prompt(self):
        """Test contradiction prompt preparation."""
        strategy = AdversarialStrategy()
        base_prompt = "Check contradictions:"
        rule = "valid(C) :- contract(C)."
        existing_rules = ["enforceable(C) :- valid(C)."]

        prompt = strategy.prepare_contradiction_prompt(
            base_prompt, rule, existing_rules
        )

        assert "Check contradictions:" in prompt
        assert "Adversarial" in prompt


class TestCooperativeStrategy:
    """Tests for CooperativeStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = CooperativeStrategy()
        assert strategy.strategy_type == CriticStrategyType.COOPERATIVE

    def test_prepare_edge_case_prompt(self):
        """Test cooperative prompt preparation."""
        strategy = CooperativeStrategy()
        prompt = strategy.prepare_edge_case_prompt("Base:", "rule", {})

        assert "Constructive" in prompt
        assert "suggest" in prompt.lower()


class TestSystematicStrategy:
    """Tests for SystematicStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = SystematicStrategy()
        assert strategy.strategy_type == CriticStrategyType.SYSTEMATIC

    def test_prepare_edge_case_prompt(self):
        """Test systematic prompt preparation."""
        strategy = SystematicStrategy()
        prompt = strategy.prepare_edge_case_prompt("Base:", "rule", {})

        assert "Checklist" in prompt
        assert "BOUNDARY" in prompt


class TestDialecticalStrategy:
    """Tests for DialecticalStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = DialecticalStrategy()
        assert strategy.strategy_type == CriticStrategyType.DIALECTICAL

    def test_prepare_edge_case_prompt(self):
        """Test dialectical prompt preparation."""
        strategy = DialecticalStrategy()
        prompt = strategy.prepare_edge_case_prompt("Base:", "rule", {})

        assert "THESIS" in prompt
        assert "ANTITHESIS" in prompt
        assert "SYNTHESIS" in prompt


class TestCreateCriticStrategy:
    """Tests for create_critic_strategy factory."""

    def test_create_adversarial_strategy(self):
        """Test creating adversarial strategy."""
        strategy = create_critic_strategy(CriticStrategyType.ADVERSARIAL)
        assert isinstance(strategy, AdversarialStrategy)

    def test_create_cooperative_strategy(self):
        """Test creating cooperative strategy."""
        strategy = create_critic_strategy(CriticStrategyType.COOPERATIVE)
        assert isinstance(strategy, CooperativeStrategy)

    def test_create_systematic_strategy(self):
        """Test creating systematic strategy."""
        strategy = create_critic_strategy(CriticStrategyType.SYSTEMATIC)
        assert isinstance(strategy, SystematicStrategy)

    def test_create_dialectical_strategy(self):
        """Test creating dialectical strategy."""
        strategy = create_critic_strategy(CriticStrategyType.DIALECTICAL)
        assert isinstance(strategy, DialecticalStrategy)


# =============================================================================
# CriticLLM Tests
# =============================================================================


class TestCriticLLMInitialization:
    """Tests for CriticLLM initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default config."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface"):
            critic = CriticLLM(mock_provider)

        assert critic.config.model == "claude-3-5-haiku-20241022"
        assert critic.config.strategy == CriticStrategyType.ADVERSARIAL

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        mock_provider = Mock()
        config = CriticConfig(
            strategy=CriticStrategyType.DIALECTICAL,
            temperature=0.5,
        )

        with patch("loft.neural.ensemble.critic.LLMInterface"):
            critic = CriticLLM(mock_provider, config)

        assert critic.config.strategy == CriticStrategyType.DIALECTICAL
        assert critic.config.temperature == 0.5

    def test_initialization_with_custom_strategy(self):
        """Test initialization with custom strategy overrides config."""
        mock_provider = Mock()
        config = CriticConfig(strategy=CriticStrategyType.ADVERSARIAL)
        custom_strategy = DialecticalStrategy()

        with patch("loft.neural.ensemble.critic.LLMInterface"):
            critic = CriticLLM(mock_provider, config, custom_strategy)

        # Custom strategy should override config
        assert critic._strategy.strategy_type == CriticStrategyType.DIALECTICAL


class TestCriticLLMFindEdgeCases:
    """Tests for CriticLLM.find_edge_cases method."""

    @pytest.fixture
    def mock_critic(self):
        """Create a mock critic with mocked LLM interface."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            critic = CriticLLM(mock_provider)
            critic._llm = mock_llm

            return critic

    def test_find_edge_cases_success(self, mock_critic):
        """Test successful edge case detection."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = EdgeCaseAnalysisResult(
            edge_cases=[
                EdgeCaseSchema(
                    description="Rule fails for unsigned contracts",
                    scenario="contract(c1). not signed(c1).",
                    failure_mode="false_positive",
                    severity="high",
                    confidence=0.85,
                    suggested_fix="Add signed(C) to body",
                    related_predicates=["signed"],
                ),
            ],
            analysis_summary="Found 1 significant edge case",
            confidence=0.85,
        )
        mock_critic._llm.query.return_value = mock_response

        edge_cases = mock_critic.find_edge_cases(
            rule="valid(C) :- contract(C).",
            context={"domain": "contracts", "predicates": ["contract", "signed"]},
        )

        assert len(edge_cases) == 1
        assert edge_cases[0].description == "Rule fails for unsigned contracts"
        assert edge_cases[0].severity == "high"

    def test_find_edge_cases_filters_low_confidence(self, mock_critic):
        """Test that low confidence edge cases are filtered."""
        mock_critic.config.confidence_threshold = 0.5

        mock_response = Mock()
        mock_response.content = EdgeCaseAnalysisResult(
            edge_cases=[
                EdgeCaseSchema(
                    description="High confidence edge case",
                    scenario="scenario1",
                    failure_mode="false_positive",
                    severity="high",
                    confidence=0.8,
                ),
                EdgeCaseSchema(
                    description="Low confidence edge case",
                    scenario="scenario2",
                    failure_mode="false_negative",
                    severity="low",
                    confidence=0.3,  # Below threshold
                ),
            ],
            analysis_summary="Found 2 edge cases",
            confidence=0.7,
        )
        mock_critic._llm.query.return_value = mock_response

        edge_cases = mock_critic.find_edge_cases(
            rule="test_rule.",
            context={},
        )

        assert len(edge_cases) == 1
        assert edge_cases[0].confidence >= 0.5

    def test_find_edge_cases_handles_error_with_retry(self, mock_critic):
        """Test that errors trigger retries."""
        mock_response = Mock()
        mock_response.content = EdgeCaseAnalysisResult(
            edge_cases=[],
            analysis_summary="No issues found",
            confidence=0.9,
        )

        # Fail first two attempts, succeed on third
        # Use ConnectionError (a specific retryable exception) instead of generic Exception
        mock_critic._llm.query.side_effect = [
            ConnectionError("API error"),
            ConnectionError("API error"),
            mock_response,
        ]

        with patch("time.sleep"):  # Skip actual delays
            edge_cases = mock_critic.find_edge_cases(
                rule="test.",
                context={},
            )

        assert edge_cases == []
        assert mock_critic._llm.query.call_count == 3


class TestCriticLLMDetectContradictions:
    """Tests for CriticLLM.detect_contradictions method."""

    @pytest.fixture
    def mock_critic(self):
        """Create a mock critic with mocked LLM interface."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            critic = CriticLLM(mock_provider)
            critic._llm = mock_llm

            return critic

    def test_detect_contradictions_success(self, mock_critic):
        """Test successful contradiction detection."""
        mock_response = Mock()
        mock_response.content = ContradictionAnalysisResult(
            contradictions=[
                ContradictionSchema(
                    description="Direct conflict on validity",
                    rule1="valid(C) :- contract(C).",
                    rule2="invalid(C) :- contract(C), not written(C).",
                    conflict_type="direct",
                    example_trigger="contract(c1). not written(c1).",
                    resolution_suggestion="Add mutual exclusion",
                    confidence=0.9,
                    affected_predicates=["valid", "invalid"],
                ),
            ],
            is_consistent=False,
            analysis_summary="Found 1 direct contradiction",
            confidence=0.9,
        )
        mock_critic._llm.query.return_value = mock_response

        contradictions = mock_critic.detect_contradictions(
            rule="valid(C) :- contract(C).",
            existing_rules=["invalid(C) :- contract(C), not written(C)."],
        )

        assert len(contradictions) == 1
        assert contradictions[0].conflict_type == "direct"
        assert contradictions[0].confidence == 0.9

    def test_detect_contradictions_no_conflicts(self, mock_critic):
        """Test when no contradictions are found."""
        mock_response = Mock()
        mock_response.content = ContradictionAnalysisResult(
            contradictions=[],
            is_consistent=True,
            analysis_summary="No contradictions found",
            confidence=0.95,
        )
        mock_critic._llm.query.return_value = mock_response

        contradictions = mock_critic.detect_contradictions(
            rule="new_rule(X) :- some_predicate(X).",
            existing_rules=["other_rule(Y) :- different_predicate(Y)."],
        )

        assert len(contradictions) == 0


class TestCriticLLMAssessGeneralization:
    """Tests for CriticLLM.assess_generalization method."""

    @pytest.fixture
    def mock_critic(self):
        """Create a mock critic with mocked LLM interface."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            critic = CriticLLM(mock_provider)
            critic._llm = mock_llm

            return critic

    def test_assess_generalization_success(self, mock_critic):
        """Test successful generalization assessment."""
        mock_response = Mock()
        mock_response.content = GeneralizationSchema(
            generalization_score=0.75,
            coverage_estimate=0.8,
            overfitting_risk=0.2,
            underfitting_risk=0.1,
            test_cases_needed=["Test oral contract", "Test implied contract"],
            reasoning="Rule generalizes well but may miss edge cases",
            confidence=0.85,
        )
        mock_critic._llm.query.return_value = mock_response

        assessment = mock_critic.assess_generalization(
            rule="valid(C) :- contract(C), signed(C).",
            test_cases=[
                {"description": "Written contract", "expected": True},
                {"description": "Oral contract", "expected": False},
            ],
        )

        assert assessment.generalization_score == 0.75
        assert assessment.coverage_estimate == 0.8
        assert len(assessment.test_cases_needed) == 2


class TestCriticLLMComprehensiveAnalysis:
    """Tests for CriticLLM.analyze_rule_comprehensive method."""

    @pytest.fixture
    def mock_critic(self):
        """Create a mock critic with mocked LLM interface."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            critic = CriticLLM(mock_provider)
            critic._llm = mock_llm

            return critic

    def test_comprehensive_analysis_accept_recommendation(self, mock_critic):
        """Test comprehensive analysis with accept recommendation."""
        # Mock all three analysis responses
        edge_case_response = Mock()
        edge_case_response.content = EdgeCaseAnalysisResult(
            edge_cases=[],
            analysis_summary="No edge cases",
            confidence=0.9,
        )

        contradiction_response = Mock()
        contradiction_response.content = ContradictionAnalysisResult(
            contradictions=[],
            is_consistent=True,
            analysis_summary="No contradictions",
            confidence=0.95,
        )

        generalization_response = Mock()
        generalization_response.content = GeneralizationSchema(
            generalization_score=0.9,
            coverage_estimate=0.85,
            overfitting_risk=0.1,
            underfitting_risk=0.1,
            test_cases_needed=[],
            reasoning="Good generalization",
            confidence=0.9,
        )

        mock_critic._llm.query.side_effect = [
            edge_case_response,
            contradiction_response,
            generalization_response,
        ]

        result = mock_critic.analyze_rule_comprehensive(
            rule="valid(C) :- contract(C), signed(C).",
            existing_rules=["enforceable(C) :- valid(C)."],
            context={"domain": "contracts"},
            test_cases=[{"description": "Test 1"}],
        )

        assert result.recommendation == "accept"
        assert result.overall_quality_score > 0.7
        assert len(result.edge_cases) == 0
        assert len(result.contradictions) == 0

    def test_comprehensive_analysis_reject_recommendation(self, mock_critic):
        """Test comprehensive analysis with reject recommendation."""
        # Mock responses with critical edge case
        edge_case_response = Mock()
        edge_case_response.content = EdgeCaseAnalysisResult(
            edge_cases=[
                EdgeCaseSchema(
                    description="Critical failure",
                    scenario="Critical scenario",
                    failure_mode="false_positive",
                    severity="critical",
                    confidence=0.95,
                ),
            ],
            analysis_summary="Critical edge case found",
            confidence=0.95,
        )

        contradiction_response = Mock()
        contradiction_response.content = ContradictionAnalysisResult(
            contradictions=[],
            is_consistent=True,
            analysis_summary="No contradictions",
            confidence=0.9,
        )

        mock_critic._llm.query.side_effect = [
            edge_case_response,
            contradiction_response,
        ]

        result = mock_critic.analyze_rule_comprehensive(
            rule="bad_rule(X) :- something(X).",
            existing_rules=[],
            context={},
        )

        assert result.recommendation == "reject"


class TestCriticLLMStatistics:
    """Tests for CriticLLM statistics tracking."""

    def test_statistics_initial_values(self):
        """Test initial statistics values."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface"):
            critic = CriticLLM(mock_provider)

        stats = critic.get_statistics()

        assert stats["total_analyses"] == 0
        assert stats["edge_case_analyses"] == 0
        assert stats["contradiction_analyses"] == 0
        assert stats["cache_hits"] == 0

    def test_statistics_after_analyses(self):
        """Test statistics after running analyses."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            critic = CriticLLM(mock_provider)
            critic._llm = mock_llm

            # Mock successful response
            mock_response = Mock()
            mock_response.content = EdgeCaseAnalysisResult(
                edge_cases=[
                    EdgeCaseSchema(
                        description="Edge case",
                        scenario="Scenario",
                        failure_mode="false_positive",
                        severity="medium",
                        confidence=0.7,
                    ),
                ],
                analysis_summary="Found 1 edge case",
                confidence=0.8,
            )
            mock_llm.query.return_value = mock_response

            # Run analysis
            critic.find_edge_cases("rule.", {})

            stats = critic.get_statistics()

            assert stats["total_analyses"] == 1
            assert stats["edge_case_analyses"] == 1
            assert stats["total_edge_cases_found"] == 1

    def test_reset_statistics(self):
        """Test resetting statistics."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface"):
            critic = CriticLLM(mock_provider)

        critic._total_analyses = 5
        critic._edge_case_analyses = 3

        critic.reset_statistics()

        stats = critic.get_statistics()
        assert stats["total_analyses"] == 0
        assert stats["edge_case_analyses"] == 0


class TestCriticLLMCaching:
    """Tests for CriticLLM caching functionality."""

    def test_cache_hit(self):
        """Test that cached results are returned."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            critic = CriticLLM(mock_provider)
            critic._llm = mock_llm

            # First call
            mock_response = Mock()
            mock_response.content = EdgeCaseAnalysisResult(
                edge_cases=[
                    EdgeCaseSchema(
                        description="Edge case",
                        scenario="Scenario",
                        failure_mode="false_positive",
                        severity="medium",
                        confidence=0.7,
                    ),
                ],
                analysis_summary="Found 1 edge case",
                confidence=0.8,
            )
            mock_llm.query.return_value = mock_response

            # First call should query LLM
            _result1 = critic.find_edge_cases("rule.", {"domain": "test"})
            assert mock_llm.query.call_count == 1

            # Second call with same inputs should use cache
            _result2 = critic.find_edge_cases("rule.", {"domain": "test"})
            assert mock_llm.query.call_count == 1  # No additional call
            assert critic._cache_hits == 1

    def test_clear_cache(self):
        """Test clearing the cache."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface"):
            critic = CriticLLM(mock_provider)

        # Add something to cache
        critic._cache["test_key"] = CriticResult(rule="test.")

        critic.clear_cache()

        assert len(critic._cache) == 0


class TestCriticLLMSetStrategy:
    """Tests for CriticLLM.set_strategy method."""

    def test_set_strategy(self):
        """Test changing strategy at runtime."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface"):
            critic = CriticLLM(mock_provider)

        assert critic._strategy.strategy_type == CriticStrategyType.ADVERSARIAL

        critic.set_strategy(DialecticalStrategy())

        assert critic._strategy.strategy_type == CriticStrategyType.DIALECTICAL


# =============================================================================
# Exception Tests
# =============================================================================


class TestCriticAnalysisError:
    """Tests for CriticAnalysisError exception."""

    def test_exception_creation(self):
        """Test exception creation with message."""
        error = CriticAnalysisError(
            "Analysis failed", analysis_type="edge_cases", attempts=3
        )

        assert str(error) == "Analysis failed"
        assert error.analysis_type == "edge_cases"
        assert error.attempts == 3


# =============================================================================
# Schema Tests
# =============================================================================


class TestEdgeCaseSchema:
    """Tests for EdgeCaseSchema Pydantic model."""

    def test_valid_schema(self):
        """Test valid schema creation."""
        schema = EdgeCaseSchema(
            description="Test edge case",
            scenario="Test scenario",
            failure_mode="false_positive",
            severity="high",
            confidence=0.85,
        )

        assert schema.description == "Test edge case"
        assert schema.confidence == 0.85

    def test_schema_with_optional_fields(self):
        """Test schema with optional fields."""
        schema = EdgeCaseSchema(
            description="Test",
            scenario="Scenario",
            failure_mode="false_negative",
            severity="medium",
            confidence=0.7,
            suggested_fix="Fix suggestion",
            related_predicates=["pred1", "pred2"],
        )

        assert schema.suggested_fix == "Fix suggestion"
        assert len(schema.related_predicates) == 2


class TestContradictionSchema:
    """Tests for ContradictionSchema Pydantic model."""

    def test_valid_schema(self):
        """Test valid schema creation."""
        schema = ContradictionSchema(
            description="Contradiction description",
            rule1="rule1.",
            rule2="rule2.",
            conflict_type="direct",
            confidence=0.9,
        )

        assert schema.conflict_type == "direct"
        assert schema.confidence == 0.9


class TestGeneralizationSchema:
    """Tests for GeneralizationSchema Pydantic model."""

    def test_valid_schema(self):
        """Test valid schema creation."""
        schema = GeneralizationSchema(
            generalization_score=0.75,
            coverage_estimate=0.8,
            overfitting_risk=0.15,
            underfitting_risk=0.1,
            test_cases_needed=["Test 1", "Test 2"],
            reasoning="Good generalization",
            confidence=0.85,
        )

        assert schema.generalization_score == 0.75
        assert len(schema.test_cases_needed) == 2


# =============================================================================
# Integration with Logic Generator Tests
# =============================================================================


class TestCriticLogicGeneratorIntegration:
    """Tests for critic integration with logic generator patterns."""

    def test_critic_accepts_logic_generator_output_format(self):
        """Test that critic can analyze rules in LogicGeneratorLLM output format."""
        mock_provider = Mock()

        with patch("loft.neural.ensemble.critic.LLMInterface") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            critic = CriticLLM(mock_provider)
            critic._llm = mock_llm

            # Mock response
            mock_response = Mock()
            mock_response.content = EdgeCaseAnalysisResult(
                edge_cases=[],
                analysis_summary="Rule is well-formed",
                confidence=0.9,
            )
            mock_llm.query.return_value = mock_response

            # Use ASP rule format from LogicGeneratorLLM
            rule = "enforceable(C) :- contract(C), has_offer(C), has_acceptance(C)."
            context = {
                "domain": "contracts",
                "predicates": [
                    "contract",
                    "has_offer",
                    "has_acceptance",
                    "enforceable",
                ],
            }

            edge_cases = critic.find_edge_cases(rule, context)

            # Should complete without errors
            assert isinstance(edge_cases, list)
