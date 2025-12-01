"""Tests for strategy evaluation framework."""

import pytest

from loft.meta.strategy import (
    AnalogicalStrategy,
    BalancingTestStrategy,
    CausalChainStrategy,
    ChecklistStrategy,
    ComparisonReport,
    CounterfactualAnalysis,
    DialecticalStrategy,
    ReasoningStrategy,
    RuleBasedStrategy,
    SelectionExplanation,
    SimpleCase,
    StrategyCharacteristics,
    StrategyEvaluator,
    StrategyMetrics,
    StrategySelector,
    StrategyType,
    create_evaluator,
    create_selector,
    get_default_strategies,
)


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_all_types_exist(self):
        """Verify all expected strategy types are defined."""
        expected_types = [
            "CHECKLIST",
            "CAUSAL_CHAIN",
            "BALANCING_TEST",
            "RULE_BASED",
            "DIALECTICAL",
            "ANALOGICAL",
            "DEFAULT",
        ]
        for type_name in expected_types:
            assert hasattr(StrategyType, type_name)

    def test_type_values(self):
        """Test strategy type string values."""
        assert StrategyType.CHECKLIST.value == "checklist"
        assert StrategyType.CAUSAL_CHAIN.value == "causal_chain"
        assert StrategyType.DIALECTICAL.value == "dialectical"


class TestStrategyCharacteristics:
    """Tests for StrategyCharacteristics dataclass."""

    def test_default_characteristics(self):
        """Test default characteristic values."""
        chars = StrategyCharacteristics()
        assert chars.speed == "medium"
        assert chars.accuracy_profile == "balanced"
        assert chars.resource_usage == "medium"
        assert chars.llm_calls_typical == 1

    def test_custom_characteristics(self):
        """Test custom characteristic values."""
        chars = StrategyCharacteristics(
            speed="fast",
            accuracy_profile="high_precision",
            resource_usage="low",
            llm_calls_typical=0,
            best_for=["simple cases"],
            limitations=["no LLM support"],
        )
        assert chars.speed == "fast"
        assert len(chars.best_for) == 1
        assert len(chars.limitations) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        chars = StrategyCharacteristics(speed="slow")
        data = chars.to_dict()
        assert data["speed"] == "slow"
        assert "best_for" in data


class TestStrategyMetrics:
    """Tests for StrategyMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating strategy metrics."""
        metrics = StrategyMetrics(
            strategy_name="checklist",
            domain="contracts",
            total_cases=100,
            successful_cases=85,
            failed_cases=15,
            accuracy=0.85,
            avg_duration_ms=150.0,
        )
        assert metrics.strategy_name == "checklist"
        assert metrics.accuracy == 0.85

    def test_success_rate_property(self):
        """Test success_rate calculation."""
        metrics = StrategyMetrics(
            strategy_name="test",
            total_cases=50,
            successful_cases=40,
            failed_cases=10,
        )
        assert metrics.success_rate == 0.8

    def test_success_rate_zero_cases(self):
        """Test success_rate with no cases."""
        metrics = StrategyMetrics(strategy_name="test", total_cases=0)
        assert metrics.success_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = StrategyMetrics(
            strategy_name="test",
            total_cases=10,
            successful_cases=8,
        )
        data = metrics.to_dict()
        assert data["strategy_name"] == "test"
        assert "success_rate" in data


class TestComparisonReport:
    """Tests for ComparisonReport dataclass."""

    def test_report_creation(self):
        """Test creating a comparison report."""
        report = ComparisonReport(
            report_id="cmp_001",
            domain="contracts",
            strategies_compared=["checklist", "rule_based"],
            best_strategy="checklist",
            best_accuracy=0.9,
            strategy_rankings=[
                {"strategy_name": "checklist", "accuracy": 0.9},
                {"strategy_name": "rule_based", "accuracy": 0.85},
            ],
            recommendations=["Use checklist for contracts"],
        )
        assert report.best_strategy == "checklist"
        assert len(report.strategy_rankings) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = ComparisonReport(
            report_id="cmp_002",
            domain="torts",
            strategies_compared=["causal_chain"],
            best_strategy="causal_chain",
            best_accuracy=0.75,
        )
        data = report.to_dict()
        assert data["domain"] == "torts"
        assert "generated_at" in data


class TestSelectionExplanation:
    """Tests for SelectionExplanation dataclass."""

    def test_explanation_creation(self):
        """Test creating a selection explanation."""
        explanation = SelectionExplanation(
            strategy_name="checklist",
            case_id="case_001",
            domain="contracts",
            reasons=["Best accuracy in domain", "Fast execution"],
            confidence=0.85,
            alternative_strategies=["rule_based", "dialectical"],
            domain_performance=0.9,
        )
        assert explanation.confidence == 0.85
        assert len(explanation.reasons) == 2

    def test_explain_method(self):
        """Test explain() generates readable text."""
        explanation = SelectionExplanation(
            strategy_name="checklist",
            case_id="case_001",
            domain="contracts",
            reasons=["Best accuracy"],
            confidence=0.85,
            domain_performance=0.9,
        )
        text = explanation.explain()
        assert "checklist" in text
        assert "contracts" in text
        assert "90" in text or "0.9" in text

    def test_to_dict(self):
        """Test conversion to dictionary."""
        explanation = SelectionExplanation(
            strategy_name="test",
            case_id="c1",
            domain="d1",
            reasons=[],
            confidence=0.5,
        )
        data = explanation.to_dict()
        assert data["strategy_name"] == "test"


class TestSimpleCase:
    """Tests for SimpleCase dataclass."""

    def test_case_creation(self):
        """Test creating a simple case."""
        case = SimpleCase(
            case_id="case_001",
            domain="contracts",
            facts=["Offer made", "Acceptance given"],
            ground_truth="enforceable",
        )
        assert case.case_id == "case_001"
        assert case.domain == "contracts"
        assert len(case.facts) == 2


class TestReasoningStrategies:
    """Tests for ReasoningStrategy implementations."""

    def test_checklist_strategy(self):
        """Test ChecklistStrategy."""
        strategy = ChecklistStrategy()
        assert strategy.name == "checklist"
        assert strategy.strategy_type == StrategyType.CHECKLIST
        assert "contracts" in strategy.applicable_domains

    def test_causal_chain_strategy(self):
        """Test CausalChainStrategy."""
        strategy = CausalChainStrategy()
        assert strategy.name == "causal_chain"
        assert strategy.strategy_type == StrategyType.CAUSAL_CHAIN
        assert "torts" in strategy.applicable_domains

    def test_balancing_test_strategy(self):
        """Test BalancingTestStrategy."""
        strategy = BalancingTestStrategy()
        assert strategy.name == "balancing_test"
        assert strategy.strategy_type == StrategyType.BALANCING_TEST
        assert "procedural" in strategy.applicable_domains

    def test_rule_based_strategy(self):
        """Test RuleBasedStrategy."""
        strategy = RuleBasedStrategy()
        assert strategy.name == "rule_based"
        assert strategy.strategy_type == StrategyType.RULE_BASED
        assert strategy.applicable_domains is None  # All domains

    def test_dialectical_strategy(self):
        """Test DialecticalStrategy."""
        strategy = DialecticalStrategy()
        assert strategy.name == "dialectical"
        assert strategy.strategy_type == StrategyType.DIALECTICAL
        assert strategy.characteristics.speed == "slow"

    def test_analogical_strategy(self):
        """Test AnalogicalStrategy."""
        strategy = AnalogicalStrategy()
        assert strategy.name == "analogical"
        assert strategy.strategy_type == StrategyType.ANALOGICAL
        assert "property_law" in strategy.applicable_domains

    def test_strategy_is_applicable(self):
        """Test is_applicable method."""
        checklist = ChecklistStrategy()
        assert checklist.is_applicable("contracts") is True
        assert checklist.is_applicable("torts") is False

        rule_based = RuleBasedStrategy()
        assert rule_based.is_applicable("any_domain") is True

    def test_strategy_apply(self):
        """Test apply method returns expected structure."""
        strategy = ChecklistStrategy()
        case = SimpleCase(case_id="c1", domain="contracts")
        result = strategy.apply(case)
        assert result["strategy"] == "checklist"
        assert result["domain"] == "contracts"

    def test_strategy_to_dict(self):
        """Test strategy conversion to dictionary."""
        strategy = ChecklistStrategy()
        data = strategy.to_dict()
        assert data["name"] == "checklist"
        assert data["strategy_type"] == "checklist"
        assert "characteristics" in data


class TestStrategyEvaluator:
    """Tests for StrategyEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a fresh evaluator."""
        return StrategyEvaluator()

    def test_evaluator_creation(self, evaluator):
        """Test evaluator initialization with default strategies."""
        assert len(evaluator.strategies) >= 5
        assert "checklist" in evaluator.strategies
        assert "rule_based" in evaluator.strategies

    def test_register_strategy(self, evaluator):
        """Test registering a new strategy."""

        class CustomStrategy(ReasoningStrategy):
            def apply(self, case):
                return {"strategy": self.name}

        custom = CustomStrategy(
            name="custom",
            strategy_type=StrategyType.DEFAULT,
            description="Custom strategy",
        )
        evaluator.register_strategy(custom)
        assert "custom" in evaluator.strategies

    def test_get_strategy(self, evaluator):
        """Test getting a strategy by name."""
        strategy = evaluator.get_strategy("checklist")
        assert strategy is not None
        assert strategy.name == "checklist"

        assert evaluator.get_strategy("nonexistent") is None

    def test_get_applicable_strategies(self, evaluator):
        """Test getting applicable strategies for a domain."""
        contracts_strategies = evaluator.get_applicable_strategies("contracts")
        assert len(contracts_strategies) >= 2
        assert any(s.name == "checklist" for s in contracts_strategies)

    def test_record_result(self, evaluator):
        """Test recording strategy results."""
        evaluator.record_result(
            strategy_name="checklist",
            domain="contracts",
            success=True,
            duration_ms=100.0,
            confidence=0.9,
        )
        evaluator.record_result(
            strategy_name="checklist",
            domain="contracts",
            success=False,
            duration_ms=150.0,
            confidence=0.6,
        )

        metrics = evaluator.evaluate_strategy(evaluator.strategies["checklist"], "contracts")
        assert metrics.total_cases == 2
        assert metrics.successful_cases == 1
        assert metrics.accuracy == 0.5

    def test_evaluate_strategy_no_data(self, evaluator):
        """Test evaluating strategy with no history."""
        metrics = evaluator.evaluate_strategy(evaluator.strategies["checklist"], "contracts")
        assert metrics.total_cases == 0
        assert metrics.accuracy == 0.0

    def test_evaluate_strategy_all_domains(self, evaluator):
        """Test evaluating strategy across all domains."""
        evaluator.record_result("checklist", "contracts", True, 100.0)
        evaluator.record_result("checklist", "statute_of_frauds", True, 120.0)
        evaluator.record_result("checklist", "contracts", False, 90.0)

        metrics = evaluator.evaluate_strategy(evaluator.strategies["checklist"], domain=None)
        assert metrics.total_cases == 3
        assert metrics.successful_cases == 2

    def test_compare_strategies(self, evaluator):
        """Test comparing multiple strategies."""
        # Record results for comparison
        for i in range(10):
            evaluator.record_result("checklist", "contracts", i < 8, 100.0)
            evaluator.record_result("rule_based", "contracts", i < 6, 150.0)

        report = evaluator.compare_strategies(["checklist", "rule_based"], "contracts")

        assert report.best_strategy == "checklist"
        assert report.best_accuracy == 0.8
        assert len(report.strategy_rankings) == 2
        assert report.strategy_rankings[0]["strategy_name"] == "checklist"

    def test_compare_strategies_empty(self, evaluator):
        """Test comparing strategies with no data."""
        report = evaluator.compare_strategies(["checklist", "rule_based"], "contracts")

        assert report.best_accuracy == 0.0
        assert len(report.recommendations) > 0


class TestStrategySelector:
    """Tests for StrategySelector class."""

    @pytest.fixture
    def evaluator_with_data(self):
        """Create evaluator with performance history."""
        evaluator = StrategyEvaluator()
        # Add history for checklist in contracts
        for i in range(20):
            evaluator.record_result("checklist", "contracts", i < 18, 100.0)
        # Add history for causal_chain in torts
        for i in range(15):
            evaluator.record_result("causal_chain", "torts", i < 12, 200.0)
        return evaluator

    @pytest.fixture
    def selector(self, evaluator_with_data):
        """Create selector with data."""
        return StrategySelector(evaluator_with_data)

    def test_selector_creation(self, selector):
        """Test selector initialization."""
        assert selector.selection_policy == "best_accuracy"
        assert selector.evaluator is not None

    def test_select_strategy_contracts(self, selector):
        """Test selecting strategy for contracts domain."""
        case = SimpleCase(case_id="c1", domain="contracts")
        strategy = selector.select_strategy(case)

        # Should select checklist (high accuracy in contracts)
        assert strategy.name == "checklist"

    def test_select_strategy_torts(self, selector):
        """Test selecting strategy for torts domain."""
        case = SimpleCase(case_id="c2", domain="torts")
        strategy = selector.select_strategy(case)

        # Should select causal_chain (best for torts)
        assert strategy.name == "causal_chain"

    def test_select_strategy_unknown_domain(self, selector):
        """Test selecting strategy for unknown domain."""
        case = SimpleCase(case_id="c3", domain="unknown")
        strategy = selector.select_strategy(case)

        # Should return a strategy (fallback)
        assert strategy is not None

    def test_select_strategy_limited_options(self, selector):
        """Test selecting from limited strategy options."""
        case = SimpleCase(case_id="c4", domain="contracts")
        strategy = selector.select_strategy(case, available_strategies=["rule_based"])

        assert strategy.name == "rule_based"

    def test_select_by_speed_policy(self, evaluator_with_data):
        """Test selection with fast policy."""
        selector = StrategySelector(evaluator_with_data, selection_policy="fast")
        case = SimpleCase(case_id="c5", domain="contracts")
        strategy = selector.select_strategy(case)

        # Should select a fast strategy
        assert strategy.characteristics.speed == "fast"

    def test_select_balanced_policy(self, evaluator_with_data):
        """Test selection with balanced policy."""
        selector = StrategySelector(evaluator_with_data, selection_policy="balanced")
        case = SimpleCase(case_id="c6", domain="contracts")
        strategy = selector.select_strategy(case)

        # Should return a strategy
        assert strategy is not None

    def test_set_domain_default(self, selector):
        """Test setting domain defaults."""
        selector.set_domain_default("custom_domain", "dialectical")

        # Verify it affects selection for new domain
        case = SimpleCase(case_id="c7", domain="custom_domain")
        strategy = selector.select_strategy(case)
        # Dialectical might be selected if applicable
        assert strategy is not None

    def test_explain_selection(self, selector):
        """Test explaining strategy selection."""
        case = SimpleCase(case_id="c8", domain="contracts")
        strategy = selector.select_strategy(case)
        explanation = selector.explain_selection(case, strategy)

        assert explanation.strategy_name == strategy.name
        assert explanation.case_id == "c8"
        assert explanation.domain == "contracts"
        assert len(explanation.reasons) > 0
        assert explanation.confidence > 0

    def test_explain_selection_with_performance(self, selector):
        """Test explanation includes performance data."""
        case = SimpleCase(case_id="c9", domain="contracts")
        strategy = selector.select_strategy(case)
        explanation = selector.explain_selection(case, strategy)

        assert explanation.domain_performance is not None
        assert explanation.domain_performance > 0

    def test_selection_callback(self, selector):
        """Test selection callback is called."""
        selections = []

        def on_select(case_id, domain, strategy):
            selections.append((case_id, domain, strategy.name))

        selector.set_callback(on_selection=on_select)

        case = SimpleCase(case_id="c10", domain="contracts")
        selector.select_strategy(case)

        assert len(selections) == 1
        assert selections[0][0] == "c10"


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_default_strategies(self):
        """Test getting default strategies."""
        strategies = get_default_strategies()
        assert len(strategies) >= 5
        assert "checklist" in strategies
        assert "causal_chain" in strategies
        assert "rule_based" in strategies

    def test_create_evaluator(self):
        """Test creating an evaluator."""
        evaluator = create_evaluator()
        assert isinstance(evaluator, StrategyEvaluator)
        assert len(evaluator.strategies) >= 5

    def test_create_evaluator_custom_strategies(self):
        """Test creating evaluator with custom strategies."""
        custom_strategies = {"rule_based": RuleBasedStrategy()}
        evaluator = create_evaluator(custom_strategies)
        assert len(evaluator.strategies) == 1

    def test_create_selector(self):
        """Test creating a selector."""
        selector = create_selector()
        assert isinstance(selector, StrategySelector)
        assert selector.selection_policy == "best_accuracy"

    def test_create_selector_custom_policy(self):
        """Test creating selector with custom policy."""
        selector = create_selector(policy="fast")
        assert selector.selection_policy == "fast"


class TestIntegration:
    """Integration tests for strategy framework."""

    def test_full_evaluation_cycle(self):
        """Test complete evaluation and selection cycle."""
        # Create evaluator and selector
        evaluator = StrategyEvaluator()
        selector = StrategySelector(evaluator)

        # Simulate case processing
        domains = ["contracts", "torts", "procedural"]
        cases_per_domain = 20

        for domain in domains:
            for i in range(cases_per_domain):
                case = SimpleCase(case_id=f"{domain}_{i}", domain=domain)

                # Select strategy
                strategy = selector.select_strategy(case)

                # Simulate result (strategies have different success rates)
                if strategy.name == "checklist":
                    success = i < 17  # 85% success
                elif strategy.name == "causal_chain":
                    success = i < 16  # 80% success
                else:
                    success = i < 14  # 70% success

                # Record result
                evaluator.record_result(
                    strategy_name=strategy.name,
                    domain=domain,
                    success=success,
                    duration_ms=100.0 + i * 5,
                    confidence=0.8 if success else 0.4,
                )

        # Verify accumulated data
        for domain in domains:
            applicable = evaluator.get_applicable_strategies(domain)
            assert len(applicable) >= 2

            # Compare strategies for this domain
            strategy_names = [s.name for s in applicable[:3]]
            report = evaluator.compare_strategies(strategy_names, domain)

            assert report.best_accuracy >= 0  # Valid accuracy value

    def test_strategy_improvement_over_random(self):
        """Test that strategy selection improves over random."""
        evaluator = StrategyEvaluator()

        # Add performance history showing checklist is best for contracts
        for i in range(30):
            evaluator.record_result("checklist", "contracts", i < 27, 100.0)  # 90%
            evaluator.record_result("rule_based", "contracts", i < 21, 100.0)  # 70%
            evaluator.record_result("dialectical", "contracts", i < 18, 300.0)  # 60%

        selector = StrategySelector(evaluator)

        # Selection should pick checklist consistently
        selected_strategies = []
        for i in range(10):
            case = SimpleCase(case_id=f"test_{i}", domain="contracts")
            strategy = selector.select_strategy(case)
            selected_strategies.append(strategy.name)

        # Should mostly select checklist
        checklist_count = sum(1 for s in selected_strategies if s == "checklist")
        assert checklist_count >= 8  # At least 80% should be checklist


class TestCounterfactualAnalysis:
    """Tests for CounterfactualAnalysis dataclass."""

    def test_counterfactual_creation(self):
        """Test creating a counterfactual analysis."""
        cf = CounterfactualAnalysis(
            alternative="rule_based",
            why_not_selected="Lower accuracy (70% vs 90%)",
            hypothetical_performance=0.7,
            confidence=0.8,
            comparison_factors=["accuracy_delta=20%", "slower"],
        )
        assert cf.alternative == "rule_based"
        assert cf.hypothetical_performance == 0.7
        assert len(cf.comparison_factors) == 2

    def test_counterfactual_to_dict(self):
        """Test conversion to dictionary."""
        cf = CounterfactualAnalysis(
            alternative="dialectical",
            why_not_selected="Too slow",
            hypothetical_performance=0.85,
            confidence=0.6,
        )
        data = cf.to_dict()
        assert data["alternative"] == "dialectical"
        assert data["hypothetical_performance"] == 0.85
        assert "comparison_factors" in data


class TestComparisonReportRankings:
    """Tests for ComparisonReport.rankings property."""

    def test_rankings_alias(self):
        """Test that rankings is an alias for strategy_rankings."""
        report = ComparisonReport(
            report_id="cmp_001",
            domain="contracts",
            strategies_compared=["checklist", "rule_based"],
            best_strategy="checklist",
            best_accuracy=0.9,
            strategy_rankings=[
                {"strategy_name": "checklist", "accuracy": 0.9},
                {"strategy_name": "rule_based", "accuracy": 0.85},
            ],
        )
        assert report.rankings == report.strategy_rankings
        assert len(report.rankings) == 2
        assert report.rankings[0]["strategy_name"] == "checklist"

    def test_rankings_in_to_dict(self):
        """Test that rankings appears in to_dict output."""
        report = ComparisonReport(
            report_id="cmp_002",
            domain="torts",
            strategies_compared=["causal_chain"],
            best_strategy="causal_chain",
            best_accuracy=0.75,
            strategy_rankings=[{"strategy_name": "causal_chain", "accuracy": 0.75}],
        )
        data = report.to_dict()
        assert "rankings" in data
        assert "strategy_rankings" in data
        assert data["rankings"] == data["strategy_rankings"]


class TestSelectionExplanationWithCounterfactuals:
    """Tests for SelectionExplanation with counterfactuals."""

    def test_explanation_with_counterfactuals(self):
        """Test creating explanation with counterfactuals."""
        cf = CounterfactualAnalysis(
            alternative="rule_based",
            why_not_selected="Lower accuracy",
            hypothetical_performance=0.7,
            confidence=0.8,
        )
        explanation = SelectionExplanation(
            strategy_name="checklist",
            case_id="case_001",
            domain="contracts",
            reasons=["Best accuracy"],
            confidence=0.9,
            counterfactuals=[cf],
        )
        assert len(explanation.counterfactuals) == 1
        assert explanation.counterfactuals[0].alternative == "rule_based"

    def test_explanation_to_dict_includes_counterfactuals(self):
        """Test that to_dict includes counterfactuals."""
        cf = CounterfactualAnalysis(
            alternative="dialectical",
            why_not_selected="Too slow",
            hypothetical_performance=0.8,
            confidence=0.7,
        )
        explanation = SelectionExplanation(
            strategy_name="checklist",
            case_id="c1",
            domain="contracts",
            reasons=[],
            confidence=0.8,
            counterfactuals=[cf],
        )
        data = explanation.to_dict()
        assert "counterfactuals" in data
        assert len(data["counterfactuals"]) == 1
        assert data["counterfactuals"][0]["alternative"] == "dialectical"

    def test_explain_method_includes_counterfactuals(self):
        """Test that explain() includes counterfactual info."""
        cf = CounterfactualAnalysis(
            alternative="rule_based",
            why_not_selected="Lower accuracy (70% vs 90%)",
            hypothetical_performance=0.7,
            confidence=0.8,
        )
        explanation = SelectionExplanation(
            strategy_name="checklist",
            case_id="c1",
            domain="contracts",
            reasons=["Best accuracy"],
            confidence=0.9,
            domain_performance=0.9,
            counterfactuals=[cf],
        )
        text = explanation.explain()
        assert "rule_based" in text
        assert "Lower accuracy" in text


class TestExplainSelectionWithoutArgs:
    """Tests for explain_selection() without arguments."""

    @pytest.fixture
    def selector_with_history(self):
        """Create selector with performance history."""
        evaluator = StrategyEvaluator()
        for i in range(20):
            evaluator.record_result("checklist", "contracts", i < 18, 100.0)
            evaluator.record_result("rule_based", "contracts", i < 14, 150.0)
        return StrategySelector(evaluator)

    def test_explain_without_args_after_selection(self, selector_with_history):
        """Test explain_selection() works without args after select_strategy."""
        case = SimpleCase(case_id="c1", domain="contracts")
        strategy = selector_with_history.select_strategy(case)

        # Now explain without args
        explanation = selector_with_history.explain_selection()

        assert explanation.strategy_name == strategy.name
        assert explanation.case_id == "c1"
        assert explanation.domain == "contracts"

    def test_explain_without_args_raises_before_selection(self):
        """Test that explain_selection() raises if no prior selection."""
        selector = StrategySelector(StrategyEvaluator())

        with pytest.raises(ValueError) as exc_info:
            selector.explain_selection()

        assert "No selection to explain" in str(exc_info.value)

    def test_explain_with_only_strategy_arg(self, selector_with_history):
        """Test explain_selection() with only strategy argument."""
        case = SimpleCase(case_id="c1", domain="contracts")
        selector_with_history.select_strategy(case)

        # Explain with different strategy
        rule_based = selector_with_history.evaluator.get_strategy("rule_based")
        explanation = selector_with_history.explain_selection(strategy=rule_based)

        assert explanation.strategy_name == "rule_based"
        assert explanation.case_id == "c1"

    def test_explain_with_only_case_arg(self, selector_with_history):
        """Test explain_selection() with only case argument."""
        case1 = SimpleCase(case_id="c1", domain="contracts")
        selector_with_history.select_strategy(case1)

        # Explain with different case
        case2 = SimpleCase(case_id="c2", domain="contracts")
        explanation = selector_with_history.explain_selection(case=case2)

        assert explanation.case_id == "c2"

    def test_explain_with_counterfactuals_enabled(self, selector_with_history):
        """Test that counterfactuals are included by default."""
        case = SimpleCase(case_id="c1", domain="contracts")
        selector_with_history.select_strategy(case)

        explanation = selector_with_history.explain_selection()

        assert len(explanation.counterfactuals) > 0
        # Counterfactuals should analyze alternatives
        alt_names = [cf.alternative for cf in explanation.counterfactuals]
        assert "rule_based" in alt_names or "dialectical" in alt_names

    def test_explain_with_counterfactuals_disabled(self, selector_with_history):
        """Test that counterfactuals can be disabled."""
        case = SimpleCase(case_id="c1", domain="contracts")
        selector_with_history.select_strategy(case)

        explanation = selector_with_history.explain_selection(include_counterfactuals=False)

        assert len(explanation.counterfactuals) == 0


class TestCounterfactualGeneration:
    """Tests for counterfactual analysis generation."""

    @pytest.fixture
    def selector_with_varied_performance(self):
        """Create selector with varied strategy performance."""
        evaluator = StrategyEvaluator()
        # Checklist: 90% accuracy, fast
        for i in range(20):
            evaluator.record_result("checklist", "contracts", i < 18, 100.0)
        # Rule-based: 70% accuracy, fast
        for i in range(20):
            evaluator.record_result("rule_based", "contracts", i < 14, 120.0)
        # Dialectical: 80% accuracy, slow
        for i in range(20):
            evaluator.record_result("dialectical", "contracts", i < 16, 300.0)
        return StrategySelector(evaluator)

    def test_counterfactual_accuracy_comparison(self, selector_with_varied_performance):
        """Test that counterfactuals compare accuracy."""
        case = SimpleCase(case_id="c1", domain="contracts")
        selector_with_varied_performance.select_strategy(case)

        explanation = selector_with_varied_performance.explain_selection()

        # Find rule_based counterfactual
        rule_cf = next(
            (cf for cf in explanation.counterfactuals if cf.alternative == "rule_based"),
            None,
        )
        if rule_cf:
            assert (
                "accuracy" in rule_cf.why_not_selected.lower()
                or "Lower" in rule_cf.why_not_selected
            )
            assert rule_cf.hypothetical_performance == 0.7

    def test_counterfactual_speed_comparison(self, selector_with_varied_performance):
        """Test that counterfactuals note speed differences."""
        case = SimpleCase(case_id="c1", domain="contracts")
        selector_with_varied_performance.select_strategy(case)

        explanation = selector_with_varied_performance.explain_selection()

        # Find dialectical counterfactual
        dial_cf = next(
            (cf for cf in explanation.counterfactuals if cf.alternative == "dialectical"),
            None,
        )
        if dial_cf:
            assert (
                "slow" in dial_cf.why_not_selected.lower() or "slower" in dial_cf.comparison_factors
            )

    def test_counterfactual_no_history(self):
        """Test counterfactuals for strategies with no history."""
        evaluator = StrategyEvaluator()
        # Only checklist has history
        for i in range(10):
            evaluator.record_result("checklist", "contracts", i < 9, 100.0)

        selector = StrategySelector(evaluator)
        case = SimpleCase(case_id="c1", domain="contracts")
        selector.select_strategy(case)

        explanation = selector.explain_selection()

        # Find an alternative with no history
        no_history_cf = next(
            (cf for cf in explanation.counterfactuals if cf.hypothetical_performance == 0.0),
            None,
        )
        if no_history_cf:
            assert (
                "No historical" in no_history_cf.why_not_selected
                or "no_history" in no_history_cf.comparison_factors
            )

    def test_counterfactual_confidence_calculation(self, selector_with_varied_performance):
        """Test that counterfactual confidence is calculated."""
        case = SimpleCase(case_id="c1", domain="contracts")
        selector_with_varied_performance.select_strategy(case)

        explanation = selector_with_varied_performance.explain_selection()

        for cf in explanation.counterfactuals:
            assert 0.0 <= cf.confidence <= 1.0
            # Alternatives with history should have higher confidence
            if cf.hypothetical_performance > 0:
                assert cf.confidence >= 0.5


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_explain_selection_with_both_args(self):
        """Test that original signature still works."""
        evaluator = StrategyEvaluator()
        for i in range(10):
            evaluator.record_result("checklist", "contracts", i < 8, 100.0)

        selector = StrategySelector(evaluator)
        case = SimpleCase(case_id="c1", domain="contracts")
        strategy = selector.select_strategy(case)

        # Original signature: explain_selection(case, strategy)
        explanation = selector.explain_selection(case, strategy)

        assert explanation.strategy_name == strategy.name
        assert explanation.case_id == "c1"
        assert len(explanation.reasons) > 0

    def test_comparison_report_both_attributes(self):
        """Test that both rankings and strategy_rankings work."""
        evaluator = StrategyEvaluator()
        for i in range(10):
            evaluator.record_result("checklist", "contracts", i < 8, 100.0)
            evaluator.record_result("rule_based", "contracts", i < 6, 100.0)

        report = evaluator.compare_strategies(["checklist", "rule_based"], "contracts")

        # Both should work
        assert len(report.strategy_rankings) == 2
        assert len(report.rankings) == 2
        assert report.rankings is report.strategy_rankings
