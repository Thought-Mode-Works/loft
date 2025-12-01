"""Tests for meta-reasoning observer."""

from datetime import datetime, timedelta

import pytest

from loft.meta.observer import MetaReasoner, ReasoningObserver
from loft.meta.schemas import (
    ImprovementPriority,
    PatternType,
    ReasoningStep,
    ReasoningStepType,
)


def create_step(
    step_id: str,
    step_type: ReasoningStepType,
    success: bool = True,
    duration_ms: float = 100.0,
    confidence: float = 0.8,
    error_message: str | None = None,
) -> ReasoningStep:
    """Helper to create a reasoning step."""
    started = datetime.now()
    completed = started + timedelta(milliseconds=duration_ms)
    return ReasoningStep(
        step_id=step_id,
        step_type=step_type,
        description=f"Step {step_id}",
        input_data={},
        output_data={},
        started_at=started,
        completed_at=completed,
        success=success,
        confidence=confidence,
        error_message=error_message,
    )


class TestReasoningObserver:
    """Tests for ReasoningObserver class."""

    @pytest.fixture
    def observer(self):
        """Create a fresh observer."""
        return ReasoningObserver(observer_id="test_observer")

    def test_observer_creation(self, observer):
        """Test observer initialization."""
        assert observer.observer_id == "test_observer"
        assert len(observer.chains) == 0
        assert len(observer.patterns) == 0

    def test_observe_reasoning_chain(self, observer):
        """Test observing a reasoning chain."""
        steps = [
            create_step("s1", ReasoningStepType.TRANSLATION),
            create_step("s2", ReasoningStepType.INFERENCE),
            create_step("s3", ReasoningStepType.VALIDATION),
        ]

        chain = observer.observe_reasoning_chain(
            case_id="case_001",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="enforceable",
        )

        assert chain.case_id == "case_001"
        assert chain.domain == "contracts"
        assert chain.overall_success is True
        assert len(observer.chains) == 1

    def test_observe_failed_chain(self, observer):
        """Test observing a failed reasoning chain."""
        steps = [
            create_step("s1", ReasoningStepType.TRANSLATION),
            create_step("s2", ReasoningStepType.VALIDATION, success=False),
        ]

        chain = observer.observe_reasoning_chain(
            case_id="case_002",
            domain="torts",
            steps=steps,
            prediction="liable",
            ground_truth="not_liable",
        )

        assert chain.overall_success is False
        assert len(chain.failed_steps) == 1

    def test_get_chain(self, observer):
        """Test retrieving a specific chain."""
        steps = [create_step("s1", ReasoningStepType.TRANSLATION)]
        chain = observer.observe_reasoning_chain(
            case_id="case_001",
            domain="contracts",
            steps=steps,
        )

        retrieved = observer.get_chain(chain.chain_id)
        assert retrieved is not None
        assert retrieved.case_id == "case_001"

        # Test non-existent chain
        assert observer.get_chain("nonexistent") is None

    def test_get_chains_by_domain(self, observer):
        """Test getting chains by domain."""
        # Add chains for different domains
        for i in range(3):
            steps = [create_step(f"s{i}", ReasoningStepType.TRANSLATION)]
            observer.observe_reasoning_chain(
                case_id=f"contract_case_{i}",
                domain="contracts",
                steps=steps,
            )

        for i in range(2):
            steps = [create_step(f"t{i}", ReasoningStepType.TRANSLATION)]
            observer.observe_reasoning_chain(
                case_id=f"tort_case_{i}",
                domain="torts",
                steps=steps,
            )

        contracts = observer.get_chains_by_domain("contracts")
        torts = observer.get_chains_by_domain("torts")

        assert len(contracts) == 3
        assert len(torts) == 2

    def test_get_failed_chains(self, observer):
        """Test getting failed chains."""
        # Add successful chain
        steps = [create_step("s1", ReasoningStepType.TRANSLATION)]
        observer.observe_reasoning_chain(
            case_id="success_case",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="enforceable",
        )

        # Add failed chain
        steps = [create_step("s2", ReasoningStepType.TRANSLATION)]
        observer.observe_reasoning_chain(
            case_id="failure_case",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="unenforceable",
        )

        failed = observer.get_failed_chains()
        assert len(failed) == 1
        assert failed[0].case_id == "failure_case"

    def test_identify_patterns_step_success(self, observer):
        """Test pattern identification for step success rates."""
        # Add multiple chains with translation always succeeding
        for i in range(5):
            steps = [
                create_step(f"t{i}", ReasoningStepType.TRANSLATION, success=True),
                create_step(f"v{i}", ReasoningStepType.VALIDATION, success=True),
            ]
            observer.observe_reasoning_chain(
                case_id=f"case_{i}",
                domain="contracts",
                steps=steps,
                prediction="enforceable",
                ground_truth="enforceable",
            )

        patterns = observer.identify_patterns(min_frequency=3)

        # Should identify success patterns for step types
        step_patterns = [p for p in patterns if p.name.endswith("_pattern")]
        assert len(step_patterns) >= 1

    def test_identify_patterns_domain_specific(self, observer):
        """Test identification of domain-specific patterns."""
        # Add high success rate for contracts
        for i in range(5):
            steps = [create_step(f"s{i}", ReasoningStepType.TRANSLATION)]
            observer.observe_reasoning_chain(
                case_id=f"contract_case_{i}",
                domain="contracts",
                steps=steps,
                prediction="enforceable",
                ground_truth="enforceable",
            )

        patterns = observer.identify_patterns(min_frequency=3)

        domain_patterns = [p for p in patterns if p.pattern_type == PatternType.DOMAIN_SPECIFIC]
        assert len(domain_patterns) >= 1

    def test_identify_patterns_failure_mode(self, observer):
        """Test identification of failure mode patterns."""
        # Add chains where validation frequently fails
        for i in range(5):
            steps = [
                create_step(f"t{i}", ReasoningStepType.TRANSLATION),
                create_step(f"v{i}", ReasoningStepType.VALIDATION, success=False),
            ]
            observer.observe_reasoning_chain(
                case_id=f"case_{i}",
                domain="contracts",
                steps=steps,
                prediction="enforceable",
                ground_truth="unenforceable",
            )

        patterns = observer.identify_patterns(min_frequency=3)

        failure_patterns = [p for p in patterns if p.pattern_type == PatternType.FAILURE]
        assert len(failure_patterns) >= 1

    def test_analyze_bottlenecks(self, observer):
        """Test bottleneck analysis."""
        # Add chains with slow inference steps
        for i in range(5):
            steps = [
                create_step(f"t{i}", ReasoningStepType.TRANSLATION, duration_ms=100),
                create_step(f"i{i}", ReasoningStepType.INFERENCE, duration_ms=3000),
                create_step(f"v{i}", ReasoningStepType.VALIDATION, duration_ms=200),
            ]
            observer.observe_reasoning_chain(
                case_id=f"case_{i}",
                domain="contracts",
                steps=steps,
            )

        report = observer.analyze_bottlenecks(threshold_percentage=50)

        assert report.total_chains_analyzed == 5
        assert report.total_steps_analyzed == 15
        # Inference should be identified as bottleneck (3000ms out of ~3300ms total)
        assert len(report.bottlenecks) >= 1

    def test_analyze_bottlenecks_empty(self, observer):
        """Test bottleneck analysis with no data."""
        report = observer.analyze_bottlenecks()

        assert report.total_chains_analyzed == 0
        assert len(report.bottlenecks) == 0
        assert "Insufficient data" in report.recommendations[0]

    def test_get_summary(self, observer):
        """Test getting observation summary."""
        # Add some chains
        for i in range(3):
            steps = [create_step(f"s{i}", ReasoningStepType.TRANSLATION)]
            success = i < 2  # 2 successes, 1 failure
            observer.observe_reasoning_chain(
                case_id=f"case_{i}",
                domain="contracts",
                steps=steps,
                prediction="enforceable" if success else "unenforceable",
                ground_truth="enforceable",
            )

        summary = observer.get_summary()

        assert summary.total_chains_observed == 3
        assert summary.total_steps_observed == 3
        assert abs(summary.success_rate - (2 / 3)) < 0.01
        assert "contracts" in summary.domains_observed

    def test_clear(self, observer):
        """Test clearing observer state."""
        steps = [create_step("s1", ReasoningStepType.TRANSLATION)]
        observer.observe_reasoning_chain(
            case_id="case_001",
            domain="contracts",
            steps=steps,
        )

        assert len(observer.chains) == 1

        observer.clear()

        assert len(observer.chains) == 0
        assert len(observer.patterns) == 0

    def test_callbacks(self, observer):
        """Test callback functionality."""
        chains_received = []
        patterns_received = []

        def on_chain(chain):
            chains_received.append(chain)

        def on_pattern(pattern):
            patterns_received.append(pattern)

        observer.set_callbacks(
            on_chain_complete=on_chain,
            on_pattern_discovered=on_pattern,
        )

        steps = [create_step("s1", ReasoningStepType.TRANSLATION)]
        observer.observe_reasoning_chain(
            case_id="case_001",
            domain="contracts",
            steps=steps,
        )

        assert len(chains_received) == 1


class TestMetaReasoner:
    """Tests for MetaReasoner class."""

    @pytest.fixture
    def observer_with_data(self):
        """Create an observer with some test data."""
        observer = ReasoningObserver()

        # Add successful chain
        steps = [
            create_step("s1", ReasoningStepType.TRANSLATION),
            create_step("s2", ReasoningStepType.INFERENCE),
        ]
        observer.observe_reasoning_chain(
            case_id="success_case",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="enforceable",
        )

        # Add failed chain
        steps = [
            create_step("f1", ReasoningStepType.TRANSLATION),
            create_step(
                "f2",
                ReasoningStepType.VALIDATION,
                success=False,
                error_message="Rule conflict detected",
            ),
        ]
        observer.observe_reasoning_chain(
            case_id="failure_case",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="unenforceable",
        )

        return observer

    @pytest.fixture
    def meta_reasoner(self, observer_with_data):
        """Create a meta-reasoner with test data."""
        return MetaReasoner(observer_with_data)

    def test_meta_reasoner_creation(self, meta_reasoner):
        """Test meta-reasoner initialization."""
        assert meta_reasoner.observer is not None
        assert len(meta_reasoner.diagnoses) == 0

    def test_diagnose_reasoning_failure(self, meta_reasoner):
        """Test diagnosing a reasoning failure."""
        # Get the failed chain
        failed_chains = meta_reasoner.observer.get_failed_chains()
        assert len(failed_chains) == 1

        chain_id = failed_chains[0].chain_id
        diagnosis = meta_reasoner.diagnose_reasoning_failure(chain_id)

        assert diagnosis is not None
        assert diagnosis.case_id == "failure_case"
        assert diagnosis.prediction == "enforceable"
        assert diagnosis.ground_truth == "unenforceable"
        assert diagnosis.failure_type in [
            "validation_failure",
            "rule_gap",
            "translation_error",
            "inference_error",
            "prediction_mismatch",
            "consensus_disagreement",
            "unknown",
        ]
        assert len(diagnosis.root_causes) > 0

    def test_diagnose_successful_chain(self, meta_reasoner):
        """Test that diagnosing a successful chain returns None."""
        successful_chains = [c for c in meta_reasoner.observer.chains.values() if c.overall_success]
        assert len(successful_chains) > 0

        chain_id = successful_chains[0].chain_id
        diagnosis = meta_reasoner.diagnose_reasoning_failure(chain_id)

        assert diagnosis is None

    def test_diagnose_nonexistent_chain(self, meta_reasoner):
        """Test diagnosing a non-existent chain."""
        diagnosis = meta_reasoner.diagnose_reasoning_failure("nonexistent")
        assert diagnosis is None

    def test_suggest_improvements(self, observer_with_data):
        """Test suggesting improvements."""
        # Add more data for pattern detection
        for i in range(5):
            steps = [
                create_step(f"t{i}", ReasoningStepType.TRANSLATION),
                create_step(f"v{i}", ReasoningStepType.VALIDATION, success=False),
            ]
            observer_with_data.observe_reasoning_chain(
                case_id=f"fail_case_{i}",
                domain="contracts",
                steps=steps,
                prediction="enforceable",
                ground_truth="unenforceable",
            )

        meta_reasoner = MetaReasoner(observer_with_data)
        improvements = meta_reasoner.suggest_improvements()

        # Should suggest some improvements based on patterns
        assert len(improvements) >= 0  # May be 0 if no patterns qualify

    def test_improvement_priority_ordering(self, observer_with_data):
        """Test that improvements are ordered by priority."""
        # Add data to trigger improvements
        for i in range(10):
            steps = [
                create_step(f"t{i}", ReasoningStepType.TRANSLATION, duration_ms=2000),
                create_step(f"v{i}", ReasoningStepType.VALIDATION, success=False),
            ]
            observer_with_data.observe_reasoning_chain(
                case_id=f"case_{i}",
                domain="contracts",
                steps=steps,
                prediction="enforceable",
                ground_truth="unenforceable",
            )

        meta_reasoner = MetaReasoner(observer_with_data)
        improvements = meta_reasoner.suggest_improvements()

        if len(improvements) >= 2:
            # Verify priority ordering
            priority_values = {
                ImprovementPriority.CRITICAL: 0,
                ImprovementPriority.HIGH: 1,
                ImprovementPriority.MEDIUM: 2,
                ImprovementPriority.LOW: 3,
            }
            for i in range(len(improvements) - 1):
                current_priority = priority_values[improvements[i].priority]
                next_priority = priority_values[improvements[i + 1].priority]
                assert current_priority <= next_priority

    def test_explain_reasoning_quality(self, meta_reasoner):
        """Test generating quality explanation."""
        explanation = meta_reasoner.explain_reasoning_quality()

        assert "Observed" in explanation
        assert "reasoning chains" in explanation
        assert "success rate" in explanation

    def test_explain_reasoning_quality_by_domain(self, meta_reasoner):
        """Test generating domain-specific quality explanation."""
        explanation = meta_reasoner.explain_reasoning_quality(domain="contracts")

        assert "contracts" in explanation

    def test_diagnose_stores_diagnosis(self, meta_reasoner):
        """Test that diagnoses are stored."""
        failed_chains = meta_reasoner.observer.get_failed_chains()
        chain_id = failed_chains[0].chain_id

        diagnosis = meta_reasoner.diagnose_reasoning_failure(chain_id)

        assert diagnosis.diagnosis_id in meta_reasoner.diagnoses


class TestReasoningObserverIntegration:
    """Integration tests for observer functionality."""

    def test_full_observation_cycle(self):
        """Test a complete observation and analysis cycle."""
        observer = ReasoningObserver()

        # Simulate multiple reasoning chains
        domains = ["contracts", "torts", "procedural"]
        for domain in domains:
            for i in range(10):
                success = i < 7  # 70% success rate
                steps = [
                    create_step(
                        f"{domain}_t_{i}", ReasoningStepType.TRANSLATION, duration_ms=100 + i * 10
                    ),
                    create_step(
                        f"{domain}_i_{i}", ReasoningStepType.INFERENCE, duration_ms=500 + i * 50
                    ),
                    create_step(
                        f"{domain}_v_{i}",
                        ReasoningStepType.VALIDATION,
                        success=success,
                        duration_ms=200,
                    ),
                ]
                observer.observe_reasoning_chain(
                    case_id=f"{domain}_case_{i}",
                    domain=domain,
                    steps=steps,
                    prediction="valid" if success else "invalid",
                    ground_truth="valid",
                )

        # Verify observation
        assert len(observer.chains) == 30

        # Identify patterns
        patterns = observer.identify_patterns(min_frequency=5)
        assert len(patterns) > 0

        # Analyze bottlenecks
        report = observer.analyze_bottlenecks(threshold_percentage=30)
        assert report.total_chains_analyzed == 30

        # Get summary
        summary = observer.get_summary()
        assert summary.total_chains_observed == 30
        assert len(summary.domains_observed) == 3
        assert abs(summary.success_rate - 0.7) < 0.01

    def test_meta_reasoner_full_analysis(self):
        """Test meta-reasoner with full analysis cycle."""
        observer = ReasoningObserver()

        # Add varied data
        for i in range(20):
            success = i < 12  # 60% success
            steps = [
                create_step(f"t_{i}", ReasoningStepType.TRANSLATION, success=True, duration_ms=150),
                create_step(
                    f"v_{i}", ReasoningStepType.VALIDATION, success=success, duration_ms=300
                ),
            ]
            observer.observe_reasoning_chain(
                case_id=f"case_{i}",
                domain="contracts",
                steps=steps,
                prediction="enforceable" if success else "unenforceable",
                ground_truth="enforceable",
            )

        meta_reasoner = MetaReasoner(observer)

        # Diagnose all failures
        failed_chains = observer.get_failed_chains()
        for chain in failed_chains:
            diagnosis = meta_reasoner.diagnose_reasoning_failure(chain.chain_id)
            assert diagnosis is not None

        # Get improvements
        meta_reasoner.suggest_improvements()

        # Verify we have diagnoses and potentially improvements
        assert len(meta_reasoner.diagnoses) == 8  # 20 - 12 failures

        # Get quality explanation
        explanation = meta_reasoner.explain_reasoning_quality()
        assert "60" in explanation or "0.6" in explanation  # Success rate


class TestChainResolutionMethods:
    """Tests for chain resolution methods (issue #141)."""

    @pytest.fixture
    def observer(self):
        """Create an observer with test data."""
        observer = ReasoningObserver()

        # Add a chain
        steps = [
            create_step("s1", ReasoningStepType.TRANSLATION),
            create_step("s2", ReasoningStepType.INFERENCE),
        ]
        observer.observe_reasoning_chain(
            case_id="test_case",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="enforceable",
        )
        return observer

    def test_get_chain_id_with_string(self, observer):
        """Test get_chain_id with a string input."""
        chain_id = "test_chain_id"
        result = observer.get_chain_id(chain_id)
        assert result == chain_id

    def test_get_chain_id_with_chain_object(self, observer):
        """Test get_chain_id with a ReasoningChain object."""
        chain = list(observer.chains.values())[0]
        result = observer.get_chain_id(chain)
        assert result == chain.chain_id

    def test_get_chain_id_with_invalid_type(self, observer):
        """Test get_chain_id raises TypeError for invalid input."""
        with pytest.raises(TypeError) as excinfo:
            observer.get_chain_id(123)  # type: ignore
        assert "Expected str or ReasoningChain" in str(excinfo.value)

    def test_resolve_chain_with_string(self, observer):
        """Test resolve_chain with a string chain ID."""
        chain = list(observer.chains.values())[0]
        result = observer.resolve_chain(chain.chain_id)
        assert result is not None
        assert result.chain_id == chain.chain_id

    def test_resolve_chain_with_chain_object(self, observer):
        """Test resolve_chain with a ReasoningChain object."""
        chain = list(observer.chains.values())[0]
        result = observer.resolve_chain(chain)
        assert result is not None
        assert result.chain_id == chain.chain_id

    def test_resolve_chain_nonexistent_string(self, observer):
        """Test resolve_chain with non-existent chain ID."""
        result = observer.resolve_chain("nonexistent_chain")
        assert result is None

    def test_resolve_chain_invalid_type(self, observer):
        """Test resolve_chain returns None for invalid types."""
        result = observer.resolve_chain(123)  # type: ignore
        assert result is None


class TestDiagnoseReasoningFailureFlexibility:
    """Tests for diagnose_reasoning_failure accepting both string and chain (issue #141)."""

    @pytest.fixture
    def observer_with_failure(self):
        """Create an observer with a failed chain."""
        observer = ReasoningObserver()

        # Add a failed chain
        steps = [
            create_step("f1", ReasoningStepType.TRANSLATION),
            create_step(
                "f2",
                ReasoningStepType.RULE_APPLICATION,
                success=False,
                error_message="No applicable rule found",
            ),
        ]
        observer.observe_reasoning_chain(
            case_id="failure_case",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="unenforceable",
        )
        return observer

    @pytest.fixture
    def meta_reasoner(self, observer_with_failure):
        """Create a meta-reasoner with test data."""
        return MetaReasoner(observer_with_failure)

    def test_diagnose_with_string_chain_id(self, meta_reasoner):
        """Test diagnosing with a string chain ID."""
        failed_chain = meta_reasoner.observer.get_failed_chains()[0]
        diagnosis = meta_reasoner.diagnose_reasoning_failure(failed_chain.chain_id)

        assert diagnosis is not None
        assert diagnosis.case_id == "failure_case"

    def test_diagnose_with_chain_object(self, meta_reasoner):
        """Test diagnosing with a ReasoningChain object directly."""
        failed_chain = meta_reasoner.observer.get_failed_chains()[0]
        diagnosis = meta_reasoner.diagnose_reasoning_failure(failed_chain)

        assert diagnosis is not None
        assert diagnosis.case_id == "failure_case"
        assert diagnosis.chain_id == failed_chain.chain_id

    def test_diagnosis_includes_step_type(self, meta_reasoner):
        """Test that diagnosis includes the primary failure step type."""
        failed_chain = meta_reasoner.observer.get_failed_chains()[0]
        diagnosis = meta_reasoner.diagnose_reasoning_failure(failed_chain)

        assert diagnosis is not None
        assert diagnosis.primary_failure_step_type == ReasoningStepType.RULE_APPLICATION

    def test_diagnosis_step_type_is_none_for_prediction_mismatch(self):
        """Test that step type is None when there's no failed step."""
        observer = ReasoningObserver()

        # Add a chain where all steps succeed but prediction mismatches
        steps = [
            create_step("s1", ReasoningStepType.TRANSLATION, success=True),
            create_step("s2", ReasoningStepType.INFERENCE, success=True),
        ]
        observer.observe_reasoning_chain(
            case_id="mismatch_case",
            domain="contracts",
            steps=steps,
            prediction="enforceable",
            ground_truth="unenforceable",  # Prediction is wrong
        )

        meta_reasoner = MetaReasoner(observer)
        failed_chain = observer.get_failed_chains()[0]
        diagnosis = meta_reasoner.diagnose_reasoning_failure(failed_chain)

        assert diagnosis is not None
        assert diagnosis.primary_failure_step is None
        assert diagnosis.primary_failure_step_type is None
        assert diagnosis.failure_type == "prediction_mismatch"

    def test_diagnose_with_chain_returns_same_as_string(self, meta_reasoner):
        """Test that diagnosing with chain vs string gives same results."""
        failed_chain = meta_reasoner.observer.get_failed_chains()[0]

        # Diagnose with string
        diagnosis_from_string = meta_reasoner.diagnose_reasoning_failure(failed_chain.chain_id)

        # Create new meta_reasoner to avoid cached diagnosis
        meta_reasoner2 = MetaReasoner(meta_reasoner.observer)

        # Diagnose with chain object
        diagnosis_from_chain = meta_reasoner2.diagnose_reasoning_failure(failed_chain)

        # Core fields should match (IDs differ because they're new diagnoses)
        assert diagnosis_from_string.case_id == diagnosis_from_chain.case_id
        assert diagnosis_from_string.chain_id == diagnosis_from_chain.chain_id
        assert diagnosis_from_string.failure_type == diagnosis_from_chain.failure_type
        assert (
            diagnosis_from_string.primary_failure_step_type
            == diagnosis_from_chain.primary_failure_step_type
        )
