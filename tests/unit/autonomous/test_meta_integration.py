"""
Unit tests for MetaReasoningOrchestrator.

Tests the meta-reasoning integration including failure pattern tracking
and prompt improvement suggestions (issue #169).
"""

from loft.autonomous.config import MetaReasoningConfig
from loft.autonomous.meta_integration import MetaReasoningOrchestrator


class TestMetaReasoningOrchestratorInit:
    """Test MetaReasoningOrchestrator initialization."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        assert orchestrator.config == config
        assert orchestrator.current_cycle == 0
        assert orchestrator.cycle_history == []

    def test_initial_failure_patterns_empty(self):
        """Test that initial failure patterns are empty."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        assert orchestrator.get_processor_failure_patterns() == {}


class TestFailurePatternTracking:
    """Test failure pattern tracking from LLM processor (issue #169)."""

    def test_update_failure_patterns_adds_patterns(self):
        """Test that update_failure_patterns adds new patterns."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns({"unsafe_variable": 5, "syntax_error": 3})

        patterns = orchestrator.get_processor_failure_patterns()
        assert patterns["unsafe_variable"] == 5
        assert patterns["syntax_error"] == 3

    def test_update_failure_patterns_accumulates(self):
        """Test that update_failure_patterns accumulates counts."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns({"unsafe_variable": 5})
        orchestrator.update_failure_patterns({"unsafe_variable": 3, "syntax_error": 2})

        patterns = orchestrator.get_processor_failure_patterns()
        assert patterns["unsafe_variable"] == 8
        assert patterns["syntax_error"] == 2

    def test_update_failure_patterns_with_details(self):
        """Test that update_failure_patterns stores details."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        details = [
            {"case_id": "case_001", "category": "unsafe_variable"},
            {"case_id": "case_002", "category": "syntax_error"},
        ]
        orchestrator.update_failure_patterns({"unsafe_variable": 1}, details=details)

        # Details are stored internally (no public getter in current API)
        # Just verify no errors occur
        patterns = orchestrator.get_processor_failure_patterns()
        assert patterns["unsafe_variable"] == 1

    def test_clear_failure_patterns(self):
        """Test that clear_failure_patterns resets tracking."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns({"unsafe_variable": 5, "syntax_error": 3})
        assert orchestrator.get_processor_failure_patterns() != {}

        orchestrator.clear_failure_patterns()
        assert orchestrator.get_processor_failure_patterns() == {}


class TestPromptImprovementSuggestions:
    """Test prompt improvement suggestions based on failure patterns (issue #169)."""

    def test_suggest_prompt_improvements_empty_when_no_patterns(self):
        """Test that no suggestions when no failure patterns."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        suggestions = orchestrator.suggest_prompt_improvements()
        assert suggestions == []

    def test_suggest_prompt_improvements_unsafe_variable(self):
        """Test suggestions for unsafe variable failures."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns({"unsafe_variable": 10})

        suggestions = orchestrator.suggest_prompt_improvements()
        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "unsafe_variable"
        assert suggestions[0]["count"] == 10
        assert "variable safety" in suggestions[0]["description"].lower()

    def test_suggest_prompt_improvements_embedded_period(self):
        """Test suggestions for embedded period failures."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns({"embedded_period": 5})

        suggestions = orchestrator.suggest_prompt_improvements()
        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "embedded_period"
        assert "period" in suggestions[0]["description"].lower()

    def test_suggest_prompt_improvements_multiple_patterns(self):
        """Test suggestions for multiple failure patterns."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns(
            {
                "unsafe_variable": 10,
                "syntax_error": 5,
                "json_parse_error": 2,
            }
        )

        suggestions = orchestrator.suggest_prompt_improvements()
        assert len(suggestions) == 3
        # Should be sorted by count (descending)
        assert suggestions[0]["count"] == 10
        assert suggestions[1]["count"] == 5
        assert suggestions[2]["count"] == 2

    def test_suggest_prompt_improvements_priority_levels(self):
        """Test that priority is assigned based on percentage."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        # One dominant failure (>30%)
        orchestrator.update_failure_patterns(
            {
                "unsafe_variable": 50,  # >30% of 100 = high
                "syntax_error": 15,  # >10% of 100 = medium
                "unknown": 5,  # <10% of 100 = low
            }
        )

        suggestions = orchestrator.suggest_prompt_improvements()

        # Find the dominant failure pattern
        unsafe_suggestion = next(s for s in suggestions if s["category"] == "unsafe_variable")

        # The dominant one should be high priority (>30%)
        # 50/70 = 71.4% so high priority
        assert unsafe_suggestion["priority"] == "high"

    def test_suggest_prompt_improvements_has_recommended_action(self):
        """Test that each suggestion has a recommended action."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns(
            {
                "unsafe_variable": 5,
                "syntax_error": 3,
                "grounding_error": 2,
            }
        )

        suggestions = orchestrator.suggest_prompt_improvements()

        for suggestion in suggestions:
            assert "recommended_action" in suggestion
            assert len(suggestion["recommended_action"]) > 0


class TestMetricsSummary:
    """Test metrics summary includes failure pattern information."""

    def test_get_metrics_summary_includes_processor_failures(self):
        """Test that get_metrics_summary includes processor failure patterns."""
        config = MetaReasoningConfig()
        orchestrator = MetaReasoningOrchestrator(config=config)

        orchestrator.update_failure_patterns(
            {
                "unsafe_variable": 10,
                "syntax_error": 5,
            }
        )

        summary = orchestrator.get_metrics_summary()

        assert "processor_failure_patterns" in summary
        assert "total_processor_failures" in summary
        assert summary["processor_failure_patterns"]["unsafe_variable"] == 10
        assert summary["processor_failure_patterns"]["syntax_error"] == 5
        assert summary["total_processor_failures"] == 15
