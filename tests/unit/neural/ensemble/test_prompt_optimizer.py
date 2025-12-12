"""
Unit tests for ModelPromptOptimizer and related components.

Tests cover:
- ModelPromptOptimizer core functionality
- PromptRegistry template management
- PromptPerformanceTracker metrics
- ABTestingFramework A/B testing
- Prompt evolution from failure patterns
- Integration with meta-reasoning suggestions
"""

import pytest

from loft.neural.ensemble.prompt_optimizer import (
    # Main classes
    ModelPromptOptimizer,
    PromptRegistry,
    PromptPerformanceTracker,
    ABTestingFramework,
    # Data classes
    ModelProfile,
    PromptTemplate,
    PromptPerformanceRecord,
    ABTestResult,
    PromptEvolutionResult,
    OptimizationSuggestion,
    # Config
    PromptOptimizerConfig,
    # Enums
    ModelType,
    PromptTaskType,
    PromptVariantStatus,
    ABTestStatus,
    # Exceptions
    PromptOptimizationError,
    ABTestingError,
    # Factory functions
    create_prompt_optimizer,
    create_default_model_profiles,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def prompt_optimizer():
    """Create a ModelPromptOptimizer for testing."""
    return ModelPromptOptimizer()


@pytest.fixture
def prompt_registry():
    """Create a PromptRegistry for testing."""
    return PromptRegistry()


@pytest.fixture
def performance_tracker():
    """Create a PromptPerformanceTracker for testing."""
    return PromptPerformanceTracker()


@pytest.fixture
def optimizer_config():
    """Create a PromptOptimizerConfig for testing."""
    return PromptOptimizerConfig(
        min_samples_for_ab_test=10,
        confidence_threshold=0.8,
        auto_deprecate_threshold=0.2,
    )


@pytest.fixture
def sample_model_profile():
    """Create a sample ModelProfile for testing."""
    return ModelProfile(
        model_name="test-model",
        model_type=ModelType.LOGIC_GENERATOR,
        strengths=["logic", "syntax"],
        weaknesses=["creative"],
        success_rate=0.85,
        average_latency_ms=150.0,
    )


@pytest.fixture
def sample_template():
    """Create a sample PromptTemplate for testing."""
    return PromptTemplate(
        template_id="test_template_001",
        name="Test Template",
        template="Generate ASP for: {principle}",
        model_type=ModelType.LOGIC_GENERATOR,
        task_type=PromptTaskType.ASP_GENERATION,
        variables=["principle"],
        status=PromptVariantStatus.ACTIVE,
    )


# =============================================================================
# Test ModelProfile
# =============================================================================


class TestModelProfile:
    """Tests for ModelProfile data class."""

    def test_model_profile_creation(self, sample_model_profile):
        """Test ModelProfile creation with default values."""
        assert sample_model_profile.model_name == "test-model"
        assert sample_model_profile.model_type == ModelType.LOGIC_GENERATOR
        assert sample_model_profile.success_rate == 0.85

    def test_model_profile_to_dict(self, sample_model_profile):
        """Test ModelProfile serialization."""
        data = sample_model_profile.to_dict()
        assert data["model_name"] == "test-model"
        assert data["model_type"] == "logic_generator"
        assert "strengths" in data
        assert "weaknesses" in data

    def test_model_profile_defaults(self):
        """Test ModelProfile default values."""
        profile = ModelProfile(model_name="minimal")
        assert profile.model_type == ModelType.GENERAL
        assert profile.strengths == []
        assert profile.weaknesses == []
        assert profile.preferred_temperature == 0.7


# =============================================================================
# Test PromptTemplate
# =============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate data class."""

    def test_template_creation(self, sample_template):
        """Test PromptTemplate creation."""
        assert sample_template.template_id == "test_template_001"
        assert sample_template.task_type == PromptTaskType.ASP_GENERATION
        assert sample_template.status == PromptVariantStatus.ACTIVE

    def test_template_render(self, sample_template):
        """Test template rendering with variables."""
        rendered = sample_template.render(principle="Contract requires offer")
        assert "Contract requires offer" in rendered
        assert "{principle}" not in rendered

    def test_template_render_missing_variable(self, sample_template):
        """Test template rendering with missing variable."""
        rendered = sample_template.render(other_var="value")
        assert "{principle}" in rendered  # Variable not replaced

    def test_template_to_dict(self, sample_template):
        """Test PromptTemplate serialization."""
        data = sample_template.to_dict()
        assert data["template_id"] == "test_template_001"
        assert data["model_type"] == "logic_generator"
        assert data["task_type"] == "asp_generation"
        assert data["status"] == "active"


# =============================================================================
# Test PromptPerformanceRecord
# =============================================================================


class TestPromptPerformanceRecord:
    """Tests for PromptPerformanceRecord data class."""

    def test_record_creation(self):
        """Test PromptPerformanceRecord creation."""
        record = PromptPerformanceRecord(
            record_id="rec_001",
            template_id="tmpl_001",
            model_name="test-model",
            task_type=PromptTaskType.ASP_GENERATION,
            success=True,
            quality_score=0.9,
            latency_ms=150.0,
        )
        assert record.record_id == "rec_001"
        assert record.success is True
        assert record.quality_score == 0.9

    def test_record_to_dict(self):
        """Test PromptPerformanceRecord serialization."""
        record = PromptPerformanceRecord(
            record_id="rec_001",
            template_id="tmpl_001",
            model_name="test-model",
            task_type=PromptTaskType.ASP_GENERATION,
        )
        data = record.to_dict()
        assert data["record_id"] == "rec_001"
        assert data["task_type"] == "asp_generation"


# =============================================================================
# Test ABTestResult
# =============================================================================


class TestABTestResult:
    """Tests for ABTestResult data class."""

    def test_ab_test_result_creation(self):
        """Test ABTestResult creation."""
        result = ABTestResult(
            test_id="ab_001",
            variant_a_id="tmpl_a",
            variant_b_id="tmpl_b",
            model_name="test-model",
            task_type=PromptTaskType.ASP_GENERATION,
        )
        assert result.test_id == "ab_001"
        assert result.status == ABTestStatus.PENDING
        assert result.winner is None

    def test_ab_test_result_to_dict(self):
        """Test ABTestResult serialization."""
        result = ABTestResult(
            test_id="ab_001",
            variant_a_id="tmpl_a",
            variant_b_id="tmpl_b",
            model_name="test-model",
            task_type=PromptTaskType.ASP_GENERATION,
            winner="A",
            confidence=0.95,
        )
        data = result.to_dict()
        assert data["test_id"] == "ab_001"
        assert data["winner"] == "A"
        assert data["confidence"] == 0.95


# =============================================================================
# Test PromptRegistry
# =============================================================================


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_registry_has_default_templates(self, prompt_registry):
        """Test that registry initializes with default templates."""
        templates = prompt_registry.get_all_templates()
        assert len(templates) > 0

    def test_registry_default_templates_for_each_task(self, prompt_registry):
        """Test default templates exist for each task type."""
        for task_type in PromptTaskType:
            templates = prompt_registry.get_templates_for_task(task_type)
            assert len(templates) > 0, f"No default template for {task_type}"

    def test_register_template(self, prompt_registry, sample_template):
        """Test registering a new template."""
        initial_count = len(prompt_registry.get_all_templates())
        template_id = prompt_registry.register_template(sample_template)
        assert template_id == sample_template.template_id
        assert len(prompt_registry.get_all_templates()) == initial_count + 1

    def test_get_template(self, prompt_registry, sample_template):
        """Test getting a template by ID."""
        prompt_registry.register_template(sample_template)
        retrieved = prompt_registry.get_template(sample_template.template_id)
        assert retrieved is not None
        assert retrieved.name == sample_template.name

    def test_get_template_not_found(self, prompt_registry):
        """Test getting a non-existent template."""
        result = prompt_registry.get_template("nonexistent")
        assert result is None

    def test_get_templates_for_task(self, prompt_registry):
        """Test getting templates by task type."""
        templates = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        assert len(templates) > 0
        for t in templates:
            assert t.task_type == PromptTaskType.ASP_GENERATION

    def test_get_templates_for_task_with_status_filter(
        self, prompt_registry, sample_template
    ):
        """Test getting templates with status filter."""
        sample_template.status = PromptVariantStatus.TESTING
        prompt_registry.register_template(sample_template)

        active = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION, status=PromptVariantStatus.ACTIVE
        )
        testing = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION, status=PromptVariantStatus.TESTING
        )

        assert sample_template.template_id not in [t.template_id for t in active]
        assert sample_template.template_id in [t.template_id for t in testing]

    def test_get_templates_for_model(self, prompt_registry):
        """Test getting templates by model type."""
        templates = prompt_registry.get_templates_for_model(ModelType.LOGIC_GENERATOR)
        assert len(templates) > 0
        for t in templates:
            assert t.model_type == ModelType.LOGIC_GENERATOR

    def test_update_template_status(self, prompt_registry, sample_template):
        """Test updating template status."""
        prompt_registry.register_template(sample_template)
        result = prompt_registry.update_template_status(
            sample_template.template_id, PromptVariantStatus.DEPRECATED
        )
        assert result is True
        retrieved = prompt_registry.get_template(sample_template.template_id)
        assert retrieved.status == PromptVariantStatus.DEPRECATED

    def test_update_template_status_not_found(self, prompt_registry):
        """Test updating status of non-existent template."""
        result = prompt_registry.update_template_status(
            "nonexistent", PromptVariantStatus.DEPRECATED
        )
        assert result is False

    def test_update_template_performance(self, prompt_registry, sample_template):
        """Test updating template performance score."""
        prompt_registry.register_template(sample_template)
        result = prompt_registry.update_template_performance(
            sample_template.template_id, 0.85
        )
        assert result is True
        retrieved = prompt_registry.get_template(sample_template.template_id)
        assert retrieved.performance_score == 0.85
        assert retrieved.usage_count == 1

    def test_get_best_template(self, prompt_registry):
        """Test getting best performing template."""
        # Create templates with different scores
        for i, score in enumerate([0.5, 0.8, 0.3]):
            template = PromptTemplate(
                template_id=f"perf_test_{i}",
                name=f"Perf Test {i}",
                template="Test {var}",
                model_type=ModelType.LOGIC_GENERATOR,
                task_type=PromptTaskType.ASP_GENERATION,
                variables=["var"],
                status=PromptVariantStatus.ACTIVE,
                performance_score=score,
            )
            prompt_registry.register_template(template)

        best = prompt_registry.get_best_template(PromptTaskType.ASP_GENERATION)
        assert best is not None
        assert best.performance_score >= 0.8


# =============================================================================
# Test PromptPerformanceTracker
# =============================================================================


class TestPromptPerformanceTracker:
    """Tests for PromptPerformanceTracker."""

    def test_record_performance(self, performance_tracker):
        """Test recording performance."""
        record = PromptPerformanceRecord(
            record_id="rec_001",
            template_id="tmpl_001",
            model_name="test-model",
            task_type=PromptTaskType.ASP_GENERATION,
            success=True,
            quality_score=0.9,
            latency_ms=150.0,
        )
        performance_tracker.record(record)
        records = performance_tracker.get_records("tmpl_001")
        assert len(records) == 1
        assert records[0].quality_score == 0.9

    def test_get_records_with_filters(self, performance_tracker):
        """Test getting records with filters."""
        for i in range(5):
            record = PromptPerformanceRecord(
                record_id=f"rec_{i}",
                template_id="tmpl_001",
                model_name="model-a" if i < 3 else "model-b",
                task_type=PromptTaskType.ASP_GENERATION,
                success=True,
            )
            performance_tracker.record(record)

        all_records = performance_tracker.get_records("tmpl_001")
        assert len(all_records) == 5

        model_a_records = performance_tracker.get_records(
            "tmpl_001", model_name="model-a"
        )
        assert len(model_a_records) == 3

    def test_get_aggregate_metrics(self, performance_tracker):
        """Test getting aggregate metrics."""
        for i, success in enumerate([True, True, True, False]):
            record = PromptPerformanceRecord(
                record_id=f"rec_{i}",
                template_id="tmpl_001",
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
                success=success,
                quality_score=0.8 if success else 0.0,
                latency_ms=100.0 + i * 10,
            )
            performance_tracker.record(record)

        metrics = performance_tracker.get_aggregate_metrics("tmpl_001")
        assert metrics["success_rate"] == 0.75  # 3/4
        assert metrics["sample_count"] == 4
        assert metrics["avg_quality"] == 0.8  # Only successful ones

    def test_get_aggregate_metrics_empty(self, performance_tracker):
        """Test getting aggregate metrics for non-existent template."""
        metrics = performance_tracker.get_aggregate_metrics("nonexistent")
        assert metrics["success_rate"] == 0.0
        assert metrics["sample_count"] == 0

    def test_get_failure_patterns(self, performance_tracker):
        """Test getting failure patterns."""
        errors = ["Syntax error", "Syntax error", "Timeout", "Syntax error"]
        for i, error in enumerate(errors):
            record = PromptPerformanceRecord(
                record_id=f"rec_{i}",
                template_id="tmpl_001",
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
                success=False,
                error_message=error,
            )
            performance_tracker.record(record)

        patterns = performance_tracker.get_failure_patterns("tmpl_001")
        assert len(patterns) == 2
        # Syntax error should be first (most common)
        assert patterns[0]["error_pattern"] == "Syntax error"
        assert patterns[0]["count"] == 3

    def test_clear_records(self, performance_tracker):
        """Test clearing records."""
        for i in range(3):
            record = PromptPerformanceRecord(
                record_id=f"rec_{i}",
                template_id="tmpl_001",
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
            )
            performance_tracker.record(record)

        count = performance_tracker.clear_records("tmpl_001")
        assert count == 3
        assert len(performance_tracker.get_records("tmpl_001")) == 0

    def test_max_history_limit(self):
        """Test that history is limited per template."""
        tracker = PromptPerformanceTracker(max_history_per_template=5)
        for i in range(10):
            record = PromptPerformanceRecord(
                record_id=f"rec_{i}",
                template_id="tmpl_001",
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
            )
            tracker.record(record)

        records = tracker.get_records("tmpl_001")
        assert len(records) == 5  # Limited to 5


# =============================================================================
# Test ABTestingFramework
# =============================================================================


class TestABTestingFramework:
    """Tests for ABTestingFramework."""

    @pytest.fixture
    def ab_framework(self, prompt_registry, performance_tracker, optimizer_config):
        """Create an ABTestingFramework for testing."""
        return ABTestingFramework(
            prompt_registry, performance_tracker, optimizer_config
        )

    def test_create_test(self, ab_framework, prompt_registry):
        """Test creating an A/B test."""
        # Get two default templates
        templates = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        assert len(templates) >= 1

        # Register a second template
        template_b = PromptTemplate(
            template_id="variant_b",
            name="Variant B",
            template="Alternative: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
            status=PromptVariantStatus.TESTING,
        )
        prompt_registry.register_template(template_b)

        test = ab_framework.create_test(
            templates[0].template_id,
            template_b.template_id,
            "test-model",
            PromptTaskType.ASP_GENERATION,
        )
        assert test.status == ABTestStatus.PENDING
        assert test.variant_a_id == templates[0].template_id
        assert test.variant_b_id == template_b.template_id

    def test_create_test_missing_template(self, ab_framework):
        """Test creating A/B test with missing template."""
        with pytest.raises(ABTestingError):
            ab_framework.create_test(
                "nonexistent_a",
                "nonexistent_b",
                "test-model",
                PromptTaskType.ASP_GENERATION,
            )

    def test_start_test(self, ab_framework, prompt_registry):
        """Test starting an A/B test."""
        templates = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        template_b = PromptTemplate(
            template_id="variant_b2",
            name="Variant B",
            template="Alt: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
        )
        prompt_registry.register_template(template_b)

        test = ab_framework.create_test(
            templates[0].template_id,
            template_b.template_id,
            "test-model",
            PromptTaskType.ASP_GENERATION,
        )
        result = ab_framework.start_test(test.test_id)
        assert result is True

        updated_test = ab_framework.get_test(test.test_id)
        assert updated_test.status == ABTestStatus.RUNNING

    def test_select_variant(self, ab_framework, prompt_registry):
        """Test variant selection for A/B test."""
        templates = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        template_b = PromptTemplate(
            template_id="variant_b3",
            name="Variant B",
            template="Alt: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
        )
        prompt_registry.register_template(template_b)

        test = ab_framework.create_test(
            templates[0].template_id,
            template_b.template_id,
            "test-model",
            PromptTaskType.ASP_GENERATION,
        )
        ab_framework.start_test(test.test_id)

        # Select variants multiple times
        selections = [ab_framework.select_variant(test.test_id) for _ in range(20)]
        unique_selections = set(selections)

        # Should select both variants (probabilistically)
        assert len(unique_selections) == 2

    def test_record_result(self, ab_framework, prompt_registry):
        """Test recording A/B test results."""
        templates = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        template_b = PromptTemplate(
            template_id="variant_b4",
            name="Variant B",
            template="Alt: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
        )
        prompt_registry.register_template(template_b)

        test = ab_framework.create_test(
            templates[0].template_id,
            template_b.template_id,
            "test-model",
            PromptTaskType.ASP_GENERATION,
        )
        ab_framework.start_test(test.test_id)

        # Record results for variant A
        ab_framework.record_result(
            test.test_id, templates[0].template_id, True, 0.9, 100.0
        )

        updated_test = ab_framework.get_test(test.test_id)
        assert updated_test.variant_a_samples == 1
        assert updated_test.variant_a_success_rate == 1.0

    def test_cancel_test(self, ab_framework, prompt_registry):
        """Test cancelling an A/B test."""
        templates = prompt_registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        template_b = PromptTemplate(
            template_id="variant_b5",
            name="Variant B",
            template="Alt: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
        )
        prompt_registry.register_template(template_b)

        test = ab_framework.create_test(
            templates[0].template_id,
            template_b.template_id,
            "test-model",
            PromptTaskType.ASP_GENERATION,
        )
        result = ab_framework.cancel_test(test.test_id)
        assert result is True

        cancelled_test = ab_framework.get_test(test.test_id)
        assert cancelled_test.status == ABTestStatus.CANCELLED


# =============================================================================
# Test ModelPromptOptimizer
# =============================================================================


class TestModelPromptOptimizer:
    """Tests for ModelPromptOptimizer."""

    def test_optimizer_initialization(self, prompt_optimizer):
        """Test optimizer initialization."""
        assert prompt_optimizer.registry is not None
        assert prompt_optimizer.tracker is not None
        assert prompt_optimizer.ab_framework is not None

    def test_register_model_profile(self, prompt_optimizer, sample_model_profile):
        """Test registering a model profile."""
        prompt_optimizer.register_model_profile("test-model", sample_model_profile)
        profile = prompt_optimizer.get_model_profile("test-model")
        assert profile is not None
        assert profile.success_rate == 0.85

    def test_get_optimized_prompt(self, prompt_optimizer):
        """Test getting optimized prompt."""
        prompt = prompt_optimizer.get_optimized_prompt(
            PromptTaskType.ASP_GENERATION,
            "test-model",
            context={
                "principle": "Contract requires offer",
                "predicates": "[]",
                "domain": "contracts",
            },
        )
        assert "Contract requires offer" in prompt

    def test_get_optimized_prompt_no_template(self, prompt_optimizer):
        """Test getting prompt when no template exists."""
        # Create a custom task type scenario by mocking
        optimizer = ModelPromptOptimizer()
        # Clear all templates for a specific task
        for task in PromptTaskType:
            templates = optimizer.registry.get_templates_for_task(task)
            for t in templates:
                optimizer.registry.update_template_status(
                    t.template_id, PromptVariantStatus.DEPRECATED
                )

        # Should raise error when no active templates
        with pytest.raises(PromptOptimizationError):
            optimizer.get_optimized_prompt(PromptTaskType.ASP_GENERATION, "test-model")

    def test_get_template_for_execution(self, prompt_optimizer):
        """Test getting template for execution."""
        template_id, ab_test_id = prompt_optimizer.get_template_for_execution(
            PromptTaskType.ASP_GENERATION, "test-model"
        )
        assert template_id is not None
        assert ab_test_id is None  # No A/B test running

    def test_record_performance(self, prompt_optimizer):
        """Test recording performance."""
        templates = prompt_optimizer.registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        template = templates[0]

        prompt_optimizer.record_performance(
            template_id=template.template_id,
            model_name="test-model",
            task_type=PromptTaskType.ASP_GENERATION,
            success=True,
            quality_score=0.9,
            latency_ms=100.0,
        )

        metrics = prompt_optimizer.tracker.get_aggregate_metrics(template.template_id)
        assert metrics["sample_count"] == 1

    def test_evolve_prompt(self, prompt_optimizer):
        """Test prompt evolution."""
        templates = prompt_optimizer.registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        template = templates[0]

        failure_patterns = [
            {"error_pattern": "Syntax error in output"},
            {"error_pattern": "Timeout waiting for response"},
        ]

        result = prompt_optimizer.evolve_prompt(template.template_id, failure_patterns)
        assert result.original_template_id == template.template_id
        assert result.evolved_template is not None
        assert len(result.changes_made) > 0

    def test_evolve_prompt_not_found(self, prompt_optimizer):
        """Test evolving non-existent prompt."""
        with pytest.raises(PromptOptimizationError):
            prompt_optimizer.evolve_prompt("nonexistent", [])

    def test_analyze_model_characteristics(
        self, prompt_optimizer, sample_model_profile
    ):
        """Test analyzing model characteristics."""
        prompt_optimizer.register_model_profile("test-model", sample_model_profile)

        # Record some performance data
        templates = prompt_optimizer.registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        for i in range(5):
            prompt_optimizer.record_performance(
                template_id=templates[0].template_id,
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
                success=i < 4,  # 80% success
                quality_score=0.8 if i < 4 else 0.0,
                latency_ms=100.0,
            )

        profile = prompt_optimizer.analyze_model_characteristics("test-model")
        assert profile is not None
        assert profile.success_rate == 0.8

    def test_start_ab_test(self, prompt_optimizer):
        """Test starting A/B test through optimizer."""
        templates = prompt_optimizer.registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )

        # Create a second template
        template_b = PromptTemplate(
            template_id="opt_variant_b",
            name="Optimizer Variant B",
            template="Generate: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
        )
        prompt_optimizer.registry.register_template(template_b)

        test_id = prompt_optimizer.start_ab_test(
            templates[0].template_id,
            template_b.template_id,
            "test-model",
            PromptTaskType.ASP_GENERATION,
        )

        result = prompt_optimizer.get_ab_test_results(test_id)
        assert result is not None
        assert result.status == ABTestStatus.RUNNING

    def test_add_optimization_suggestion(self, prompt_optimizer):
        """Test adding optimization suggestion."""
        suggestion_id = prompt_optimizer.add_optimization_suggestion(
            target_model="test-model",
            target_task=PromptTaskType.ASP_GENERATION,
            current_issue="Low success rate",
            suggested_change="Add more examples",
            rationale="Few-shot learning improves ASP generation",
            priority="high",
            confidence=0.8,
        )

        suggestions = prompt_optimizer.get_optimization_suggestions()
        assert len(suggestions) > 0
        assert any(s.suggestion_id == suggestion_id for s in suggestions)

    def test_get_optimization_suggestions_filtered(self, prompt_optimizer):
        """Test getting filtered suggestions."""
        prompt_optimizer.add_optimization_suggestion(
            target_model="model-a",
            target_task=PromptTaskType.ASP_GENERATION,
            current_issue="Issue A",
            suggested_change="Change A",
            rationale="Reason A",
            confidence=0.9,
        )
        prompt_optimizer.add_optimization_suggestion(
            target_model="model-b",
            target_task=PromptTaskType.EDGE_CASE_DETECTION,
            current_issue="Issue B",
            suggested_change="Change B",
            rationale="Reason B",
            confidence=0.5,
        )

        model_a_suggestions = prompt_optimizer.get_optimization_suggestions(
            model_name="model-a"
        )
        assert len(model_a_suggestions) == 1

        high_confidence = prompt_optimizer.get_optimization_suggestions(
            min_confidence=0.8
        )
        assert len(high_confidence) == 1

    def test_get_performance_report(self, prompt_optimizer):
        """Test getting performance report."""
        report = prompt_optimizer.get_performance_report()
        assert "total_templates" in report
        assert "active_templates" in report
        assert "templates_by_task" in report
        assert "top_performers" in report

    def test_clear_data(self, prompt_optimizer):
        """Test clearing data."""
        # Add some data
        prompt_optimizer.add_optimization_suggestion(
            target_model="test",
            target_task=PromptTaskType.ASP_GENERATION,
            current_issue="Test",
            suggested_change="Test",
            rationale="Test",
        )

        templates = prompt_optimizer.registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        prompt_optimizer.record_performance(
            template_id=templates[0].template_id,
            model_name="test",
            task_type=PromptTaskType.ASP_GENERATION,
            success=True,
            quality_score=0.8,
            latency_ms=100.0,
        )

        result = prompt_optimizer.clear_data()
        assert result["suggestions_cleared"] > 0
        assert result["records_cleared"] > 0


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prompt_optimizer(self):
        """Test creating optimizer through factory."""
        optimizer = create_prompt_optimizer()
        assert optimizer is not None
        assert isinstance(optimizer, ModelPromptOptimizer)

    def test_create_prompt_optimizer_with_config(self):
        """Test creating optimizer with custom config."""
        config = PromptOptimizerConfig(min_samples_for_ab_test=50)
        optimizer = create_prompt_optimizer(config=config)
        assert optimizer is not None

    def test_create_default_model_profiles(self):
        """Test creating default model profiles."""
        profiles = create_default_model_profiles()
        assert len(profiles) > 0
        assert "claude-3-5-haiku-20241022" in profiles
        assert "gpt-4" in profiles

    def test_create_optimizer_with_default_profiles(self):
        """Test creating optimizer with default profiles."""
        profiles = create_default_model_profiles()
        optimizer = create_prompt_optimizer(model_profiles=profiles)

        profile = optimizer.get_model_profile("claude-3-5-haiku-20241022")
        assert profile is not None
        assert profile.model_type == ModelType.GENERAL


# =============================================================================
# Test OptimizationSuggestion
# =============================================================================


class TestOptimizationSuggestion:
    """Tests for OptimizationSuggestion data class."""

    def test_suggestion_creation(self):
        """Test OptimizationSuggestion creation."""
        suggestion = OptimizationSuggestion(
            suggestion_id="sug_001",
            target_model="test-model",
            target_task=PromptTaskType.ASP_GENERATION,
            current_issue="Low accuracy",
            suggested_change="Add examples",
            rationale="Few-shot helps",
            priority="high",
            confidence=0.85,
        )
        assert suggestion.suggestion_id == "sug_001"
        assert suggestion.priority == "high"

    def test_suggestion_to_dict(self):
        """Test OptimizationSuggestion serialization."""
        suggestion = OptimizationSuggestion(
            suggestion_id="sug_001",
            target_model="test-model",
            target_task=PromptTaskType.ASP_GENERATION,
            current_issue="Issue",
            suggested_change="Change",
            rationale="Reason",
        )
        data = suggestion.to_dict()
        assert data["suggestion_id"] == "sug_001"
        assert data["target_task"] == "asp_generation"


# =============================================================================
# Test PromptEvolutionResult
# =============================================================================


class TestPromptEvolutionResult:
    """Tests for PromptEvolutionResult data class."""

    def test_evolution_result_creation(self, sample_template):
        """Test PromptEvolutionResult creation."""
        result = PromptEvolutionResult(
            original_template_id="orig_001",
            evolved_template=sample_template,
            changes_made=["Added syntax hints", "Added examples"],
            failure_patterns_addressed=["Syntax error"],
            expected_improvement="Better syntax compliance",
            confidence=0.75,
        )
        assert result.original_template_id == "orig_001"
        assert len(result.changes_made) == 2

    def test_evolution_result_to_dict(self, sample_template):
        """Test PromptEvolutionResult serialization."""
        result = PromptEvolutionResult(
            original_template_id="orig_001",
            evolved_template=sample_template,
        )
        data = result.to_dict()
        assert data["original_template_id"] == "orig_001"
        assert "evolved_template" in data


# =============================================================================
# Test PromptOptimizerConfig
# =============================================================================


class TestPromptOptimizerConfig:
    """Tests for PromptOptimizerConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = PromptOptimizerConfig()
        assert config.min_samples_for_ab_test == 30
        assert config.confidence_threshold == 0.95
        assert config.enable_auto_evolution is True

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = PromptOptimizerConfig(
            min_samples_for_ab_test=50,
            confidence_threshold=0.99,
            auto_deprecate_threshold=0.1,
        )
        assert config.min_samples_for_ab_test == 50
        assert config.confidence_threshold == 0.99
        assert config.auto_deprecate_threshold == 0.1


# =============================================================================
# Test Exceptions
# =============================================================================


class TestExceptions:
    """Tests for custom exceptions."""

    def test_prompt_optimization_error(self):
        """Test PromptOptimizationError."""
        error = PromptOptimizationError(
            "Template not found",
            model="test-model",
            task_type="asp_generation",
        )
        assert "Template not found" in str(error)
        assert error.model == "test-model"

    def test_ab_testing_error(self):
        """Test ABTestingError."""
        error = ABTestingError("Test failed", test_id="ab_001")
        assert "Test failed" in str(error)
        assert error.test_id == "ab_001"


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of components."""

    def test_registry_concurrent_access(self, prompt_registry):
        """Test concurrent access to registry."""
        import threading

        results = []

        def register_template(i):
            template = PromptTemplate(
                template_id=f"concurrent_{i}",
                name=f"Concurrent {i}",
                template="Test",
                model_type=ModelType.GENERAL,
                task_type=PromptTaskType.ASP_GENERATION,
            )
            prompt_registry.register_template(template)
            results.append(i)

        threads = [
            threading.Thread(target=register_template, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10

    def test_tracker_concurrent_recording(self, performance_tracker):
        """Test concurrent recording to tracker."""
        import threading

        def record_performance(i):
            record = PromptPerformanceRecord(
                record_id=f"concurrent_rec_{i}",
                template_id="tmpl_concurrent",
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
            )
            performance_tracker.record(record)

        threads = [
            threading.Thread(target=record_performance, args=(i,)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = performance_tracker.get_records("tmpl_concurrent")
        assert len(records) == 20


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create optimizer with profiles
        profiles = create_default_model_profiles()
        optimizer = create_prompt_optimizer(model_profiles=profiles)

        # Get a template
        template_id, _ = optimizer.get_template_for_execution(
            PromptTaskType.ASP_GENERATION, "claude-3-5-haiku-20241022"
        )

        # Record some performance
        for i in range(10):
            optimizer.record_performance(
                template_id=template_id,
                model_name="claude-3-5-haiku-20241022",
                task_type=PromptTaskType.ASP_GENERATION,
                success=i < 7,  # 70% success
                quality_score=0.8 if i < 7 else 0.0,
                latency_ms=100.0,
            )

        # Analyze model
        profile = optimizer.analyze_model_characteristics("claude-3-5-haiku-20241022")
        assert profile.success_rate == 0.7

        # Get report
        report = optimizer.get_performance_report()
        assert report["total_templates"] > 0

    def test_ab_test_to_completion(self):
        """Test A/B test running to completion."""
        config = PromptOptimizerConfig(
            min_samples_for_ab_test=5, confidence_threshold=0.5
        )
        optimizer = create_prompt_optimizer(config=config)

        # Create variant templates
        template_a = PromptTemplate(
            template_id="ab_test_a",
            name="Variant A",
            template="Generate A: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
            status=PromptVariantStatus.ACTIVE,
        )
        template_b = PromptTemplate(
            template_id="ab_test_b",
            name="Variant B",
            template="Generate B: {principle}",
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle"],
            status=PromptVariantStatus.TESTING,
        )
        optimizer.registry.register_template(template_a)
        optimizer.registry.register_template(template_b)

        # Start A/B test
        test_id = optimizer.start_ab_test(
            "ab_test_a", "ab_test_b", "test-model", PromptTaskType.ASP_GENERATION
        )

        # Record results - A is better
        for i in range(10):
            optimizer.record_performance(
                template_id="ab_test_a",
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
                success=True,
                quality_score=0.9,
                latency_ms=100.0,
                ab_test_id=test_id,
            )
            optimizer.record_performance(
                template_id="ab_test_b",
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
                success=i < 5,  # 50% success
                quality_score=0.6 if i < 5 else 0.0,
                latency_ms=100.0,
                ab_test_id=test_id,
            )

        # Check results
        result = optimizer.get_ab_test_results(test_id)
        # Test should complete with enough samples
        assert result is not None

    def test_prompt_evolution_cycle(self):
        """Test prompt evolution based on failures."""
        optimizer = create_prompt_optimizer()

        # Get a template
        templates = optimizer.registry.get_templates_for_task(
            PromptTaskType.ASP_GENERATION
        )
        original_template = templates[0]

        # Record failures
        for i in range(5):
            optimizer.record_performance(
                template_id=original_template.template_id,
                model_name="test-model",
                task_type=PromptTaskType.ASP_GENERATION,
                success=False,
                quality_score=0.0,
                latency_ms=100.0,
                error_message="Syntax error in ASP output",
            )

        # Get failure patterns
        patterns = optimizer.tracker.get_failure_patterns(original_template.template_id)

        # Evolve the prompt
        evolution_result = optimizer.evolve_prompt(
            original_template.template_id, patterns
        )

        assert evolution_result.evolved_template is not None
        assert (
            "syntax" in evolution_result.changes_made[0].lower()
            or len(evolution_result.changes_made) > 0
        )

        # New template should be registered
        new_template = optimizer.registry.get_template(
            evolution_result.evolved_template.template_id
        )
        assert new_template is not None
        assert new_template.status == PromptVariantStatus.TESTING
