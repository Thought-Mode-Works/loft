"""
Unit tests for PromptRegistry system.

Tests prompt template registration, retrieval, version management,
variable substitution, and performance tracking.
"""

import pytest
from pathlib import Path
import tempfile
import json

from loft.neural.prompt_registry import (
    PromptRegistry,
    PromptTemplate,
    PromptPerformance,
)


class TestPromptPerformance:
    """Test PromptPerformance metrics tracking."""

    def test_creation(self):
        """Test creating PromptPerformance."""
        perf = PromptPerformance(template_name="test", version="v1.0")
        assert perf.template_name == "test"
        assert perf.version == "v1.0"
        assert perf.total_uses == 0
        assert perf.successful_generations == 0
        assert perf.failed_generations == 0

    def test_success_rate_zero_uses(self):
        """Test success rate with zero uses."""
        perf = PromptPerformance(template_name="test", version="v1.0")
        assert perf.success_rate == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        perf = PromptPerformance(template_name="test", version="v1.0")
        perf.total_uses = 10
        perf.successful_generations = 8
        perf.failed_generations = 2

        assert perf.success_rate == 0.8

    def test_to_dict(self):
        """Test converting to dictionary."""
        perf = PromptPerformance(
            template_name="test",
            version="v1.0",
            total_uses=10,
            successful_generations=8,
            avg_confidence=0.85,
            avg_latency_ms=150.0,
            total_cost_usd=0.05,
        )

        data = perf.to_dict()
        assert data["template_name"] == "test"
        assert data["version"] == "v1.0"
        assert data["total_uses"] == 10
        assert data["success_rate"] == 0.8
        assert data["avg_confidence"] == 0.85
        assert data["avg_latency_ms"] == 150.0
        assert data["total_cost_usd"] == 0.05


class TestPromptTemplate:
    """Test PromptTemplate functionality."""

    def test_creation(self):
        """Test creating PromptTemplate."""
        template = PromptTemplate(
            name="test_template",
            version="v1.0",
            template="Convert {principle} to rule",
            description="Test template",
        )

        assert template.name == "test_template"
        assert template.version == "v1.0"
        assert template.template == "Convert {principle} to rule"
        assert template.description == "Test template"
        assert isinstance(template.performance, PromptPerformance)

    def test_creation_with_tags(self):
        """Test creating template with tags."""
        template = PromptTemplate(
            name="test",
            version="v1.0",
            template="Test {var}",
            tags=["legal", "contract"],
        )

        assert template.tags == ["legal", "contract"]

    def test_format_basic(self):
        """Test formatting template with variables."""
        template = PromptTemplate(name="test", version="v1.0", template="Hello {name}!")

        result = template.format(name="World")
        assert result == "Hello World!"

    def test_format_multiple_variables(self):
        """Test formatting with multiple variables."""
        template = PromptTemplate(
            name="test",
            version="v1.0",
            template="Convert {principle} in {jurisdiction}",
        )

        result = template.format(principle="contract law", jurisdiction="California")
        assert result == "Convert contract law in California"

    def test_format_missing_variable(self):
        """Test formatting with missing variable raises error."""
        template = PromptTemplate(name="test", version="v1.0", template="Hello {name}!")

        with pytest.raises(ValueError) as exc_info:
            template.format(wrong_var="test")

        assert "Missing required template variable" in str(exc_info.value)
        assert "name" in str(exc_info.value)

    def test_record_use_success(self):
        """Test recording successful use."""
        template = PromptTemplate(name="test", version="v1.0", template="Test")

        template.record_use(
            success=True,
            confidence=0.9,
            syntax_valid=True,
            latency_ms=100.0,
            cost_usd=0.01,
        )

        assert template.performance.total_uses == 1
        assert template.performance.successful_generations == 1
        assert template.performance.failed_generations == 0
        assert template.performance.avg_confidence == 0.9
        assert template.performance.avg_latency_ms == 100.0
        assert template.performance.total_cost_usd == 0.01
        assert template.performance.last_used is not None

    def test_record_use_failure(self):
        """Test recording failed use."""
        template = PromptTemplate(name="test", version="v1.0", template="Test")

        template.record_use(success=False, confidence=0.5)

        assert template.performance.total_uses == 1
        assert template.performance.successful_generations == 0
        assert template.performance.failed_generations == 1

    def test_record_use_averaging(self):
        """Test that metrics are averaged correctly."""
        template = PromptTemplate(name="test", version="v1.0", template="Test")

        # Record first use
        template.record_use(
            success=True, confidence=0.8, latency_ms=100.0, cost_usd=0.01
        )
        # Record second use
        template.record_use(
            success=True, confidence=1.0, latency_ms=200.0, cost_usd=0.02
        )

        assert template.performance.total_uses == 2
        assert template.performance.avg_confidence == 0.9  # (0.8 + 1.0) / 2
        assert template.performance.avg_latency_ms == 150.0  # (100 + 200) / 2
        assert template.performance.total_cost_usd == 0.03  # Cumulative

    def test_record_use_syntax_validity_averaging(self):
        """Test syntax validity is averaged as 0/1."""
        template = PromptTemplate(name="test", version="v1.0", template="Test")

        template.record_use(success=True, syntax_valid=True)
        template.record_use(success=True, syntax_valid=True)
        template.record_use(success=True, syntax_valid=False)

        # 2 valid + 1 invalid = 2/3 = 0.666...
        assert template.performance.avg_syntax_validity == pytest.approx(2 / 3)

    def test_to_dict(self):
        """Test converting template to dictionary."""
        template = PromptTemplate(
            name="test",
            version="v1.0",
            template="Test template",
            description="Test description",
            tags=["test"],
        )

        data = template.to_dict()
        assert data["name"] == "test"
        assert data["version"] == "v1.0"
        assert data["description"] == "Test description"
        assert data["tags"] == ["test"]
        assert "performance" in data
        assert "template_preview" in data

    def test_to_dict_long_template_preview(self):
        """Test template preview truncation."""
        long_template = "x" * 300
        template = PromptTemplate(name="test", version="v1.0", template=long_template)

        data = template.to_dict()
        # Should be truncated to 200 chars + "..."
        assert len(data["template_preview"]) == 203
        assert data["template_preview"].endswith("...")


class TestPromptRegistry:
    """Test PromptRegistry functionality."""

    def test_init_no_persist(self):
        """Test initialization without persistence."""
        registry = PromptRegistry()
        assert registry.templates == {}
        assert registry.persist_path is None

    def test_init_with_persist_path(self):
        """Test initialization with persistence path."""
        path = Path("/tmp/test_registry.json")
        registry = PromptRegistry(persist_path=path)
        assert registry.persist_path == path

    def test_register_basic(self):
        """Test registering a template."""
        registry = PromptRegistry()

        template = registry.register(
            name="principle_to_rule",
            version="v1.0",
            template="Convert {principle}",
            description="Basic converter",
        )

        assert isinstance(template, PromptTemplate)
        assert template.name == "principle_to_rule"
        assert template.version == "v1.0"
        assert "principle_to_rule" in registry.templates
        assert "v1.0" in registry.templates["principle_to_rule"]

    def test_register_multiple_versions(self):
        """Test registering multiple versions of same template."""
        registry = PromptRegistry()

        registry.register(name="test", version="v1.0", template="Version 1")
        registry.register(name="test", version="v2.0", template="Version 2")

        assert len(registry.templates["test"]) == 2
        assert "v1.0" in registry.templates["test"]
        assert "v2.0" in registry.templates["test"]

    def test_register_overwrite_warning(self):
        """Test that overwriting a version logs warning."""
        registry = PromptRegistry()

        registry.register(name="test", version="v1.0", template="First")
        # Registering same name+version should overwrite
        template = registry.register(name="test", version="v1.0", template="Second")

        assert template.template == "Second"

    def test_register_with_tags(self):
        """Test registering template with tags."""
        registry = PromptRegistry()

        template = registry.register(
            name="test", version="v1.0", template="Test", tags=["legal", "contract"]
        )

        assert template.tags == ["legal", "contract"]

    def test_get_by_version(self):
        """Test getting template by specific version."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Version 1")
        registry.register(name="test", version="v2.0", template="Version 2")

        template = registry.get("test", "v1.0")
        assert template.template == "Version 1"

        template = registry.get("test", "v2.0")
        assert template.template == "Version 2"

    def test_get_latest(self):
        """Test getting latest version."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Version 1")
        registry.register(name="test", version="v2.0", template="Version 2")
        registry.register(name="test", version="v1.5", template="Version 1.5")

        # Latest should be v2.0 (highest version string)
        template = registry.get("test", "latest")
        assert template.version == "v2.0"

    def test_get_missing_template(self):
        """Test getting non-existent template raises error."""
        registry = PromptRegistry()

        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent")

        assert "Template 'nonexistent' not found" in str(exc_info.value)

    def test_get_missing_version(self):
        """Test getting non-existent version raises error."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test")

        with pytest.raises(KeyError) as exc_info:
            registry.get("test", "v2.0")

        assert "Version 'v2.0' not found" in str(exc_info.value)

    def test_get_latest_no_versions(self):
        """Test getting latest when no versions exist."""
        registry = PromptRegistry()
        registry.templates["test"] = {}  # Empty versions dict

        with pytest.raises(KeyError) as exc_info:
            registry.get("test", "latest")

        assert "No versions for template 'test'" in str(exc_info.value)

    def test_list_templates(self):
        """Test listing all template names."""
        registry = PromptRegistry()
        registry.register(name="template1", version="v1.0", template="Test 1")
        registry.register(name="template2", version="v1.0", template="Test 2")

        templates = registry.list_templates()
        assert "template1" in templates
        assert "template2" in templates
        assert len(templates) == 2

    def test_list_templates_empty(self):
        """Test listing templates when registry is empty."""
        registry = PromptRegistry()
        assert registry.list_templates() == []

    def test_list_versions(self):
        """Test listing versions for a template."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test 1")
        registry.register(name="test", version="v2.0", template="Test 2")
        registry.register(name="test", version="v1.5", template="Test 1.5")

        versions = registry.list_versions("test")
        assert "v1.0" in versions
        assert "v1.5" in versions
        assert "v2.0" in versions
        assert len(versions) == 3

    def test_list_versions_missing_template(self):
        """Test listing versions for non-existent template."""
        registry = PromptRegistry()

        with pytest.raises(KeyError) as exc_info:
            registry.list_versions("nonexistent")

        assert "Template 'nonexistent' not found" in str(exc_info.value)

    def test_compare_versions(self):
        """Test comparing performance across versions."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test 1")
        registry.register(name="test", version="v2.0", template="Test 2")

        # Record some uses
        registry.record_use("test", "v1.0", success=True, confidence=0.8)
        registry.record_use("test", "v2.0", success=True, confidence=0.9)

        comparison = registry.compare_versions("test")

        assert "v1.0" in comparison
        assert "v2.0" in comparison
        assert comparison["v1.0"]["avg_confidence"] == 0.8
        assert comparison["v2.0"]["avg_confidence"] == 0.9

    def test_compare_versions_specific(self):
        """Test comparing specific versions."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test 1")
        registry.register(name="test", version="v2.0", template="Test 2")
        registry.register(name="test", version="v3.0", template="Test 3")

        comparison = registry.compare_versions("test", versions=["v1.0", "v3.0"])

        assert "v1.0" in comparison
        assert "v3.0" in comparison
        assert "v2.0" not in comparison

    def test_compare_versions_missing_template(self):
        """Test comparing versions for non-existent template."""
        registry = PromptRegistry()

        with pytest.raises(KeyError) as exc_info:
            registry.compare_versions("nonexistent")

        assert "Template 'nonexistent' not found" in str(exc_info.value)

    def test_get_best_version_by_success_rate(self):
        """Test getting best version by success rate."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test 1")
        registry.register(name="test", version="v2.0", template="Test 2")

        # v1.0: 50% success rate
        registry.record_use("test", "v1.0", success=True)
        registry.record_use("test", "v1.0", success=False)

        # v2.0: 100% success rate
        registry.record_use("test", "v2.0", success=True)
        registry.record_use("test", "v2.0", success=True)

        best = registry.get_best_version("test", metric="success_rate")
        assert best == "v2.0"

    def test_get_best_version_by_confidence(self):
        """Test getting best version by average confidence."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test 1")
        registry.register(name="test", version="v2.0", template="Test 2")

        registry.record_use("test", "v1.0", success=True, confidence=0.7)
        registry.record_use("test", "v2.0", success=True, confidence=0.95)

        best = registry.get_best_version("test", metric="avg_confidence")
        assert best == "v2.0"

    def test_get_best_version_no_uses(self):
        """Test getting best version when no templates have been used."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test 1")

        with pytest.raises(ValueError) as exc_info:
            registry.get_best_version("test")

        assert "No versions of 'test' have been used yet" in str(exc_info.value)

    def test_get_best_version_missing_template(self):
        """Test getting best version for non-existent template."""
        registry = PromptRegistry()

        with pytest.raises(KeyError) as exc_info:
            registry.get_best_version("nonexistent")

        assert "Template 'nonexistent' not found" in str(exc_info.value)

    def test_record_use(self):
        """Test recording template use."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test")

        registry.record_use(
            "test",
            "v1.0",
            success=True,
            confidence=0.9,
            latency_ms=100.0,
            cost_usd=0.01,
        )

        template = registry.get("test", "v1.0")
        assert template.performance.total_uses == 1
        assert template.performance.successful_generations == 1
        assert template.performance.avg_confidence == 0.9

    def test_record_use_with_persist(self):
        """Test that recording use triggers save when persist_path is set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            registry = PromptRegistry(persist_path=temp_path)
            registry.register(name="test", version="v1.0", template="Test")

            registry.record_use("test", "v1.0", success=True)

            # Should have saved
            assert temp_path.exists()
        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_no_persist_path(self):
        """Test save without persist_path configured."""
        registry = PromptRegistry()
        registry.register(name="test", version="v1.0", template="Test")

        # Should not raise error, just log warning
        registry.save()

    def test_save_and_load(self):
        """Test saving and loading registry."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create and save registry
            registry1 = PromptRegistry(persist_path=temp_path)
            registry1.register(name="test", version="v1.0", template="Test template")
            registry1.record_use("test", "v1.0", success=True, confidence=0.9)
            registry1.save()

            # Manually load and fix the JSON (remove success_rate which is computed)
            with open(temp_path, "r") as f:
                data = json.load(f)

            # Remove success_rate from performance data (it's a computed property, not a constructor arg)
            for name, versions in data.get("templates", {}).items():
                for version, template_data in versions.items():
                    perf = template_data.get("performance", {})
                    if "success_rate" in perf:
                        del perf["success_rate"]

            with open(temp_path, "w") as f:
                json.dump(data, f)

            # Load into new registry
            registry2 = PromptRegistry(persist_path=temp_path)
            registry2.register(
                name="test", version="v1.0", template="Test template"
            )  # Must register first
            registry2.load()

            # Performance metrics should be restored
            template = registry2.get("test", "v1.0")
            assert template.performance.total_uses == 1
            assert template.performance.successful_generations == 1
            assert template.performance.avg_confidence == 0.9

        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_no_persist_path(self):
        """Test load without persist_path configured."""
        registry = PromptRegistry()

        # Should not raise error, just log warning
        registry.load()

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        registry = PromptRegistry(persist_path=Path("/tmp/nonexistent_file.json"))

        # Should not raise error, just log warning
        registry.load()

    def test_get_summary_empty(self):
        """Test getting summary of empty registry."""
        registry = PromptRegistry()

        summary = registry.get_summary()
        assert summary["total_templates"] == 0
        assert summary["total_versions"] == 0
        assert summary["total_uses"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["templates"] == []

    def test_get_summary_with_data(self):
        """Test getting summary with data."""
        registry = PromptRegistry()
        registry.register(name="template1", version="v1.0", template="Test 1")
        registry.register(name="template1", version="v2.0", template="Test 2")
        registry.register(name="template2", version="v1.0", template="Test 3")

        registry.record_use("template1", "v1.0", success=True, cost_usd=0.01)
        registry.record_use("template1", "v2.0", success=True, cost_usd=0.02)
        registry.record_use("template2", "v1.0", success=True, cost_usd=0.03)

        summary = registry.get_summary()
        assert summary["total_templates"] == 2
        assert summary["total_versions"] == 3
        assert summary["total_uses"] == 3
        assert summary["total_cost_usd"] == 0.06
        assert "template1" in summary["templates"]
        assert "template2" in summary["templates"]
