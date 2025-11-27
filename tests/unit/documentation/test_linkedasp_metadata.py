"""
Unit tests for LinkedASP metadata export.

Tests RDF metadata generation, serialization, and SPARQL query generation.
Target coverage: 80%+ (from 69%)
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime
from loft.documentation.linkedasp_metadata import (
    RuleMetadata,
    ModuleMetadata,
    LinkedASPExporter,
)


class TestRuleMetadata:
    """Test RuleMetadata data structure."""

    def test_rule_metadata_creation(self):
        """Test creating rule metadata."""
        metadata = RuleMetadata(
            rule_id="rule_001",
            rule_text="enforceable(C) :- valid_contract(C).",
            predicate_name="enforceable",
            stratification_level="tactical",
        )

        assert metadata.rule_id == "rule_001"
        assert metadata.predicate_name == "enforceable"
        assert metadata.stratification_level == "tactical"

    def test_rule_metadata_with_optional_fields(self):
        """Test rule metadata with all optional fields."""
        gen_time = datetime.now()
        inc_time = datetime.now()

        metadata = RuleMetadata(
            rule_id="rule_002",
            rule_text="test :- condition.",
            predicate_name="test",
            stratification_level="operational",
            genre="DisjunctiveRequirement",
            legal_source="UCC 2-201",
            jurisdiction="California",
            confidence=0.95,
            source_type="llm_generated",
            source_llm="anthropic/claude-3-opus",
            generation_timestamp=gen_time,
            incorporation_timestamp=inc_time,
        )

        assert metadata.genre == "DisjunctiveRequirement"
        assert metadata.legal_source == "UCC 2-201"
        assert metadata.jurisdiction == "California"
        assert metadata.confidence == 0.95
        assert metadata.source_llm == "anthropic/claude-3-opus"

    def test_rule_metadata_to_dict(self):
        """Test converting rule metadata to dictionary."""
        metadata = RuleMetadata(
            rule_id="rule_003",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["rule_id"] == "rule_003"
        assert data["predicate_name"] == "test"

    def test_rule_metadata_to_dict_with_timestamps(self):
        """Test datetime conversion in to_dict."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)

        metadata = RuleMetadata(
            rule_id="rule_004",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
            generation_timestamp=timestamp,
        )

        data = metadata.to_dict()

        assert data["generation_timestamp"] == timestamp.isoformat()

    def test_rule_metadata_to_turtle_basic(self):
        """Test basic Turtle RDF generation."""
        metadata = RuleMetadata(
            rule_id="rule_005",
            rule_text="test.",
            predicate_name="test_pred",
            stratification_level="tactical",
        )

        turtle = metadata.to_turtle()

        assert "@prefix loft:" in turtle
        assert "asp:test_pred" in turtle
        assert "asp:ASPRule" in turtle
        assert '"tactical"' in turtle

    def test_rule_metadata_to_turtle_with_genre(self):
        """Test Turtle generation with genre."""
        metadata = RuleMetadata(
            rule_id="rule_006",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
            genre="Exception",
        )

        turtle = metadata.to_turtle()

        assert "loft:hasGenre legal:Exception" in turtle

    def test_rule_metadata_to_turtle_with_legal_source(self):
        """Test Turtle generation with legal source."""
        metadata = RuleMetadata(
            rule_id="rule_007",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
            legal_source="UCC 2-201",
            jurisdiction="Federal",
        )

        turtle = metadata.to_turtle()

        assert "loft:legalSource" in turtle
        assert "UCC 2-201" in turtle
        assert "loft:jurisdiction" in turtle
        assert "Federal" in turtle

    def test_rule_metadata_to_turtle_with_confidence(self):
        """Test Turtle generation includes confidence."""
        metadata = RuleMetadata(
            rule_id="rule_008",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
            confidence=0.85,
        )

        turtle = metadata.to_turtle()

        assert "loft:confidence 0.85" in turtle

    def test_rule_metadata_to_turtle_with_source_info(self):
        """Test Turtle generation with source information."""
        metadata = RuleMetadata(
            rule_id="rule_009",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
            source_type="llm_generated",
            source_llm="anthropic/claude-3-opus",
            incorporated_by="autonomous_system",
        )

        turtle = metadata.to_turtle()

        assert "loft:sourceType" in turtle
        assert "llm_generated" in turtle
        assert "loft:sourceLLM" in turtle
        assert "claude-3-opus" in turtle

    def test_rule_metadata_to_turtle_with_dependencies(self):
        """Test Turtle generation with dependencies."""
        metadata = RuleMetadata(
            rule_id="rule_010",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
            requires_elements=["predicate_a", "predicate_b"],
            has_alternatives=["alternative_rule"],
        )

        turtle = metadata.to_turtle()

        assert "loft:requiresElement asp:predicate_a" in turtle
        assert "loft:requiresElement asp:predicate_b" in turtle
        assert "loft:hasAlternative asp:alternative_rule" in turtle

    def test_rule_metadata_defaults(self):
        """Test rule metadata default values."""
        metadata = RuleMetadata(
            rule_id="rule_011",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )

        assert metadata.confidence == 1.0
        assert metadata.source_type == "manual"
        assert metadata.incorporated_by == "system"
        assert metadata.requires_elements == []
        assert metadata.has_alternatives == []


class TestModuleMetadata:
    """Test ModuleMetadata data structure."""

    def test_module_metadata_creation(self):
        """Test creating module metadata."""
        metadata = ModuleMetadata(
            module_name="contract_validity",
            domain="contract_law",
            phase="Phase 1",
            stratification_level="tactical",
            description="Rules for contract validity",
        )

        assert metadata.module_name == "contract_validity"
        assert metadata.domain == "contract_law"
        assert metadata.phase == "Phase 1"

    def test_module_metadata_with_rules(self):
        """Test module metadata with rules."""
        rule = RuleMetadata(
            rule_id="r1",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )

        metadata = ModuleMetadata(
            module_name="test_module",
            domain="test_domain",
            phase="Phase 1",
            stratification_level="tactical",
            description="Test",
            rules=[rule],
        )

        assert len(metadata.rules) == 1
        assert metadata.rules[0].rule_id == "r1"

    def test_module_metadata_to_dict(self):
        """Test converting module metadata to dict."""
        rule = RuleMetadata(
            rule_id="r1",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )

        metadata = ModuleMetadata(
            module_name="test_module",
            domain="test_domain",
            phase="Phase 1",
            stratification_level="tactical",
            description="Test",
            rules=[rule],
        )

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["module_name"] == "test_module"
        assert len(data["rules"]) == 1
        assert isinstance(data["rules"][0], dict)

    def test_module_metadata_with_exports_imports(self):
        """Test module with exports and imports."""
        metadata = ModuleMetadata(
            module_name="test_module",
            domain="test_domain",
            phase="Phase 1",
            stratification_level="tactical",
            description="Test",
            exports=["predicate_a", "predicate_b"],
            imports=["imported_pred"],
        )

        assert metadata.exports == ["predicate_a", "predicate_b"]
        assert metadata.imports == ["imported_pred"]


class TestLinkedASPExporter:
    """Test LinkedASP exporter."""

    def test_exporter_initialization(self):
        """Test exporter initialization."""
        exporter = LinkedASPExporter()

        assert isinstance(exporter.rules, list)
        assert isinstance(exporter.modules, list)
        assert len(exporter.rules) == 0
        assert len(exporter.modules) == 0

    def test_add_rule(self):
        """Test adding rule to exporter."""
        exporter = LinkedASPExporter()

        rule = RuleMetadata(
            rule_id="r1",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )

        exporter.add_rule(rule)

        assert len(exporter.rules) == 1
        assert exporter.rules[0] == rule

    def test_add_multiple_rules(self):
        """Test adding multiple rules."""
        exporter = LinkedASPExporter()

        for i in range(3):
            rule = RuleMetadata(
                rule_id=f"r{i}",
                rule_text=f"test{i}.",
                predicate_name=f"test{i}",
                stratification_level="tactical",
            )
            exporter.add_rule(rule)

        assert len(exporter.rules) == 3

    def test_add_module(self):
        """Test adding module to exporter."""
        exporter = LinkedASPExporter()

        module = ModuleMetadata(
            module_name="test_module",
            domain="test",
            phase="Phase 1",
            stratification_level="tactical",
            description="Test module",
        )

        exporter.add_module(module)

        assert len(exporter.modules) == 1
        assert exporter.modules[0] == module

    def test_export_json_ld(self):
        """Test JSON-LD export."""
        exporter = LinkedASPExporter()

        rule = RuleMetadata(
            rule_id="r1",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )
        exporter.add_rule(rule)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonld") as f:
            output_path = f.name

        try:
            content = exporter.export_json_ld(output_path)

            # Verify content is valid JSON
            data = json.loads(content)

            assert "@context" in data
            assert "rules" in data
            assert len(data["rules"]) == 1
            assert "generated_at" in data
            assert "generator" in data

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_json_ld_with_modules(self):
        """Test JSON-LD export with modules."""
        exporter = LinkedASPExporter()

        module = ModuleMetadata(
            module_name="test_module",
            domain="test",
            phase="Phase 1",
            stratification_level="tactical",
            description="Test",
        )
        exporter.add_module(module)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonld") as f:
            output_path = f.name

        try:
            content = exporter.export_json_ld(output_path)
            data = json.loads(content)

            assert "modules" in data
            assert len(data["modules"]) == 1

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_turtle(self):
        """Test Turtle RDF export."""
        exporter = LinkedASPExporter()

        rule = RuleMetadata(
            rule_id="r1",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )
        exporter.add_rule(rule)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ttl") as f:
            output_path = f.name

        try:
            content = exporter.export_turtle(output_path)

            assert "@prefix loft:" in content
            assert "@prefix asp:" in content
            assert "asp:test a asp:ASPRule" in content

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_turtle_with_modules(self):
        """Test Turtle export with modules."""
        exporter = LinkedASPExporter()

        module = ModuleMetadata(
            module_name="test_module",
            domain="contract_law",
            phase="Phase 1",
            stratification_level="tactical",
            description="Test module",
        )
        exporter.add_module(module)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ttl") as f:
            output_path = f.name

        try:
            content = exporter.export_turtle(output_path)

            assert "loft:ASPModule" in content
            assert "test_module" in content
            assert "contract_law" in content

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_turtle_includes_header(self):
        """Test Turtle export includes header comments."""
        exporter = LinkedASPExporter()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ttl") as f:
            output_path = f.name

        try:
            content = exporter.export_turtle(output_path)

            assert "# LinkedASP Metadata Export" in content
            assert "# Generated:" in content
            assert "# Compatible with docs/MAINTAINABILITY.md" in content

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_query_examples(self):
        """Test SPARQL query examples generation."""
        exporter = LinkedASPExporter()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sparql") as f:
            output_path = f.name

        try:
            content = exporter.generate_query_examples(output_path)

            # Check for example queries
            assert "PREFIX loft:" in content
            assert "PREFIX legal:" in content
            assert "SELECT" in content
            assert "WHERE" in content

            # Check for specific query examples
            assert "Find all rules of a specific genre" in content
            assert "Find rules with low confidence" in content
            assert "Analyze impact of modifying a predicate" in content

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_creates_files(self):
        """Test export actually creates files."""
        exporter = LinkedASPExporter()

        rule = RuleMetadata(
            rule_id="r1",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
        )
        exporter.add_rule(rule)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "export.jsonld"
            turtle_path = Path(tmpdir) / "export.ttl"

            exporter.export_json_ld(str(json_path))
            exporter.export_turtle(str(turtle_path))

            assert json_path.exists()
            assert turtle_path.exists()
            assert json_path.stat().st_size > 0
            assert turtle_path.stat().st_size > 0


class TestLinkedASPIntegration:
    """Integration tests for LinkedASP metadata."""

    def test_complete_workflow(self):
        """Test complete metadata export workflow."""
        # Create exporter
        exporter = LinkedASPExporter()

        # Create rules
        rule1 = RuleMetadata(
            rule_id="rule_001",
            rule_text="enforceable(C) :- valid_contract(C).",
            predicate_name="enforceable",
            stratification_level="tactical",
            genre="Requirement",
            legal_source="UCC 2-201",
            jurisdiction="Federal",
            confidence=0.95,
            source_type="manual",
        )

        rule2 = RuleMetadata(
            rule_id="rule_002",
            rule_text="valid_contract(C) :- offer(C), acceptance(C).",
            predicate_name="valid_contract",
            stratification_level="tactical",
            requires_elements=["offer", "acceptance"],
        )

        # Create module
        module = ModuleMetadata(
            module_name="contract_validity",
            domain="contract_law",
            phase="Phase 1",
            stratification_level="tactical",
            description="Contract validity rules",
            exports=["enforceable", "valid_contract"],
            rules=[rule1, rule2],
        )

        # Add to exporter
        exporter.add_rule(rule1)
        exporter.add_rule(rule2)
        exporter.add_module(module)

        # Export to all formats
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metadata.jsonld"
            turtle_path = Path(tmpdir) / "metadata.ttl"
            query_path = Path(tmpdir) / "queries.sparql"

            exporter.export_json_ld(str(json_path))
            exporter.export_turtle(str(turtle_path))
            exporter.generate_query_examples(str(query_path))

            # Verify all files created
            assert json_path.exists()
            assert turtle_path.exists()
            assert query_path.exists()

            # Verify content
            json_content = json_path.read_text()
            turtle_content = turtle_path.read_text()
            query_content = query_path.read_text()

            # Check JSON-LD
            data = json.loads(json_content)
            assert len(data["rules"]) == 2
            assert len(data["modules"]) == 1

            # Check Turtle
            assert "enforceable" in turtle_content
            assert "valid_contract" in turtle_content

            # Check queries
            assert "SELECT" in query_content

    def test_roundtrip_rule_metadata(self):
        """Test metadata can be serialized and deserialized."""
        original = RuleMetadata(
            rule_id="r1",
            rule_text="test.",
            predicate_name="test",
            stratification_level="tactical",
            confidence=0.88,
        )

        # Convert to dict and back
        data = original.to_dict()
        # Would need from_dict method for true roundtrip, but we test dict structure
        assert data["rule_id"] == original.rule_id
        assert data["confidence"] == original.confidence

    def test_empty_exporter_export(self):
        """Test exporting with no rules or modules."""
        exporter = LinkedASPExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "empty.jsonld"
            turtle_path = Path(tmpdir) / "empty.ttl"

            exporter.export_json_ld(str(json_path))
            exporter.export_turtle(str(turtle_path))

            # Files should still be created
            assert json_path.exists()
            assert turtle_path.exists()

            # JSON should be valid
            data = json.loads(json_path.read_text())
            assert data["rules"] == []
            assert data["modules"] == []
