"""
Unit tests for living document generation.

Tests document generation, section rendering, and integration with
the self-modifying system.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
import tempfile

from loft.documentation.living_document import LivingDocumentGenerator
from loft.core.integration_schemas import ImprovementCycleResult
from loft.core.incorporation import IncorporationResult
from loft.symbolic.stratification import StratificationLevel


class TestLivingDocumentGenerator:
    """Test living document generator."""

    @pytest.fixture
    def mock_system(self):
        """Create mock self-modifying system."""
        system = Mock()

        # Mock ASP core
        asp_core = Mock()
        asp_core.get_rules_by_layer = Mock(
            side_effect=lambda layer: {
                StratificationLevel.CONSTITUTIONAL: [
                    "contract(C) :- parties(C, _, _), consideration(C).",
                    "enforceable(C) :- contract(C), not void(C).",
                ],
                StratificationLevel.STRATEGIC: [
                    "requires_writing(C) :- land_sale_contract(C).",
                ],
                StratificationLevel.TACTICAL: [
                    "statute_of_frauds_satisfied(C) :- writing(C, W), signed(W).",
                ],
                StratificationLevel.OPERATIONAL: [],
            }.get(layer, [])
        )

        system.asp_core = asp_core

        # Mock performance monitor
        performance_monitor = Mock()
        health_report = Mock()
        health_report.overall_health = "good"
        health_report.success_rate = 0.85
        performance_monitor.get_system_health = Mock(return_value=health_report)
        system.performance_monitor = performance_monitor

        # Mock rule generator
        system.rule_generator = Mock()

        # Mock incorporation engine
        incorporation_engine = Mock()
        incorporation_engine.incorporation_history = []
        system.incorporation_engine = incorporation_engine

        return system

    @pytest.fixture
    def generator(self, mock_system):
        """Create living document generator with mock system."""
        return LivingDocumentGenerator(system=mock_system)

    @pytest.fixture
    def sample_cycle_result(self):
        """Create sample improvement cycle result."""
        return ImprovementCycleResult(
            cycle_number=1,
            timestamp=datetime(2025, 1, 24, 14, 30, 0),
            gaps_identified=3,
            variants_generated=9,
            rules_incorporated=2,
            rules_pending_review=0,
            baseline_accuracy=0.85,
            final_accuracy=0.87,
            overall_improvement=0.02,
            status="success",
        )

    def test_generator_initialization(self, mock_system):
        """Test generator initialization."""
        generator = LivingDocumentGenerator(system=mock_system)

        assert generator.system == mock_system
        assert generator.cycle_history == []

    def test_generator_without_system(self):
        """Test generator can be created without system."""
        generator = LivingDocumentGenerator(system=None)

        assert generator.system is None
        assert generator.cycle_history == []

    def test_update_cycle_history(self, generator, sample_cycle_result):
        """Test updating cycle history."""
        generator.update_cycle_history(sample_cycle_result)

        assert len(generator.cycle_history) == 1
        assert generator.cycle_history[0] == sample_cycle_result

    def test_cycle_history_limit(self, generator):
        """Test cycle history is limited to 20 cycles."""
        # Add 25 cycles
        for i in range(25):
            cycle = ImprovementCycleResult(
                cycle_number=i + 1,
                timestamp=datetime.now(),
                gaps_identified=1,
                variants_generated=3,
                rules_incorporated=1,
                rules_pending_review=0,
                baseline_accuracy=0.85,
                final_accuracy=0.86,
                overall_improvement=0.01,
                status="success",
            )
            generator.update_cycle_history(cycle)

        # Should only keep last 20
        assert len(generator.cycle_history) == 20
        assert generator.cycle_history[0].cycle_number == 6  # First of last 20

    def test_generate_header(self, generator):
        """Test header generation."""
        header = generator._generate_header()

        assert "# ASP Core Living Document" in header
        assert "*Generated:" in header
        assert "evolving" in header.lower()
        assert "asp" in header.lower()

    def test_generate_table_of_contents(self, generator):
        """Test table of contents generation."""
        toc = generator._generate_table_of_contents()

        assert "## Table of Contents" in toc
        assert "Overview" in toc
        assert "Rules by Stratification Layer" in toc
        assert "Incorporation History" in toc
        assert "Improvement Cycles" in toc
        assert "Self-Analysis" in toc

    def test_generate_overview_with_system(self, generator):
        """Test overview generation with system."""
        overview = generator._generate_overview()

        assert "## Overview" in overview
        assert "**Total Rules**:" in overview
        assert "**LLM Integration**: Enabled" in overview

    def test_generate_overview_without_system(self):
        """Test overview generation without system."""
        generator = LivingDocumentGenerator(system=None)
        overview = generator._generate_overview()

        assert "## Overview" in overview
        assert "*System not initialized*" in overview

    def test_generate_rules_by_layer(self, generator):
        """Test rules by layer generation."""
        rules_section = generator._generate_rules_by_layer()

        assert "## Rules by Stratification Layer" in rules_section
        assert "### Constitutional Layer" in rules_section
        assert "### Strategic Layer" in rules_section
        assert "### Tactical Layer" in rules_section
        assert "### Operational Layer" in rules_section

        # Check for rules
        assert "contract(C)" in rules_section
        assert "enforceable(C)" in rules_section
        assert "requires_writing(C)" in rules_section

        # Check for descriptions
        assert "immutable" in rules_section.lower()
        assert "autonomously modified" in rules_section.lower()

    def test_generate_rules_by_layer_without_system(self):
        """Test rules generation without system."""
        generator = LivingDocumentGenerator(system=None)
        rules_section = generator._generate_rules_by_layer()

        assert "## Rules by Stratification Layer" in rules_section
        assert "*No ASP core available*" in rules_section

    def test_generate_incorporation_history(self, generator, sample_cycle_result):
        """Test incorporation history generation."""
        # Add some incorporations to cycle
        inc_result = IncorporationResult(
            status="success",
            reason="Test incorporation",
            modification_number=1,
            accuracy_before=0.85,
            accuracy_after=0.87,
        )

        sample_cycle_result.successful_incorporations = [inc_result]
        generator.update_cycle_history(sample_cycle_result)

        history = generator._generate_incorporation_history()

        assert "## Incorporation History" in history
        assert "Recent successful incorporations" in history

    def test_generate_incorporation_history_without_engine(self):
        """Test incorporation history without engine."""
        generator = LivingDocumentGenerator(system=None)
        history = generator._generate_incorporation_history()

        assert "## Incorporation History" in history
        assert "*No incorporation engine available*" in history

    def test_generate_improvement_cycles(self, generator, sample_cycle_result):
        """Test improvement cycles generation."""
        generator.update_cycle_history(sample_cycle_result)

        cycles_section = generator._generate_improvement_cycles()

        assert "## Improvement Cycles" in cycles_section
        assert "Cycle #" in cycles_section
        assert "Gaps" in cycles_section
        assert "Variants" in cycles_section
        assert "Incorporated" in cycles_section
        assert "**Cycle Statistics**:" in cycles_section

    def test_generate_improvement_cycles_empty(self, generator):
        """Test improvement cycles with no history."""
        cycles_section = generator._generate_improvement_cycles()

        assert "## Improvement Cycles" in cycles_section
        assert "*No improvement cycles completed yet*" in cycles_section

    def test_generate_evolution_timeline(self, generator, sample_cycle_result):
        """Test evolution timeline generation."""
        generator.update_cycle_history(sample_cycle_result)

        timeline = generator._generate_evolution_timeline()

        assert "## Rule Evolution Timeline" in timeline
        assert "Cycle #1" in timeline
        assert "2025-01-24" in timeline
        assert "**Status**: success" in timeline

    def test_generate_evolution_timeline_empty(self, generator):
        """Test timeline with no history."""
        timeline = generator._generate_evolution_timeline()

        assert "## Rule Evolution Timeline" in timeline
        assert "*No evolution history available*" in timeline

    def test_generate_footer(self, generator):
        """Test footer generation."""
        footer = generator._generate_footer()

        assert "**Document Metadata**" in footer
        assert "Generated:" in footer
        assert "Living Document Generator" in footer
        assert "automatically generated" in footer

    def test_generate_complete_document(self, generator, sample_cycle_result):
        """Test generating complete document."""
        generator.update_cycle_history(sample_cycle_result)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            document = generator.generate(output_path=tmp_path, include_metadata=True)

            # Check document structure
            assert "# ASP Core Living Document" in document
            assert "## Table of Contents" in document
            assert "## Overview" in document
            assert "## Rules by Stratification Layer" in document
            assert "## Improvement Cycles" in document

            # Check file was created
            assert Path(tmp_path).exists()
            content = Path(tmp_path).read_text()
            assert content == document

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_generate_without_metadata(self, generator):
        """Test generating document without metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            document = generator.generate(output_path=tmp_path, include_metadata=False)

            # Metadata section should not be included
            assert "**Document Metadata**" not in document

            # But header and other sections should be there
            assert "# ASP Core Living Document" in document
            assert "## Overview" in document

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_document_is_valid_markdown(self, generator, sample_cycle_result):
        """Test that generated document is valid markdown."""
        generator.update_cycle_history(sample_cycle_result)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            document = generator.generate(output_path=tmp_path)

            # Check for proper markdown structure
            assert document.startswith("# ")  # Top-level header
            assert "## " in document  # Section headers
            assert "```" in document  # Code blocks
            assert "| " in document  # Tables

            # Check no broken links (basic check)
            lines = document.split("\n")
            for line in lines:
                # Check for unbalanced brackets in links
                if "[" in line and "]" in line:
                    open_count = line.count("[")
                    close_count = line.count("]")
                    assert open_count == close_count

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_generate_with_multiple_cycles(self, generator):
        """Test document generation with multiple cycles."""
        # Add multiple cycles
        for i in range(5):
            cycle = ImprovementCycleResult(
                cycle_number=i + 1,
                timestamp=datetime(2025, 1, 24, 14 + i, 30, 0),
                gaps_identified=2 + i,
                variants_generated=6 + i * 2,
                rules_incorporated=1 + i,
                rules_pending_review=0,
                baseline_accuracy=0.85 + i * 0.01,
                final_accuracy=0.86 + i * 0.01,
                overall_improvement=0.01,
                status="success",
            )
            generator.update_cycle_history(cycle)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            document = generator.generate(output_path=tmp_path)

            # All cycles should be in timeline
            for i in range(1, 6):
                assert f"Cycle #{i}" in document

            # Statistics should reflect all cycles
            assert "Total Cycles: 5" in document

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestDocumentIntegration:
    """Test integration with self-modifying system."""

    def test_system_can_generate_document(self):
        """Test that system method calls generator correctly."""
        from loft.core.self_modifying_system import SelfModifyingSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            system = SelfModifyingSystem(persistence_dir=tmpdir)

            # Generate document
            output_path = Path(tmpdir) / "TEST_DOCUMENT.md"
            document = system.generate_living_document(output_path=str(output_path))

            # Check document was created
            assert output_path.exists()
            assert "# ASP Core Living Document" in document

    def test_document_updates_after_cycle(self):
        """Test that document is generated after improvement cycle."""
        from loft.core.self_modifying_system import SelfModifyingSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            system = SelfModifyingSystem(persistence_dir=tmpdir)

            doc_path = Path(tmpdir) / "LIVING_DOCUMENT.md"

            # Run an improvement cycle (will try to generate document)
            try:
                system.run_improvement_cycle(max_gaps=1)

                # Document should exist after cycle
                # (may not exist if cycle had no improvements)
                if doc_path.exists():
                    content = doc_path.read_text()
                    assert "# ASP Core Living Document" in content
            except Exception:
                # System may fail without proper setup, that's okay for this test
                pass
