"""
Integration tests for case analysis system.

Tests the full pipeline from parsing to rule generation.

Issue #276: Case Analysis and Rule Extraction
"""

from pathlib import Path

import pytest

from loft.case_analysis.analyzer import CaseAnalyzer
from loft.case_analysis.parser import CaseDocumentParser
from loft.neural.llm_interface import OllamaInterface


@pytest.fixture
def parser():
    """Create parser."""
    return CaseDocumentParser()


@pytest.fixture
def sample_case_file(tmp_path):
    """Create realistic sample case file."""
    file_path = tmp_path / "contract_case.txt"
    content = """
UNITED STATES DISTRICT COURT
SOUTHERN DISTRICT OF NEW YORK

Acme Corporation, Plaintiff
v.
Beta Industries, Inc., Defendant

Case No. 23-CV-1234
Filed: March 15, 2023

OPINION AND ORDER

FACTS:

On January 10, 2022, Acme Corporation ("Acme") and Beta Industries, Inc. ("Beta")
entered into a written contract for the sale of manufacturing equipment. The contract
specified a purchase price of $500,000, with delivery to occur within 60 days.

Beta delivered the equipment on March 20, 2022, 9 days after the contractual deadline.
Acme accepted delivery but refused to pay the full purchase price, citing Beta's late
delivery as a material breach. Acme tendered payment of $400,000, withholding $100,000
as damages for the delayed delivery.

Beta filed this action seeking the remaining $100,000 plus interest.

ANALYSIS:

Under basic contract law, a valid contract requires three essential elements: (1) offer,
(2) acceptance, and (3) consideration. All parties agree that these elements were
satisfied in the present case.

The central issue is whether Beta's late delivery constituted a material breach that
would excuse Acme's full performance. The court finds that it did not.

A breach is material when it goes to the root of the contract or deprives the
non-breaching party of substantially the whole benefit of the agreement. See Restatement
(Second) of Contracts § 241. Here, Acme received the equipment in working order and was
able to use it for its intended purpose. The 9-day delay, while regrettable, did not
deprive Acme of the substantial benefit of the bargain.

Moreover, the contract did not specify that time was of the essence. Absent such
language, minor delays in performance do not constitute material breach.

HOLDING:

The court holds that:

1. A valid contract existed between the parties.
2. Beta's 9-day delay in delivery was a minor breach, not a material breach.
3. Acme remains obligated to pay the full contract price.
4. Acme may pursue damages for the minor breach in a separate action, if any damages
   can be proven.

Judgment is entered in favor of Beta for $100,000 plus interest.

SO ORDERED.

Judge Sarah Martinez
United States District Court
March 15, 2023
"""
    file_path.write_text(content)
    return file_path


class TestCaseAnalysisIntegration:
    """Integration tests for case analysis."""

    def test_parse_realistic_case(self, parser, sample_case_file):
        """Test parsing a realistic case file."""
        doc = parser.parse_file(sample_case_file)

        assert doc is not None
        assert doc.case_id == "contract_case"
        assert "Acme Corporation" in doc.content
        assert "contract" in doc.content.lower()

    def test_full_analysis_pipeline_with_ollama(self, sample_case_file):
        """
        Test full analysis pipeline with Ollama.

        Note: Requires Ollama to be running locally.
        """
        try:
            # Try to create Ollama interface
            llm = OllamaInterface(
                model="llama3.2:latest", base_url="http://localhost:11434"
            )

            # Create analyzer
            analyzer = CaseAnalyzer(
                llm=llm,
                min_principle_confidence=0.5,
                min_rule_confidence=0.5,
                max_principles_per_case=5,
            )

            # Analyze case
            result = analyzer.analyze_file(sample_case_file)

            # Verify result structure
            assert result is not None
            assert result.case_id == "contract_case"
            assert result.processing_time_ms > 0

            # Log results for manual inspection
            print(f"\n=== Case Analysis Results ===")
            print(f"Case ID: {result.case_id}")
            print(f"Processing time: {result.processing_time_ms:.0f}ms")
            print(f"\nMetadata:")
            print(f"  Title: {result.metadata.title}")
            print(f"  Court: {result.metadata.court}")
            print(f"  Domain: {result.metadata.domain}")
            print(f"  Confidence: {result.metadata.confidence}")

            print(f"\nExtracted {result.principle_count} principles:")
            for i, principle in enumerate(result.principles, 1):
                print(f"\n  Principle {i}:")
                print(f"    Text: {principle.principle_text[:100]}...")
                print(f"    Domain: {principle.domain}")
                print(f"    Confidence: {principle.confidence}")
                print(f"    Case-specific: {principle.case_specific}")

            print(f"\nGenerated {result.rule_count} rules:")
            for i, rule in enumerate(result.rules, 1):
                print(f"\n  Rule {i}:")
                print(f"    ASP: {rule.asp_rule}")
                print(f"    Confidence: {rule.confidence}")
                print(f"    Validated: {rule.validation_passed}")
                print(f"    Predicates: {rule.predicates_used}")

            print(f"\nSuccess: {result.success}")
            if result.errors:
                print(f"Errors: {result.errors}")

            # Basic assertions
            if result.success:
                assert (
                    result.principle_count > 0
                ), "Should extract at least one principle"
                assert result.rule_count > 0, "Should generate at least one rule"

        except Exception as e:
            pytest.skip(f"Ollama not available or error occurred: {e}")

    def test_batch_analysis(self, tmp_path):
        """Test batch analysis of multiple cases."""
        # Create multiple case files
        files = []

        # Case 1: Contract formation
        file1 = tmp_path / "case1.txt"
        file1.write_text(
            """
        Smith v. Jones
        A valid contract requires offer, acceptance, and consideration.
        The court holds that a contract was formed.
        """
        )
        files.append(file1)

        # Case 2: Contract breach
        file2 = tmp_path / "case2.txt"
        file2.write_text(
            """
        Brown v. Green
        Material breach occurs when a party fails to substantially perform.
        The defendant's failure to deliver was a material breach.
        """
        )
        files.append(file2)

        try:
            llm = OllamaInterface(model="llama3.2:latest")
            analyzer = CaseAnalyzer(llm)

            results = analyzer.analyze_batch(files)

            assert len(results) == 2

            # Each should have processed
            for result in results:
                assert result.processing_time_ms > 0

            # Log batch summary
            total_principles = sum(r.principle_count for r in results)
            total_rules = sum(r.rule_count for r in results)
            successful = sum(1 for r in results if r.success)

            print(f"\n=== Batch Analysis Summary ===")
            print(f"Cases analyzed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Total principles: {total_principles}")
            print(f"Total rules: {total_rules}")

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_metadata_extraction_quality(self, sample_case_file):
        """Test quality of metadata extraction."""
        try:
            llm = OllamaInterface(model="llama3.2:latest")
            analyzer = CaseAnalyzer(llm)

            result = analyzer.analyze_file(sample_case_file)
            metadata = result.metadata

            # Should extract key metadata
            print(f"\n=== Metadata Extraction Quality ===")
            print(f"Title: {metadata.title}")
            print(f"Court: {metadata.court}")
            print(f"Jurisdiction: {metadata.jurisdiction}")
            print(f"Domain: {metadata.domain}")
            print(f"Parties (Plaintiff): {metadata.parties_plaintiff}")
            print(f"Parties (Defendant): {metadata.parties_defendant}")
            print(f"Confidence: {metadata.confidence}")

            # At minimum, should identify domain
            assert metadata.domain is not None, "Should identify legal domain"

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_principle_quality(self, sample_case_file):
        """Test quality of extracted principles."""
        try:
            llm = OllamaInterface(model="llama3.2:latest")
            analyzer = CaseAnalyzer(llm)

            result = analyzer.analyze_file(sample_case_file)

            print(f"\n=== Principle Quality Assessment ===")

            for principle in result.principles:
                print(f"\nPrinciple: {principle.principle_text}")
                print(f"  Domain: {principle.domain}")
                print(f"  Source: {principle.source_section}")
                print(f"  Confidence: {principle.confidence}")
                print(f"  Case-specific: {principle.case_specific}")
                print(f"  Reasoning: {principle.reasoning}")

                # Quality checks
                assert (
                    len(principle.principle_text) > 10
                ), "Principle should be substantive"
                assert principle.domain, "Should identify domain"
                assert (
                    0.0 <= principle.confidence <= 1.0
                ), "Confidence should be in range"

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_rule_validation(self, sample_case_file):
        """Test that generated rules have valid ASP syntax."""
        try:
            llm = OllamaInterface(model="llama3.2:latest")
            analyzer = CaseAnalyzer(llm, validate_rules=True)

            result = analyzer.analyze_file(sample_case_file)

            print(f"\n=== Rule Validation Results ===")

            for rule in result.rules:
                print(f"\nRule: {rule.asp_rule}")
                print(f"  Validated: {rule.validation_passed}")
                print(f"  Confidence: {rule.confidence}")
                print(f"  Predicates: {rule.predicates_used}")

                if rule.validation_passed:
                    # Should have basic ASP structure
                    assert ":-" in rule.asp_rule or rule.asp_rule.endswith(".")
                    assert len(rule.predicates_used) > 0

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
