"""
Demo script for case analysis and rule extraction.

Demonstrates how to:
1. Parse case documents
2. Extract legal principles
3. Generate ASP rules
4. Analyze complete cases

Issue #276: Case Analysis and Rule Extraction

Usage:
    python examples/case_analysis_demo.py

Requires Ollama to be running locally.
"""

from pathlib import Path

from loft.case_analysis.analyzer import CaseAnalyzer
from loft.case_analysis.parser import CaseDocumentParser
from loft.neural.llm_interface import OllamaInterface


def demo_parser():
    """Demo 1: Parsing case documents."""
    print("\n" + "=" * 60)
    print("DEMO 1: Parsing Case Documents")
    print("=" * 60)

    parser = CaseDocumentParser()

    # Parse text content
    case_text = """
    SUPREME COURT OF EXAMPLE STATE

    Johnson v. Smith
    Case No. 2023-SC-0123
    Decided: June 1, 2023

    FACTS:
    Johnson entered into a contract with Smith to purchase a vehicle for $25,000.

    ANALYSIS:
    A valid contract requires mutual assent, evidenced by offer and acceptance.
    Here, Johnson made an offer which Smith accepted.

    HOLDING:
    The court holds that a valid contract was formed.
    """

    doc = parser.parse_text(case_text, case_id="johnson_v_smith")

    print(f"\nParsed Document:")
    print(f"  Case ID: {doc.case_id}")
    print(f"  Format: {doc.format.value}")
    print(f"  Content length: {len(doc.content)} characters")
    print(f"  Content preview: {doc.content[:100]}...")


def demo_metadata_extraction():
    """Demo 2: Extracting case metadata."""
    print("\n" + "=" * 60)
    print("DEMO 2: Metadata Extraction")
    print("=" * 60)

    try:
        llm = OllamaInterface(
            model="llama3.2:latest", base_url="http://localhost:11434"
        )
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        print("Please ensure Ollama is running: ollama serve")
        return

    from loft.case_analysis.extractor import MetadataExtractor
    from loft.case_analysis.schemas import CaseDocument, CaseFormat

    case_doc = CaseDocument(
        content="""
        UNITED STATES COURT OF APPEALS
        NINTH CIRCUIT

        Alice Corp. v. Bob Industries
        No. 22-1234

        Filed: April 15, 2023
        Before: Judges Martinez, Chen, and O'Brien

        This case involves a contract dispute...
        """,
        format=CaseFormat.TEXT,
        case_id="alice_v_bob",
    )

    extractor = MetadataExtractor(llm)
    metadata = extractor.extract_metadata(case_doc)

    print(f"\nExtracted Metadata:")
    print(f"  Case ID: {metadata.case_id}")
    print(f"  Title: {metadata.title}")
    print(f"  Court: {metadata.court}")
    print(f"  Jurisdiction: {metadata.jurisdiction}")
    print(f"  Domain: {metadata.domain}")
    print(f"  Confidence: {metadata.confidence:.2f}")


def demo_principle_extraction():
    """Demo 3: Extracting legal principles."""
    print("\n" + "=" * 60)
    print("DEMO 3: Legal Principle Extraction")
    print("=" * 60)

    try:
        llm = OllamaInterface(
            model="llama3.2:latest", base_url="http://localhost:11434"
        )
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        return

    from loft.case_analysis.extractor import PrincipleExtractor
    from loft.case_analysis.schemas import CaseDocument, CaseFormat

    case_doc = CaseDocument(
        content="""
        STATE SUPREME COURT

        Miller v. Davis
        No. 2023-456

        ANALYSIS:

        The essential elements of a valid contract are well-established:
        (1) offer, (2) acceptance, and (3) consideration. Each element must
        be present for a binding agreement to exist.

        Moreover, consideration must be bargained-for and given in exchange
        for the promise. Past consideration, already performed before the
        promise is made, is generally insufficient to support a contract.

        HOLDING:

        We hold that the alleged contract lacks valid consideration because
        the plaintiff's performance preceded the defendant's promise.
        """,
        format=CaseFormat.TEXT,
        case_id="miller_v_davis",
        title="Miller v. Davis",
    )

    extractor = PrincipleExtractor(llm, min_confidence=0.5, max_principles=5)
    principles = extractor.extract_principles(case_doc)

    print(f"\nExtracted {len(principles)} legal principles:\n")
    for i, principle in enumerate(principles, 1):
        print(f"Principle {i}:")
        print(f"  Text: {principle.principle_text}")
        print(f"  Domain: {principle.domain}")
        print(f"  Source: {principle.source_section}")
        print(f"  Confidence: {principle.confidence:.2f}")
        print(f"  Case-specific: {principle.case_specific}")
        if principle.reasoning:
            print(f"  Reasoning: {principle.reasoning[:100]}...")
        print()


def demo_full_analysis():
    """Demo 4: Complete case analysis pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 4: Complete Case Analysis")
    print("=" * 60)

    try:
        llm = OllamaInterface(
            model="llama3.2:latest", base_url="http://localhost:11434"
        )
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        return

    # Create a realistic case document
    case_text = """
    FEDERAL DISTRICT COURT
    EASTERN DISTRICT OF EXAMPLE

    TechStart LLC v. InnovateCo
    Civil Action No. 23-CV-789
    Decided: May 10, 2023

    FACTS:

    TechStart LLC ("TechStart") and InnovateCo entered into a Software
    Development Agreement on January 15, 2022. Under the agreement, InnovateCo
    promised to develop custom software for TechStart in exchange for $100,000.

    The contract specified that the software would be delivered by June 1, 2022.
    InnovateCo delivered the software on August 15, 2022, 75 days late.

    TechStart accepted the software but withheld $25,000 of the contract price,
    claiming InnovateCo's late delivery was a material breach.

    ANALYSIS:

    A valid contract requires three essential elements: offer, acceptance, and
    consideration. All parties concede these elements were present.

    The key question is whether InnovateCo's late delivery constituted a
    material breach excusing TechStart's full performance. We apply the test
    from Restatement (Second) of Contracts § 241.

    A breach is material when it substantially deprives the non-breaching party
    of the benefit they reasonably expected. Here, TechStart received functional
    software that met the contract specifications. The delay, while significant,
    did not deprive TechStart of the substantial benefit of the bargain.

    Furthermore, the contract did not contain a "time is of the essence" clause.
    Absent such language, delays in performance are typically considered minor
    breaches rather than material breaches.

    HOLDING:

    The court holds:

    1. A valid, enforceable contract existed between TechStart and InnovateCo.
    2. InnovateCo's 75-day delay was a breach of contract.
    3. The breach was minor, not material, because TechStart received the
       substantial benefit of the agreement.
    4. TechStart remains obligated to pay the full contract price.
    5. TechStart may seek damages for the delay in a separate action.

    Judgment for InnovateCo in the amount of $25,000.

    SO ORDERED.

    Judge Rebecca Thompson
    May 10, 2023
    """

    parser = CaseDocumentParser()
    case_doc = parser.parse_text(case_text, case_id="techstart_v_innovateco")

    analyzer = CaseAnalyzer(
        llm=llm,
        min_principle_confidence=0.5,
        min_rule_confidence=0.5,
        max_principles_per_case=10,
        validate_rules=True,
    )

    print("\nAnalyzing case document...")
    result = analyzer.analyze_document(case_doc)

    print(f"\n{'─' * 60}")
    print("ANALYSIS RESULTS")
    print(f"{'─' * 60}")

    print(f"\nCase: {result.case_id}")
    print(f"Processing time: {result.processing_time_ms:.0f}ms")
    print(f"Success: {result.success}")

    if result.errors:
        print(f"\nErrors encountered:")
        for error in result.errors:
            print(f"  - {error}")

    print(f"\n{'─' * 60}")
    print("METADATA")
    print(f"{'─' * 60}")
    print(f"Title: {result.metadata.title}")
    print(f"Court: {result.metadata.court}")
    print(f"Jurisdiction: {result.metadata.jurisdiction}")
    print(f"Domain: {result.metadata.domain}")
    print(f"Confidence: {result.metadata.confidence:.2f}")

    print(f"\n{'─' * 60}")
    print(f"LEGAL PRINCIPLES ({result.principle_count})")
    print(f"{'─' * 60}")
    for i, principle in enumerate(result.principles, 1):
        print(f"\n{i}. {principle.principle_text}")
        print(f"   Domain: {principle.domain}")
        print(f"   Source: {principle.source_section}")
        print(f"   Confidence: {principle.confidence:.2f}")
        print(f"   General rule: {not principle.case_specific}")

    print(f"\n{'─' * 60}")
    print(f"GENERATED ASP RULES ({result.rule_count})")
    print(f"{'─' * 60}")
    for i, rule in enumerate(result.rules, 1):
        print(f"\n{i}. {rule.asp_rule}")
        print(f"   Confidence: {rule.confidence:.2f}")
        print(f"   Validated: {'✓' if rule.validation_passed else '✗'}")
        print(f"   Predicates: {', '.join(rule.predicates_used)}")
        print(f"   Reasoning: {rule.reasoning[:100]}...")

    print(f"\n{'─' * 60}")
    print("SUMMARY")
    print(f"{'─' * 60}")
    print(f"Principles extracted: {result.principle_count}")
    print(f"Rules generated: {result.rule_count}")
    print(f"Average confidence: {result.avg_confidence:.2f}")
    print(f"Overall success: {'✓' if result.success else '✗'}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("CASE ANALYSIS AND RULE EXTRACTION DEMO")
    print("=" * 60)
    print("\nThis demo shows the complete case analysis pipeline:")
    print("  1. Parsing case documents")
    print("  2. Extracting metadata")
    print("  3. Identifying legal principles")
    print("  4. Generating ASP rules")

    try:
        demo_parser()
        demo_metadata_extraction()
        demo_principle_extraction()
        demo_full_analysis()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("\nThe case analysis module successfully:")
        print("  ✓ Parsed case documents from text")
        print("  ✓ Extracted structured metadata")
        print("  ✓ Identified legal principles")
        print("  ✓ Generated formal ASP rules")
        print("  ✓ Validated rule syntax")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
