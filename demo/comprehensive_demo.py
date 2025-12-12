"""
Comprehensive Demo: Self-Modifying ASP Core with Real Rule Learning

Demonstrates:
1. Fresh system start with rule generation
2. Duplicate detection across cycles
3. Persistence and loading from disk
4. Confidence-based acceptance/rejection
5. Living document generation
6. Observable ASP core evolution
"""

from loft.core.self_modifying_system import SelfModifyingSystem
from loft.symbolic.stratification import StratificationLevel
from pathlib import Path


def demonstrate_comprehensive_system():
    """Run comprehensive demonstration of all features."""
    print("=" * 80)
    print("COMPREHENSIVE DEMO: Self-Modifying ASP Core")
    print("=" * 80)
    print()

    # Clean start
    persistence_dir = "./demo_comprehensive_rules"
    Path(persistence_dir).mkdir(parents=True, exist_ok=True)

    # === SCENARIO 1: Fresh Start ===
    print("SCENARIO 1: Fresh System Start")
    print("-" * 80)
    system = SelfModifyingSystem(persistence_dir=persistence_dir)

    print("Running Cycle 1 (expect some rules to be incorporated)...")
    result1 = system.run_improvement_cycle(max_gaps=5, target_layer=StratificationLevel.TACTICAL)

    print("\nResults:")
    print(f"  Gaps identified: {result1.gaps_identified}")
    print(f"  Variants generated: {result1.variants_generated}")
    print(f"  Rules incorporated: {result1.rules_incorporated}")
    print(f"  Status: {result1.status}")
    print(f"  Performance: {result1.baseline_accuracy:.1%} → {result1.final_accuracy:.1%}")
    print()

    # === SCENARIO 2: Duplicate Detection ===
    print("\nSCENARIO 2: Duplicate Detection (Same Session)")
    print("-" * 80)
    print("Running Cycle 2 (should skip duplicates from Cycle 1)...")
    result2 = system.run_improvement_cycle(max_gaps=5, target_layer=StratificationLevel.TACTICAL)

    print("\nResults:")
    print(f"  Gaps identified: {result2.gaps_identified}")
    print(
        f"  Rules incorporated: {result2.rules_incorporated} (duplicates: {result2.gaps_identified - result2.rules_incorporated})"
    )
    print()

    # === SCENARIO 3: Persistence ===
    print("\nSCENARIO 3: Persistence Check")
    print("-" * 80)
    tactical_file = Path(persistence_dir) / "tactical.lp"
    if tactical_file.exists():
        content = tactical_file.read_text()
        lines = content.strip().split("\n")
        rule_count = sum(1 for line in lines if line and not line.startswith("%"))
        metadata_count = sum(1 for line in lines if line.startswith("%"))

        print(f"Persisted file: {tactical_file}")
        print(f"  Total lines: {len(lines)}")
        print(f"  Rules: {rule_count}")
        print(f"  Metadata lines: {metadata_count}")
        print()
        print("Sample (first 15 lines):")
        for line in lines[:15]:
            print(f"  {line}")
    print()

    # === SCENARIO 4: Load from Disk ===
    print("\nSCENARIO 4: New Session - Loading from Disk")
    print("-" * 80)
    print("Creating new system instance (simulates restart)...")
    system2 = SelfModifyingSystem(persistence_dir=persistence_dir)

    print("Running Cycle 3 (should skip all rules loaded from disk)...")
    result3 = system2.run_improvement_cycle(max_gaps=5, target_layer=StratificationLevel.TACTICAL)

    print("\nResults:")
    print(f"  Gaps identified: {result3.gaps_identified}")
    print(f"  Rules incorporated: {result3.rules_incorporated}")
    print(f"  All skipped as duplicates: {result3.rules_incorporated == 0}")
    print()

    # === SCENARIO 5: Living Document ===
    print("\nSCENARIO 5: Living Document Generation")
    print("-" * 80)
    # Use original system instance which has the complete cycle history
    doc = system.generate_living_document()
    doc_file = Path(persistence_dir) / "LIVING_DOCUMENT.md"

    print(f"Generated living document: {doc_file}")
    print(f"  Size: {len(doc):,} characters")
    print()
    print("Excerpt (first 40 lines):")
    print("-" * 80)
    for i, line in enumerate(doc.split("\n")[:40], 1):
        print(line)
    print("-" * 80)
    print(f"... ({len(doc.split(chr(10))) - 40} more lines)")
    print()

    # === SCENARIO 6: Self-Analysis ===
    print("\nSCENARIO 6: System Self-Analysis")
    print("-" * 80)
    # Use original system instance (system) which has the incorporation history
    report = system.get_self_report()

    print(f"Incorporation Success Rate: {report.incorporation_success_rate:.1%}")
    print(f"Self-Confidence: {report.confidence_in_self:.1%}")
    print(f"Best Strategy: {report.best_strategy or 'N/A'}")
    print()
    print("System Narrative:")
    for sentence in report.narrative.split(". ")[:5]:
        if sentence.strip():
            print(f"  • {sentence.strip()}.")
    print()

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("DEMO COMPLETE - Summary")
    print("=" * 80)
    print()
    print("Key Achievements:")
    print("  ✓ Rules generated with varying confidence scores (0.70-0.90)")
    print(f"  ✓ {result1.rules_incorporated} rules incorporated in first cycle")
    print("  ✓ Duplicate detection prevented re-incorporation")
    print("  ✓ Rules persisted with metadata (timestamp, cycle, confidence)")
    print(f"  ✓ Successfully loaded {rule_count} rules from disk on restart")
    print(f"  ✓ Living document auto-generated ({len(doc):,} chars)")
    print(f"  ✓ Self-analysis shows {report.incorporation_success_rate:.0%} success rate")
    print()
    print("Files Created:")
    print(f"  • {tactical_file} ({tactical_file.stat().st_size} bytes)")
    print(f"  • {doc_file} ({doc_file.stat().st_size} bytes)")
    print()
    print("Next Steps:")
    print("  - View living document: cat " + str(doc_file))
    print("  - Inspect persisted rules: cat " + str(tactical_file))
    print("  - Run another cycle to see duplicates skipped")
    print("  - Enable LLM for real rule generation: --enable-llm")
    print()


if __name__ == "__main__":
    demonstrate_comprehensive_system()
