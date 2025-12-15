"""
Demo: LLM Integration and Living Document Generation

This demonstrates:
1. System with LLM integration (if API key configured)
2. Persistent ASP rule storage to disk
3. Living document generation showing rule evolution
4. Observable ASP core that grows naturally with testing
"""

import os
from pathlib import Path

from loft.core.self_modifying_system import SelfModifyingSystem
from loft.symbolic.stratification import StratificationLevel


def demonstrate_persistence_and_llm():
    """Demonstrate persistent storage and LLM integration."""
    print("=" * 80)
    print("DEMO: LLM Integration & Persistent ASP Core")
    print("=" * 80)
    print()

    # Check if LLM API key is configured
    api_key = os.getenv("ANTHROPIC_API_KEY")
    enable_llm = bool(api_key)

    print(f"LLM Integration: {'Enabled ✓' if enable_llm else 'Disabled (Mock Mode)'}")
    if not enable_llm:
        print("  (Set ANTHROPIC_API_KEY in .env to enable real LLM generation)")
    print()

    # Setup persistence directory
    persistence_dir = "./demo_asp_rules"
    print(f"Persistence Directory: {persistence_dir}")
    print()

    # Initialize system with LLM and persistence
    print("1. Initializing Self-Modifying System...")
    system = SelfModifyingSystem(enable_llm=enable_llm, persistence_dir=persistence_dir)

    # Check if rules were loaded from previous run
    initial_rules = {}
    for layer in StratificationLevel:
        count = len(system.asp_core.get_rules_by_layer(layer))
        initial_rules[layer] = count

    total_initial = sum(initial_rules.values())
    if total_initial > 0:
        print(f"   ✓ Loaded {total_initial} existing rules from disk")
        for layer, count in initial_rules.items():
            if count > 0:
                print(f"     - {layer.value}: {count} rules")
    else:
        print("   ✓ No existing rules (fresh start)")
    print()

    # Run improvement cycles
    print("2. Running Improvement Cycles...")
    print()

    for cycle_num in range(1, 4):
        print(f"   --- Cycle {cycle_num} ---")
        result = system.run_improvement_cycle(
            max_gaps=2, target_layer=StratificationLevel.TACTICAL
        )
        print(f"   Status: {result.status}")
        print(
            f"   Gaps: {result.gaps_identified}, Incorporated: {result.rules_incorporated}"
        )
        print(
            f"   Performance: {result.baseline_accuracy:.1%} → {result.final_accuracy:.1%}"
        )
        print()

    # Show final rule counts
    print("3. Final ASP Core State:")
    total_rules = 0
    for layer in StratificationLevel:
        count = len(system.asp_core.get_rules_by_layer(layer))
        total_rules += count
        if count > 0:
            delta = count - initial_rules[layer]
            delta_str = f" (+{delta})" if delta > 0 else ""
            print(f"   {layer.value:15} : {count:3} rules{delta_str}")

    print(f"   {'TOTAL':15} : {total_rules:3} rules")
    print()

    # Show persisted files
    print("4. Persisted Files:")
    persistence_path = Path(persistence_dir)
    if persistence_path.exists():
        for layer_file in sorted(persistence_path.glob("*.lp")):
            size = layer_file.stat().st_size
            print(f"   {layer_file.name:20} : {size:5} bytes")

        doc_file = persistence_path / "LIVING_DOCUMENT.md"
        if doc_file.exists():
            size = doc_file.stat().st_size
            print(f"   {'LIVING_DOCUMENT.md':20} : {size:5} bytes")
    print()

    # Generate and show living document excerpt
    print("5. Living Document Generated:")
    doc = system.generate_living_document()
    print(f"   Location: {persistence_dir}/LIVING_DOCUMENT.md")
    print(f"   Size: {len(doc):,} characters")
    print()

    # Show excerpt from living document
    print("6. Living Document Excerpt:")
    print("   " + "-" * 76)
    lines = doc.split("\n")
    for line in lines[:25]:  # Show first 25 lines
        print(f"   {line}")
    print("   " + "-" * 76)
    print(f"   ... ({len(lines) - 25} more lines)")
    print()

    # Self-analysis
    print("7. System Self-Analysis:")
    report = system.get_self_report()
    print(f"   Incorporation Success Rate: {report.incorporation_success_rate:.1%}")
    print(f"   Self-Confidence: {report.confidence_in_self:.1%}")
    print(f"   Best Strategy: {report.best_strategy or 'N/A'}")
    print()
    print("   Narrative:")
    for line in report.narrative.split(". ")[:3]:  # First 3 sentences
        if line.strip():
            print(f"   - {line.strip()}.")
    print()

    # Summary
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ LLM integration for rule generation")
    print("  ✓ Persistent ASP rule storage (survives restarts)")
    print("  ✓ Living document auto-generation")
    print("  ✓ Observable ASP core with stratification")
    print("  ✓ Self-analysis and meta-reasoning")
    print()
    print("Next Steps:")
    print("  - View living document: cat ./demo_asp_rules/LIVING_DOCUMENT.md")
    print("  - Inspect persisted rules: ls -lh ./demo_asp_rules/")
    print("  - Run again to see rules loaded from disk")
    print("  - Enable LLM: export ANTHROPIC_API_KEY=your_key")
    print()


if __name__ == "__main__":
    demonstrate_persistence_and_llm()
