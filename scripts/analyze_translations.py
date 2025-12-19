#!/usr/bin/env python3
"""
Analyze Translation Logs Script

Analyzes translation logs, generates pattern guides, and identifies
common success/failure patterns.

Usage:
    python scripts/analyze_translations.py \\
        --translations data/translation_logs/ \\
        --output reports/translation_patterns.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.translation.pattern_documenter import (
    TranslationPatternDocumenter,
)


def load_translation_logs(log_dir: Path) -> List[Dict[str, Any]]:
    """
    Load translation logs from directory.

    Args:
        log_dir: Directory containing translation log files

    Returns:
        List of translation dictionaries
    """
    translations = []

    if not log_dir.exists():
        print(f"Warning: Log directory {log_dir} does not exist")
        return translations

    # Load all JSON files
    for json_file in log_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

                # Handle different log formats
                if isinstance(data, list):
                    translations.extend(data)
                elif isinstance(data, dict):
                    # Single translation or batch
                    if "translations" in data:
                        translations.extend(data["translations"])
                    else:
                        translations.append(data)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return translations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze translation logs and generate pattern guide"
    )
    parser.add_argument(
        "--translations",
        type=str,
        required=True,
        help="Directory containing translation log files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/translation_patterns.md",
        help="Output path for pattern guide (default: reports/translation_patterns.md)",
    )
    parser.add_argument(
        "--min-fidelity",
        type=float,
        default=0.0,
        help="Minimum fidelity threshold for inclusion (default: 0.0)",
    )
    parser.add_argument(
        "--max-translations",
        type=int,
        default=None,
        help="Maximum number of translations to analyze",
    )

    args = parser.parse_args()

    log_dir = Path(args.translations)
    output_path = Path(args.output)

    print(f"Loading translation logs from {log_dir}...")
    translation_logs = load_translation_logs(log_dir)

    if not translation_logs:
        print("No translation logs found. Exiting.")
        return 1

    print(f"Loaded {len(translation_logs)} translations")

    # Filter by fidelity if specified
    if args.min_fidelity > 0.0:
        filtered = [
            t for t in translation_logs if t.get("fidelity", 0.0) >= args.min_fidelity
        ]
        print(
            f"Filtered to {len(filtered)} translations with fidelity >= {args.min_fidelity}"
        )
        translation_logs = filtered

    # Limit if specified
    if args.max_translations:
        translation_logs = translation_logs[: args.max_translations]
        print(f"Limited to {len(translation_logs)} translations")

    # Create documenter
    documenter = TranslationPatternDocumenter(output_path=output_path)

    # Record all translations
    print("Analyzing translation patterns...")
    for idx, trans in enumerate(translation_logs):
        original = trans.get("original", "")
        translated = trans.get("translated", "")
        back_translated = trans.get("back_translated", "")
        fidelity = trans.get("fidelity", 0.0)
        metadata = trans.get("metadata", {})

        documenter.record_translation(
            original=original,
            translated=translated,
            back_translated=back_translated,
            fidelity=fidelity,
            metadata=metadata,
        )

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(translation_logs)} translations...")

    # Analyze patterns
    print("Generating pattern analysis...")
    analysis = documenter.analyze_patterns()

    print("\n=== Analysis Summary ===")
    print(f"Total translations: {analysis.total_translations}")
    print(f"Average fidelity: {analysis.avg_fidelity:.2%}")
    print("\nFidelity by rule type:")
    for rtype, fidelity in sorted(
        analysis.fidelity_by_type.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {rtype}: {fidelity:.2%}")

    # Generate and save pattern guide
    print(f"\nSaving pattern guide to {output_path}...")
    saved_path = documenter.save_guide(str(output_path))

    print(f"\n✓ Pattern guide generated successfully: {saved_path}")

    # Print summary statistics
    if analysis.common_failure_patterns:
        print(
            f"\n⚠ Found {len(analysis.common_failure_patterns)} common failure patterns"
        )
        for idx, failure in enumerate(analysis.common_failure_patterns[:3], 1):
            print(
                f"  {idx}. {failure['rule_type']} ({failure['count']} occurrences, "
                f"avg fidelity: {failure['avg_fidelity']:.2%})"
            )

    if analysis.successful_patterns:
        print(f"\n✓ Found {len(analysis.successful_patterns)} successful patterns")
        for idx, success in enumerate(analysis.successful_patterns[:3], 1):
            print(
                f"  {idx}. {success['rule_type']} ({success['count']} occurrences, "
                f"avg fidelity: {success['avg_fidelity']:.2%})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
