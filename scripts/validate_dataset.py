#!/usr/bin/env python3
"""
Dataset validation script for transfer learning datasets.

Validates that datasets meet the requirements for effective transfer learning
as documented in docs/datasets/DESIGN_GUIDELINES.md.

Usage:
    python scripts/validate_dataset.py datasets/adverse_possession/
    python scripts/validate_dataset.py datasets/adverse_possession/ --strict
    python scripts/validate_dataset.py datasets/ --recursive

Exit codes:
    0: All validations passed
    1: Validation errors found
    2: Invalid arguments or file not found
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str  # "error", "warning", "info"
    code: str  # e.g., "MIN_SCENARIOS"
    message: str
    file: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    dataset_path: Path
    scenario_count: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    predicates_found: Set[str] = field(default_factory=set)
    common_predicates: Set[str] = field(default_factory=set)
    outcome_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(i.level == "warning" for i in self.issues)

    def print_report(self) -> None:
        """Print validation report to stdout."""
        print(f"\n{'=' * 60}")
        print(f"Dataset Validation Report: {self.dataset_path}")
        print(f"{'=' * 60}")

        print(f"\nScenarios found: {self.scenario_count}")
        print(f"Unique predicates: {len(self.predicates_found)}")
        print(f"Common predicates: {len(self.common_predicates)}")

        if self.outcome_distribution:
            print("\nOutcome distribution:")
            for outcome, count in sorted(self.outcome_distribution.items()):
                pct = (
                    count / self.scenario_count * 100 if self.scenario_count > 0 else 0
                )
                print(f"  {outcome}: {count} ({pct:.1f}%)")

        if self.common_predicates:
            print("\nCommon predicates across all scenarios:")
            for pred in sorted(self.common_predicates):
                print(f"  - {pred}")

        if self.issues:
            print(f"\n{'-' * 40}")
            print("Issues Found:")
            print(f"{'-' * 40}")

            errors = [i for i in self.issues if i.level == "error"]
            warnings = [i for i in self.issues if i.level == "warning"]
            infos = [i for i in self.issues if i.level == "info"]

            for issue in errors:
                file_str = f" [{issue.file}]" if issue.file else ""
                print(f"  ERROR [{issue.code}]{file_str}: {issue.message}")

            for issue in warnings:
                file_str = f" [{issue.file}]" if issue.file else ""
                print(f"  WARNING [{issue.code}]{file_str}: {issue.message}")

            for issue in infos:
                file_str = f" [{issue.file}]" if issue.file else ""
                print(f"  INFO [{issue.code}]{file_str}: {issue.message}")

            print(
                f"\nTotal: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info"
            )
        else:
            print("\nNo issues found.")

        print(f"\n{'=' * 60}")
        if self.has_errors:
            print("RESULT: FAILED")
        elif self.has_warnings:
            print("RESULT: PASSED WITH WARNINGS")
        else:
            print("RESULT: PASSED")
        print(f"{'=' * 60}\n")


class DatasetValidator:
    """Validate datasets for transfer learning requirements."""

    # Required JSON fields
    REQUIRED_FIELDS = {"id", "asp_facts", "ground_truth", "rationale"}

    # Optional but recommended fields
    RECOMMENDED_FIELDS = {"description", "facts", "question"}

    # Valid ground truth values
    VALID_OUTCOMES = {"enforceable", "unenforceable"}

    # Minimum requirements
    MIN_SCENARIOS = 10
    MIN_COMMON_PREDICATES = 3
    MIN_MINORITY_RATIO = 0.3  # At least 30% minority class

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, warnings become errors
        """
        self.strict = strict

    def validate_directory(self, dataset_path: Path) -> ValidationResult:
        """
        Validate all scenarios in a dataset directory.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(dataset_path=dataset_path)

        if not dataset_path.exists():
            result.issues.append(
                ValidationIssue(
                    level="error",
                    code="PATH_NOT_FOUND",
                    message=f"Dataset path does not exist: {dataset_path}",
                )
            )
            return result

        if not dataset_path.is_dir():
            result.issues.append(
                ValidationIssue(
                    level="error",
                    code="NOT_DIRECTORY",
                    message=f"Path is not a directory: {dataset_path}",
                )
            )
            return result

        # Find all JSON files
        scenario_files = list(dataset_path.glob("*.json"))
        result.scenario_count = len(scenario_files)

        if result.scenario_count == 0:
            result.issues.append(
                ValidationIssue(
                    level="error",
                    code="NO_SCENARIOS",
                    message="No JSON scenario files found in directory",
                )
            )
            return result

        # Validate each scenario and collect data
        all_predicates: List[Set[str]] = []
        outcomes: List[str] = []
        fact_hashes: List[int] = []
        scenario_ids: List[str] = []

        for scenario_file in sorted(scenario_files):
            file_issues, predicates, outcome, fact_hash, scenario_id = (
                self._validate_scenario_file(scenario_file)
            )
            result.issues.extend(file_issues)

            if predicates:
                all_predicates.append(predicates)
                result.predicates_found.update(predicates)

            if outcome:
                outcomes.append(outcome)

            if fact_hash is not None:
                fact_hashes.append(fact_hash)

            if scenario_id:
                scenario_ids.append(scenario_id)

        # Check minimum scenario count
        if result.scenario_count < self.MIN_SCENARIOS:
            level = "error" if self.strict else "warning"
            result.issues.append(
                ValidationIssue(
                    level=level,
                    code="MIN_SCENARIOS",
                    message=f"Dataset has {result.scenario_count} scenarios, "
                    f"recommended minimum is {self.MIN_SCENARIOS}",
                )
            )

        # Check common predicates
        if all_predicates:
            result.common_predicates = set.intersection(*all_predicates)
            if len(result.common_predicates) < self.MIN_COMMON_PREDICATES:
                result.issues.append(
                    ValidationIssue(
                        level="error",
                        code="PREDICATE_MISMATCH",
                        message=f"Only {len(result.common_predicates)} common predicates "
                        f"across scenarios (minimum: {self.MIN_COMMON_PREDICATES}). "
                        "Transfer learning may not work effectively.",
                    )
                )

        # Check outcome balance
        if outcomes:
            result.outcome_distribution = dict(Counter(outcomes))
            total = len(outcomes)
            for outcome, count in result.outcome_distribution.items():
                ratio = count / total
                if ratio < self.MIN_MINORITY_RATIO:
                    level = "error" if self.strict else "warning"
                    result.issues.append(
                        ValidationIssue(
                            level=level,
                            code="IMBALANCED_OUTCOMES",
                            message=f"Outcome '{outcome}' is only {ratio:.1%} of dataset "
                            f"(minimum: {self.MIN_MINORITY_RATIO:.0%})",
                        )
                    )

        # Check for duplicates
        if len(fact_hashes) != len(set(fact_hashes)):
            result.issues.append(
                ValidationIssue(
                    level="error",
                    code="DUPLICATE_FACTS",
                    message="Duplicate scenario facts detected. Each scenario should be unique.",
                )
            )

        # Check for duplicate IDs
        id_counts = Counter(scenario_ids)
        duplicates = [id for id, count in id_counts.items() if count > 1]
        if duplicates:
            result.issues.append(
                ValidationIssue(
                    level="error",
                    code="DUPLICATE_IDS",
                    message=f"Duplicate scenario IDs found: {duplicates}",
                )
            )

        return result

    def _validate_scenario_file(self, file_path: Path) -> Tuple[
        List[ValidationIssue],
        Optional[Set[str]],
        Optional[str],
        Optional[int],
        Optional[str],
    ]:
        """
        Validate a single scenario file.

        Returns:
            Tuple of (issues, predicates, outcome, fact_hash, scenario_id)
        """
        issues: List[ValidationIssue] = []
        predicates: Optional[Set[str]] = None
        outcome: Optional[str] = None
        fact_hash: Optional[int] = None
        scenario_id: Optional[str] = None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="INVALID_JSON",
                    message=f"Invalid JSON: {e}",
                    file=file_path.name,
                )
            )
            return issues, predicates, outcome, fact_hash, scenario_id

        # Check required fields
        missing_required = self.REQUIRED_FIELDS - set(data.keys())
        if missing_required:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="MISSING_REQUIRED",
                    message=f"Missing required fields: {missing_required}",
                    file=file_path.name,
                )
            )

        # Check recommended fields
        missing_recommended = self.RECOMMENDED_FIELDS - set(data.keys())
        if missing_recommended:
            issues.append(
                ValidationIssue(
                    level="info",
                    code="MISSING_RECOMMENDED",
                    message=f"Missing recommended fields: {missing_recommended}",
                    file=file_path.name,
                )
            )

        # Validate ground_truth
        if "ground_truth" in data:
            outcome = data["ground_truth"]
            if outcome not in self.VALID_OUTCOMES:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="INVALID_OUTCOME",
                        message=f"Invalid ground_truth '{outcome}'. "
                        f"Must be one of: {self.VALID_OUTCOMES}",
                        file=file_path.name,
                    )
                )

        # Validate asp_facts
        if "asp_facts" in data:
            asp_facts = data["asp_facts"]
            fact_hash = hash(asp_facts)

            # Extract predicates
            predicates = self._extract_predicates(asp_facts)

            # Validate ASP syntax
            syntax_issues = self._validate_asp_syntax(asp_facts)
            for issue_msg in syntax_issues:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="INVALID_ASP",
                        message=issue_msg,
                        file=file_path.name,
                    )
                )

        # Get scenario ID
        if "id" in data:
            scenario_id = data["id"]

        return issues, predicates, outcome, fact_hash, scenario_id

    def _extract_predicates(self, asp_facts: str) -> Set[str]:
        """Extract predicate names from ASP facts."""
        predicates = set()

        # Pattern to match predicates: name(args).
        pattern = r"([a-z_][a-z0-9_]*)\s*\("
        for match in re.finditer(pattern, asp_facts):
            predicates.add(match.group(1))

        return predicates

    def _validate_asp_syntax(self, asp_facts: str) -> List[str]:
        """Validate ASP syntax and return list of issues."""
        issues = []

        # Check for balanced parentheses
        open_count = asp_facts.count("(")
        close_count = asp_facts.count(")")
        if open_count != close_count:
            issues.append(
                f"Unbalanced parentheses: {open_count} '(' vs {close_count} ')'"
            )

        # Check facts end with periods
        facts = [f.strip() for f in asp_facts.split(".") if f.strip()]
        for fact in facts:
            # Skip if it's just whitespace after cleaning
            if not fact:
                continue

            # Check for basic predicate structure
            if not re.match(r"^[a-z_][a-z0-9_]*\s*\(", fact):
                if not fact.startswith("%"):  # Not a comment
                    issues.append(f"Invalid fact syntax: '{fact[:50]}...'")

        # Try to parse with Clingo if available
        try:
            import clingo

            ctl = clingo.Control()
            try:
                ctl.add("base", [], asp_facts)
                ctl.ground([("base", [])])
            except Exception as e:
                error_msg = str(e)
                # Extract just the relevant part of the error
                if ":" in error_msg:
                    error_msg = error_msg.split(":")[-1].strip()
                issues.append(f"Clingo parse error: {error_msg[:100]}")
        except ImportError:
            # Clingo not available, skip deep validation
            pass

        return issues


def validate_recursive(base_path: Path, strict: bool = False) -> List[ValidationResult]:
    """Validate all dataset directories under a base path."""
    results = []
    validator = DatasetValidator(strict=strict)

    # Find directories containing JSON files
    for path in base_path.iterdir():
        if path.is_dir():
            json_files = list(path.glob("*.json"))
            if json_files:
                result = validator.validate_directory(path)
                results.append(result)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate transfer learning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_dataset.py datasets/adverse_possession/
  python scripts/validate_dataset.py datasets/adverse_possession/ --strict
  python scripts/validate_dataset.py datasets/ --recursive
  python scripts/validate_dataset.py --all-domains
        """,
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to dataset directory or parent directory with --recursive",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Validate all dataset directories under the given path",
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Validate all domains in the datasets directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Determine which mode to use
    if args.all_domains:
        # Find datasets directory relative to script or cwd
        datasets_path = None
        candidates = [
            Path(__file__).parent.parent / "datasets",
            Path.cwd() / "datasets",
        ]
        for candidate in candidates:
            if candidate.exists():
                datasets_path = candidate
                break

        if datasets_path is None:
            print("Error: Could not find datasets directory", file=sys.stderr)
            sys.exit(2)

        results = validate_recursive(datasets_path, strict=args.strict)
    elif args.recursive:
        if args.path is None:
            print("Error: --recursive requires a path argument", file=sys.stderr)
            sys.exit(2)
        results = validate_recursive(args.path, strict=args.strict)
    else:
        if args.path is None:
            print(
                "Error: path argument required (or use --all-domains)", file=sys.stderr
            )
            sys.exit(2)
        validator = DatasetValidator(strict=args.strict)
        results = [validator.validate_directory(args.path)]

    if args.json:
        # JSON output
        output = []
        for result in results:
            output.append(
                {
                    "path": str(result.dataset_path),
                    "scenario_count": result.scenario_count,
                    "has_errors": result.has_errors,
                    "has_warnings": result.has_warnings,
                    "common_predicates": list(result.common_predicates),
                    "outcome_distribution": result.outcome_distribution,
                    "issues": [
                        {
                            "level": i.level,
                            "code": i.code,
                            "message": i.message,
                            "file": i.file,
                        }
                        for i in result.issues
                    ],
                }
            )
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        for result in results:
            result.print_report()

    # Return appropriate exit code
    has_errors = any(r.has_errors for r in results)
    if has_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
