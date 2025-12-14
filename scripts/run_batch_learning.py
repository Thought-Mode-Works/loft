#!/usr/bin/env python3
"""
CLI for running batch learning experiments.

Processes test cases through the learning pipeline with checkpointing,
progress tracking, and metrics collection.

Usage:
    # Run batch on a single dataset (demo mode)
    python scripts/run_batch_learning.py --dataset datasets/contracts/

    # Run with full LLM pipeline (Issue #253)
    python scripts/run_batch_learning.py \
        --dataset datasets/contracts/ \
        --enable-llm \
        --model claude-3-5-haiku-20241022

    # Run with custom model and persistence directory
    python scripts/run_batch_learning.py \
        --dataset datasets/contracts/ \
        --enable-llm \
        --model claude-3-5-sonnet-20241022 \
        --rules-dir ./my_rules

    # Run with multiple datasets
    python scripts/run_batch_learning.py \
        --datasets datasets/contracts/ datasets/torts/

    # Resume from checkpoint
    python scripts/run_batch_learning.py \
        --resume data/batch_runs/batch_20241129/checkpoints/checkpoint_0010.json \
        --dataset datasets/contracts/

    # Run with custom configuration
    python scripts/run_batch_learning.py \
        --dataset datasets/contracts/ \
        --checkpoint-interval 5 \
        --max-cases 50 \
        --validation-threshold 0.7

    # List previous batch runs
    python scripts/run_batch_learning.py --list-batches

    # Show batch result
    python scripts/run_batch_learning.py --show-batch batch_20241129_120000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.batch import (
    BatchConfig,
    BatchLearningHarness,
    BatchProgress,
    CaseResult,
    CaseStatus,
)


def load_test_cases(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """Load test cases from dataset directories."""
    cases = []

    for dataset_path in dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            continue

        # Load JSON files from directory
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    case = json.load(f)
                    # Ensure case has an ID
                    if "id" not in case:
                        case["id"] = json_file.stem
                    # Track source
                    case["_source_file"] = str(json_file)
                    case["_domain"] = path.name
                    cases.append(case)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {json_file}: {e}")

    return cases


def create_demo_processor():
    """
    Create a demo case processor for testing.

    This processor simulates rule generation and validation
    without actually calling LLMs.
    """

    def process_case(case: Dict[str, Any], accumulated_rules: List[str]) -> CaseResult:
        start_time = time.time()
        case_id = case.get("id", "unknown")

        # Simulate processing
        time.sleep(0.1)  # Simulate some work

        # Check if we have asp_facts to work with
        asp_facts = case.get("asp_facts", "")
        ground_truth = case.get("ground_truth", "")

        # Simulate a prediction (demo: randomly correct 60% of time)
        import random

        prediction_correct = random.random() < 0.6
        confidence = random.uniform(0.5, 0.95)

        # Simulate rule generation if prediction was wrong
        rules_generated = 0
        rules_accepted = 0
        rules_rejected = 0
        generated_rule_ids = []

        if not prediction_correct and asp_facts:
            # "Generate" 1-2 rules
            rules_generated = random.randint(1, 2)
            for i in range(rules_generated):
                # "Validate" each rule (80% acceptance rate)
                if random.random() < 0.8:
                    rules_accepted += 1
                    generated_rule_ids.append(f"rule_{case_id}_{i}")
                else:
                    rules_rejected += 1

        return CaseResult(
            case_id=case_id,
            status=CaseStatus.SUCCESS,
            processed_at=datetime.now(),
            processing_time_ms=(time.time() - start_time) * 1000,
            rules_generated=rules_generated,
            rules_accepted=rules_accepted,
            rules_rejected=rules_rejected,
            prediction_correct=prediction_correct,
            confidence=confidence,
            generated_rule_ids=generated_rule_ids,
            metadata={
                "domain": case.get("_domain", "unknown"),
                "ground_truth": ground_truth,
            },
        )

    return process_case


def progress_callback(progress: BatchProgress) -> None:
    """Print progress update."""
    pct = progress.completion_percentage
    acc = progress.current_accuracy * 100
    rules = progress.total_rules_accepted

    # Create progress bar
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "=" * filled + "-" * (bar_width - filled)

    print(
        f"\r[{bar}] {pct:.1f}% | "
        f"Cases: {progress.processed_cases}/{progress.total_cases} | "
        f"Accuracy: {acc:.1f}% | "
        f"Rules: {rules}",
        end="",
        flush=True,
    )


def run_batch(args) -> None:
    """Run batch learning on datasets."""
    # Load test cases
    cases = load_test_cases(args.datasets)
    if not cases:
        print("Error: No test cases found in specified datasets")
        sys.exit(1)

    print(f"Loaded {len(cases)} test cases from {len(args.datasets)} dataset(s)")

    # Create configuration
    config = BatchConfig(
        max_cases=args.max_cases,
        checkpoint_interval=args.checkpoint_interval,
        validation_threshold=args.validation_threshold,
        min_confidence=args.min_confidence,
        continue_on_error=not args.stop_on_error,
        use_dialectical=not args.no_dialectical,
    )

    # Create harness
    harness = BatchLearningHarness(
        config=config,
        output_dir=Path(args.output_dir),
    )

    # Set progress callback
    if not args.quiet:
        harness.set_callbacks(on_progress=progress_callback)

    # Create processor
    if args.enable_llm:
        print(f"Using full pipeline processor with model: {args.model}")
        try:
            from loft.batch.full_pipeline import create_full_pipeline_processor

            full_processor = create_full_pipeline_processor(
                model=args.model,
                persistence_dir=args.rules_dir,
                enable_persistence=not args.no_persistence,
                validation_threshold=args.validation_threshold,
            )
            processor = full_processor.process_case
            print(
                f"Persistence: {args.rules_dir if not args.no_persistence else 'disabled'}"
            )
        except Exception as e:
            print(f"Error creating full pipeline processor: {e}")
            print("Falling back to demo processor")
            processor = create_demo_processor()
    elif args.demo:
        print("Using demo processor (no LLM calls)")
        processor = create_demo_processor()
    else:
        print("Note: Using demo processor (use --enable-llm for full pipeline)")
        processor = create_demo_processor()

    # Run or resume
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        result = harness.resume_from_checkpoint(
            checkpoint_path=args.resume,
            test_cases=cases,
            process_case_fn=processor,
        )
    else:
        print("\nStarting batch processing...")
        result = harness.run_batch(
            test_cases=cases,
            process_case_fn=processor,
        )

    # Print results
    print("\n")  # New line after progress bar
    print("=" * 60)
    print("Batch Processing Complete")
    print("=" * 60)
    print(f"Batch ID: {result.batch_id}")
    print(f"Status: {result.status.value}")
    print(f"Duration: {(result.completed_at - result.started_at).total_seconds():.1f}s")
    print()
    print("Results:")
    print(f"  Total cases: {len(result.case_results)}")
    print(f"  Successful: {result.success_count}")
    print(f"  Failed: {result.failure_count}")
    print()
    print("Rules:")
    print(f"  Generated: {result.metrics.rules_generated if result.metrics else 0}")
    print(f"  Accepted: {result.total_rules_accepted}")
    print()
    print("Accuracy:")
    print(
        f"  Final: {result.metrics.accuracy_after * 100:.1f}%"
        if result.metrics
        else "  N/A"
    )
    print()
    print(f"Results saved to: {args.output_dir}/{result.batch_id}/")


def list_batches(args) -> None:
    """List all batch runs."""
    batches = BatchLearningHarness.list_batches(args.output_dir)

    if not batches:
        print("No batch runs found.")
        return

    print(f"Found {len(batches)} batch run(s):\n")
    print(f"{'Batch ID':<30} {'Status':<12} {'Cases':>8} {'Rules':>8} {'Accuracy':>10}")
    print("-" * 75)

    for batch in batches:
        batch_id = batch.get("batch_id", "unknown")[:30]
        status = batch.get("status", "unknown")
        cases = batch.get("total_cases", 0)
        rules = batch.get("rules_accepted", 0)
        accuracy = batch.get("accuracy", 0) * 100

        print(f"{batch_id:<30} {status:<12} {cases:>8} {rules:>8} {accuracy:>9.1f}%")


def show_batch(args) -> None:
    """Show details of a batch run."""
    result_path = Path(args.output_dir) / args.show_batch / "result.json"

    if not result_path.exists():
        print(f"Error: Batch result not found: {result_path}")
        sys.exit(1)

    result = BatchLearningHarness.load_result(str(result_path))

    print(f"Batch: {result.batch_id}")
    print(f"Status: {result.status.value}")
    print(f"Started: {result.started_at.isoformat()}")
    print(
        f"Completed: {result.completed_at.isoformat() if result.completed_at else 'N/A'}"
    )
    print()
    print("Configuration:")
    for key, value in result.config.items():
        print(f"  {key}: {value}")
    print()
    print("Results:")
    print(f"  Total cases: {len(result.case_results)}")
    print(f"  Successful: {result.success_count}")
    print(f"  Failed: {result.failure_count}")
    print(f"  Rules accepted: {result.total_rules_accepted}")

    if result.metrics:
        print()
        print("Metrics:")
        print(f"  Processing time: {result.metrics.total_processing_time_ms:.0f}ms")
        print(f"  Avg case time: {result.metrics.avg_case_time_ms:.0f}ms")
        print(f"  Accuracy: {result.metrics.accuracy_after * 100:.1f}%")
        print(f"  Errors: {result.metrics.total_errors}")

        if result.metrics.error_types:
            print()
            print("Error types:")
            for error_type, count in result.metrics.error_types.items():
                print(f"  {error_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        "-d",
        dest="datasets",
        action="append",
        help="Dataset directory to process (can specify multiple)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Multiple dataset directories to process",
    )

    # Resume option
    parser.add_argument(
        "--resume",
        "-r",
        help="Resume from checkpoint file",
    )

    # Configuration options
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of cases to process",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Create checkpoint every N cases (default: 10)",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.8,
        help="Minimum validation score for rule acceptance (default: 0.8)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for predictions (default: 0.6)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing on first error (default: continue)",
    )
    parser.add_argument(
        "--no-dialectical",
        action="store_true",
        help="Disable dialectical refinement",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/batch_runs",
        help="Directory for batch results (default: data/batch_runs)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    # Mode options
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo processor (no LLM calls)",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Use full pipeline processor with LLM integration (Issue #253)",
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-haiku-20241022",
        help="LLM model to use with --enable-llm (default: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--rules-dir",
        default="./asp_rules",
        help="Directory for ASP rule persistence (default: ./asp_rules)",
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable rule persistence to disk",
    )

    # List/show options
    parser.add_argument(
        "--list-batches",
        action="store_true",
        help="List all batch runs",
    )
    parser.add_argument(
        "--show-batch",
        metavar="BATCH_ID",
        help="Show details of a batch run",
    )

    args = parser.parse_args()

    # Handle list/show modes
    if args.list_batches:
        list_batches(args)
        return

    if args.show_batch:
        show_batch(args)
        return

    # Combine dataset arguments
    all_datasets = []
    if args.datasets:
        all_datasets.extend(args.datasets)
    if hasattr(args, "dataset") and args.dataset:
        all_datasets.extend(
            args.dataset if isinstance(args.dataset, list) else [args.dataset]
        )

    # For resume, dataset is optional but recommended
    if not all_datasets and not args.resume:
        parser.error("At least one --dataset is required (or use --resume)")

    # Set default datasets for running
    if not all_datasets:
        all_datasets = []

    args.datasets = all_datasets

    # Run batch
    run_batch(args)


if __name__ == "__main__":
    main()
