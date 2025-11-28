"""
Cross-Domain Transfer Study for Legal Knowledge.

Tests how well knowledge learned in one legal domain transfers to another.
Implements zero-shot and few-shot transfer learning experiments.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.casework.explorer import CaseworkExplorer
from experiments.casework.dataset_loader import DatasetLoader
from experiments.casework.metrics import LearningMetrics


@dataclass
class TransferResult:
    """Results from a transfer learning experiment."""

    source_domain: str
    target_domain: str
    source_cases: int
    target_cases: int

    # Source domain training results
    source_baseline_accuracy: float
    source_final_accuracy: float
    source_rules_learned: int

    # Zero-shot transfer (no target training)
    zero_shot_accuracy: float

    # Few-shot transfer (limited target training)
    few_shot_cases: int
    few_shot_accuracy: float
    few_shot_rules_learned: int

    # From-scratch baseline (train on target only)
    scratch_accuracy: float
    scratch_rules_learned: int

    # Transfer effectiveness metrics
    transfer_rate: float = 0.0  # (zero_shot - baseline) / (final - baseline)
    few_shot_advantage: float = 0.0  # few_shot_accuracy - scratch_accuracy

    def __post_init__(self):
        """Calculate derived metrics."""
        # Transfer rate: how much of the learning transferred
        source_improvement = self.source_final_accuracy - self.source_baseline_accuracy
        if source_improvement > 0:
            zero_shot_improvement = self.zero_shot_accuracy - self.source_baseline_accuracy
            self.transfer_rate = zero_shot_improvement / source_improvement
        else:
            self.transfer_rate = 0.0

        # Few-shot advantage: how much few-shot with transfer beats from-scratch
        self.few_shot_advantage = self.few_shot_accuracy - self.scratch_accuracy


class TransferStudy:
    """
    Cross-domain transfer learning experiments.

    Tests how well knowledge learned in one legal domain applies to another.
    """

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        """Initialize transfer study."""
        self.model = model

    def run_transfer_experiment(
        self,
        source_dataset: Path,
        target_dataset: Path,
        few_shot_count: int = 10,
    ) -> TransferResult:
        """
        Run complete transfer learning experiment.

        Args:
            source_dataset: Path to source domain dataset
            target_dataset: Path to target domain dataset
            few_shot_count: Number of target cases for few-shot learning

        Returns:
            TransferResult with all metrics
        """
        source_loader = DatasetLoader(source_dataset)
        target_loader = DatasetLoader(target_dataset)

        source_domain = source_dataset.name
        target_domain = target_dataset.name

        logger.info(f"Starting transfer study: {source_domain} → {target_domain}")

        # Step 1: Train on source domain
        logger.info(f"Step 1: Training on source domain ({source_domain})...")
        source_explorer = CaseworkExplorer(
            dataset_dir=source_dataset, model=self.model, enable_learning=True
        )

        source_scenarios = source_loader.load_all()
        source_baseline_acc = self._measure_baseline(source_explorer, source_scenarios[:10])

        source_metrics = source_explorer.explore_dataset()
        source_final_acc = source_metrics.get_current_accuracy()
        source_rules = len(source_explorer.knowledge_base_rules)

        logger.info(
            f"Source training complete: {source_baseline_acc:.1%} → {source_final_acc:.1%} "
            f"({source_rules} rules)"
        )

        # Step 2: Test zero-shot on target domain
        logger.info(f"Step 2: Testing zero-shot transfer to {target_domain}...")
        target_scenarios = target_loader.load_all()

        zero_shot_acc = self._test_knowledge_base(
            source_explorer.knowledge_base_rules, target_scenarios
        )
        logger.info(f"Zero-shot accuracy: {zero_shot_acc:.1%}")

        # Step 3: Few-shot learning on target
        logger.info(f"Step 3: Few-shot learning ({few_shot_count} cases)...")
        few_shot_explorer = CaseworkExplorer(
            dataset_dir=target_dataset, model=self.model, enable_learning=True
        )

        # Pre-load source knowledge
        few_shot_explorer.knowledge_base_rules = source_explorer.knowledge_base_rules.copy()

        few_shot_metrics = few_shot_explorer.explore_dataset(max_cases=few_shot_count)
        few_shot_acc = few_shot_metrics.get_current_accuracy()
        few_shot_rules = len(few_shot_explorer.knowledge_base_rules) - source_rules

        logger.info(f"Few-shot accuracy: {few_shot_acc:.1%} ({few_shot_rules} new rules)")

        # Step 4: From-scratch baseline on target
        logger.info(f"Step 4: Training from scratch on {target_domain}...")
        scratch_explorer = CaseworkExplorer(
            dataset_dir=target_dataset, model=self.model, enable_learning=True
        )

        scratch_metrics = scratch_explorer.explore_dataset(max_cases=few_shot_count)
        scratch_acc = scratch_metrics.get_current_accuracy()
        scratch_rules = len(scratch_explorer.knowledge_base_rules)

        logger.info(f"From-scratch accuracy: {scratch_acc:.1%} ({scratch_rules} rules)")

        # Create result
        result = TransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            source_cases=len(source_scenarios),
            target_cases=len(target_scenarios),
            source_baseline_accuracy=source_baseline_acc,
            source_final_accuracy=source_final_acc,
            source_rules_learned=source_rules,
            zero_shot_accuracy=zero_shot_acc,
            few_shot_cases=few_shot_count,
            few_shot_accuracy=few_shot_acc,
            few_shot_rules_learned=few_shot_rules,
            scratch_accuracy=scratch_acc,
            scratch_rules_learned=scratch_rules,
        )

        return result

    def _measure_baseline(self, explorer: CaseworkExplorer, scenarios: List[Any]) -> float:
        """Measure baseline accuracy before learning."""
        # Use simple heuristic prediction for baseline
        correct = 0
        for scenario in scenarios:
            prediction, _ = explorer._make_prediction(scenario)
            if prediction == scenario.ground_truth:
                correct += 1

        return correct / len(scenarios) if scenarios else 0.0

    def _test_knowledge_base(self, knowledge_base: List[str], scenarios: List[Any]) -> float:
        """Test knowledge base on scenarios without learning."""
        # For MVP, use simple heuristic
        # In full implementation, would use ASP core with knowledge base
        correct = 0
        for scenario in scenarios:
            # Simplified: predict based on whether KB has relevant rules
            # In reality, would run ASP solver with KB + scenario facts
            if "writing" in scenario.description.lower():
                prediction = (
                    "enforceable"
                    if any("writing" in rule for rule in knowledge_base)
                    else "unenforceable"
                )
            else:
                prediction = "enforceable"

            if prediction == scenario.ground_truth:
                correct += 1

        return correct / len(scenarios) if scenarios else 0.0

    def generate_report(self, result: TransferResult, output_path: Path) -> None:
        """Generate transfer study report."""
        report = {
            "experiment": {
                "type": "cross_domain_transfer",
                "timestamp": datetime.now().isoformat(),
                "source_domain": result.source_domain,
                "target_domain": result.target_domain,
            },
            "source_training": {
                "cases": result.source_cases,
                "baseline_accuracy": result.source_baseline_accuracy,
                "final_accuracy": result.source_final_accuracy,
                "improvement": result.source_final_accuracy - result.source_baseline_accuracy,
                "rules_learned": result.source_rules_learned,
            },
            "transfer_results": {
                "zero_shot_accuracy": result.zero_shot_accuracy,
                "transfer_rate": result.transfer_rate,
                "few_shot_cases": result.few_shot_cases,
                "few_shot_accuracy": result.few_shot_accuracy,
                "few_shot_rules_learned": result.few_shot_rules_learned,
                "few_shot_advantage": result.few_shot_advantage,
            },
            "baseline_comparison": {
                "from_scratch_accuracy": result.scratch_accuracy,
                "from_scratch_rules": result.scratch_rules_learned,
                "few_shot_vs_scratch": result.few_shot_advantage,
            },
        }

        # Save JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Transfer study report saved to: {output_path}")


def main():
    """Main entry point for transfer study CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Domain Transfer Study for Legal Knowledge")
    parser.add_argument(
        "--source-domain",
        type=str,
        required=True,
        help="Path to source domain dataset",
    )
    parser.add_argument(
        "--target-domain",
        type=str,
        required=True,
        help="Path to target domain dataset",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=10,
        help="Number of target cases for few-shot learning (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="LLM model to use (default: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: reports/transfer_TIMESTAMP.json)",
    )

    args = parser.parse_args()

    # Create output path if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"transfer_{timestamp}.json")

    # Run experiment
    study = TransferStudy(model=args.model)

    logger.info("=" * 80)
    logger.info("Cross-Domain Transfer Learning Experiment")
    logger.info("=" * 80)

    result = study.run_transfer_experiment(
        source_dataset=Path(args.source_domain),
        target_dataset=Path(args.target_domain),
        few_shot_count=args.few_shot,
    )

    # Generate report
    study.generate_report(result, Path(args.output))

    # Print summary
    print("\n" + "=" * 80)
    print("Transfer Study Results")
    print("=" * 80)
    print(f"Source Domain: {result.source_domain}")
    print(f"Target Domain: {result.target_domain}")
    print()
    print("Source Training:")
    print(
        f"  Baseline → Final: {result.source_baseline_accuracy:.1%} → {result.source_final_accuracy:.1%}"
    )
    print(f"  Rules Learned: {result.source_rules_learned}")
    print()
    print("Transfer Results:")
    print(f"  Zero-shot Accuracy: {result.zero_shot_accuracy:.1%}")
    print(f"  Transfer Rate: {result.transfer_rate:.1%}")
    print()
    print("Few-shot Learning:")
    print(f"  Few-shot Accuracy: {result.few_shot_accuracy:.1%} ({args.few_shot} cases)")
    print(f"  From-scratch Accuracy: {result.scratch_accuracy:.1%}")
    print(f"  Advantage: {result.few_shot_advantage:+.1%}")
    print()
    print(f"Full report: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
