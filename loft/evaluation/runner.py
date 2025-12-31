"""
Evaluation runner for legal QA benchmarks.

Orchestrates running benchmark evaluations, collecting results,
and generating performance reports.

Issue #277: Legal Question Test Suite
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loft.evaluation.benchmark import BenchmarkLoader, BenchmarkSuite
from loft.evaluation.metrics import MetricsCalculator, PerformanceMetrics
from loft.qa.interface import LegalQAInterface
from loft.qa.schemas import QuestionResult

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runs benchmark evaluations and generates reports.

    Coordinates loading benchmarks, running QA system, calculating
    metrics, and saving results.
    """

    def __init__(
        self,
        qa_interface: LegalQAInterface,
        benchmark_loader: Optional[BenchmarkLoader] = None,
    ):
        """
        Initialize evaluation runner.

        Args:
            qa_interface: Legal QA interface to evaluate
            benchmark_loader: Benchmark loader (creates default if None)
        """
        self.qa = qa_interface
        self.loader = benchmark_loader or BenchmarkLoader()
        self.metrics_calc = MetricsCalculator()

    def run_suite(
        self,
        suite: BenchmarkSuite,
        max_questions: Optional[int] = None,
        verbose: bool = False,
    ) -> tuple[List[QuestionResult], PerformanceMetrics]:
        """
        Run evaluation on a single benchmark suite.

        Args:
            suite: BenchmarkSuite to evaluate
            max_questions: Limit number of questions (for testing)
            verbose: Print progress during evaluation

        Returns:
            Tuple of (results, metrics)
        """
        logger.info(
            f"Running evaluation on {suite.domain} ({suite.question_count} questions)"
        )

        questions_to_run = suite.questions
        if max_questions:
            questions_to_run = questions_to_run[:max_questions]

        results = []
        difficulties = {}

        for i, bench_q in enumerate(questions_to_run, 1):
            if verbose:
                print(f"[{i}/{len(questions_to_run)}] {bench_q.question[:60]}...")

            try:
                # Ask the question
                answer = self.qa.ask(bench_q.question, domain=bench_q.domain)

                # Create result
                result = QuestionResult(
                    question=bench_q.question,
                    expected_answer=bench_q.expected_answer,
                    actual_answer=answer,
                    domain=bench_q.domain,
                )

                results.append(result)
                difficulties[bench_q.id] = bench_q.difficulty

                if verbose:
                    status = "✓" if result.correct else "✗"
                    print(
                        f"  {status} Expected: {bench_q.expected_answer}, Got: {answer.answer}"
                    )

            except Exception as e:
                logger.error(f"Error evaluating question {bench_q.id}: {e}")
                if verbose:
                    print(f"  ✗ Error: {e}")

        # Calculate metrics
        metrics = self.metrics_calc.calculate(results, difficulties)

        logger.info(
            f"Completed {suite.domain}: {metrics.correct}/{metrics.total_questions} "
            f"correct ({metrics.accuracy:.1%})"
        )

        return results, metrics

    def run_all_suites(
        self,
        max_questions_per_suite: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, tuple[List[QuestionResult], PerformanceMetrics]]:
        """
        Run evaluation on all available benchmark suites.

        Args:
            max_questions_per_suite: Limit questions per suite
            verbose: Print progress

        Returns:
            Dictionary mapping domain to (results, metrics) tuples
        """
        suites = self.loader.load_all_suites()

        if not suites:
            logger.warning("No benchmark suites found")
            return {}

        all_results = {}

        for domain, suite in suites.items():
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {domain.upper()}")
            print(f"{'=' * 60}\n")

            results, metrics = self.run_suite(
                suite, max_questions=max_questions_per_suite, verbose=verbose
            )
            all_results[domain] = (results, metrics)

            print(f"\n{metrics.format_report()}")

        return all_results

    def run_domain(
        self,
        domain: str,
        max_questions: Optional[int] = None,
        verbose: bool = False,
    ) -> tuple[List[QuestionResult], PerformanceMetrics]:
        """
        Run evaluation on a specific domain.

        Args:
            domain: Domain name (e.g., "contracts", "torts")
            max_questions: Limit number of questions
            verbose: Print progress

        Returns:
            Tuple of (results, metrics)
        """
        suite = self.loader.load_suite(domain)
        return self.run_suite(suite, max_questions=max_questions, verbose=verbose)

    def save_results(
        self,
        results: List[QuestionResult],
        metrics: PerformanceMetrics,
        output_file: Path,
    ):
        """
        Save evaluation results to JSON file.

        Args:
            results: List of question results
            metrics: Performance metrics
            output_file: Path to output JSON file
        """
        output_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics.to_dict(),
            "results": [
                {
                    "question": r.question,
                    "expected": r.expected_answer,
                    "actual": r.actual_answer.answer,
                    "correct": r.correct,
                    "confidence": r.actual_answer.confidence,
                    "domain": r.domain,
                }
                for r in results
            ],
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved results to {output_file}")

    def generate_baseline_report(
        self,
        output_dir: Path,
        max_questions_per_suite: Optional[int] = None,
    ) -> Path:
        """
        Generate baseline performance report for all suites.

        Args:
            output_dir: Directory to save reports
            max_questions_per_suite: Limit questions per suite

        Returns:
            Path to generated report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"baseline_report_{timestamp}.txt"

        # Run all evaluations
        all_results = self.run_all_suites(
            max_questions_per_suite=max_questions_per_suite, verbose=True
        )

        # Generate comprehensive report
        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LEGAL QA BASELINE PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}\n")
            f.write(f"Total Suites: {len(all_results)}\n\n")

            # Overall summary
            total_questions = sum(m.total_questions for _, m in all_results.values())
            total_correct = sum(m.correct for _, m in all_results.values())
            overall_accuracy = (
                total_correct / total_questions if total_questions > 0 else 0
            )

            f.write("OVERALL SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Questions:  {total_questions}\n")
            f.write(f"Total Correct:    {total_correct}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.1%}\n\n")

            # Per-domain reports
            for domain, (results, metrics) in sorted(all_results.items()):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"DOMAIN: {domain.upper()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(metrics.format_report())
                f.write("\n")

                # Save individual domain results
                domain_file = output_dir / f"{domain}_results_{timestamp}.json"
                self.save_results(results, metrics, domain_file)

        logger.info(f"Generated baseline report: {report_file}")
        return report_file
