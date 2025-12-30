"""
Legal QA Evaluation Demo.

Demonstrates how to use the benchmark test suite to evaluate
legal question answering performance.

Issue #277: Legal Question Test Suite

Usage:
    python examples/legal_qa_evaluation_demo.py
"""

import logging

from loft.evaluation.benchmark import BenchmarkLoader
from loft.evaluation.metrics import MetricsCalculator
from loft.neural.llm_interface import OllamaInterface

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_load_benchmarks():
    """Demo 1: Loading benchmark test suites."""
    print("\n" + "=" * 70)
    print("DEMO 1: Loading Benchmark Test Suites")
    print("=" * 70 + "\n")

    # Create loader
    loader = BenchmarkLoader()

    # Get available suites
    available = loader.get_available_suites()
    print(f"Available benchmark suites: {', '.join(available)}\n")

    # Load contracts suite
    print("Loading contracts benchmark...")
    contracts = loader.load_suite("contracts")
    print(f"  Domain: {contracts.domain}")
    print(f"  Description: {contracts.description}")
    print(f"  Total questions: {contracts.question_count}")

    # Show sample questions
    print("\n  Sample questions:")
    for i, q in enumerate(contracts.questions[:3], 1):
        print(f"    {i}. {q.question}")
        print(f"       Expected: {q.expected_answer}, Difficulty: {q.difficulty}")

    # Filter by difficulty
    easy_questions = contracts.get_by_difficulty("easy")
    medium_questions = contracts.get_by_difficulty("medium")
    hard_questions = contracts.get_by_difficulty("hard")

    print("\n  By difficulty:")
    print(f"    Easy: {len(easy_questions)} questions")
    print(f"    Medium: {len(medium_questions)} questions")
    print(f"    Hard: {len(hard_questions)} questions")

    # Load all suites
    print("\nLoading all benchmark suites...")
    all_suites = loader.load_all_suites()
    total_questions = sum(s.question_count for s in all_suites.values())
    print(f"  Loaded {len(all_suites)} suites with {total_questions} total questions")

    for domain, suite in sorted(all_suites.items()):
        print(f"    - {domain}: {suite.question_count} questions")


def demo_calculate_metrics():
    """Demo 2: Calculating performance metrics."""
    print("\n" + "=" * 70)
    print("DEMO 2: Calculating Performance Metrics")
    print("=" * 70 + "\n")

    from loft.qa.schemas import Answer, QuestionResult

    # Create sample results
    print("Creating sample evaluation results...\n")
    results = [
        QuestionResult(
            question="Is a contract valid without consideration?",
            expected_answer="no",
            actual_answer=Answer(
                answer="no",
                confidence=0.92,
                explanation="Contracts require consideration",
            ),
            domain="contracts",
        ),
        QuestionResult(
            question="Does negligence require duty, breach, causation, and damages?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="yes", confidence=0.88, explanation="All four elements required"
            ),
            domain="torts",
        ),
        QuestionResult(
            question="Can adverse possession ripen into title?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="no", confidence=0.65, explanation="Incorrect answer"
            ),
            domain="property",
        ),
        QuestionResult(
            question="Is wealth a suspect classification?",
            expected_answer="no",
            actual_answer=Answer(
                answer="unknown",
                confidence=0.4,
                explanation="Insufficient information",
            ),
            domain="constitutional",
        ),
    ]

    # Calculate metrics
    calc = MetricsCalculator()
    metrics = calc.calculate(results)

    # Display metrics
    print(metrics.format_report())


def demo_run_evaluation():
    """Demo 3: Running full evaluation (requires Ollama)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Running Full Evaluation")
    print("=" * 70 + "\n")

    try:
        # Initialize LLM
        print("Initializing Ollama interface...")
        _llm = OllamaInterface(  # noqa: F841
            model="llama3.2:latest", base_url="http://localhost:11434"
        )

        # Initialize QA components (mock/minimal implementation)
        print("Initializing QA components...")

        # Note: This is a simplified demo. Full implementation would require
        # complete QA interface with ASP core and knowledge database

        # For demo purposes, we'll just show the structure
        print("\nFull evaluation structure:")
        print("  1. Initialize QA interface with:")
        print("     - LLM (Ollama)")
        print("     - Question parser")
        print("     - Reasoner with ASP core")
        print("     - Knowledge database")
        print("  2. Create EvaluationRunner")
        print("  3. Run evaluation on benchmark suite")
        print("  4. Generate performance report")

        print("\nExample command:")
        print("  runner = EvaluationRunner(qa_interface)")
        print("  results, metrics = runner.run_domain('contracts', max_questions=10)")
        print("  print(metrics.format_report())")

        print("\nNOTE: Full evaluation requires:")
        print("  - Ollama running locally (llama3.2:latest model)")
        print("  - Complete QA interface setup")
        print("  - Populated knowledge database with rules")

    except Exception as e:
        print(f"\nSkipping full evaluation: {e}")
        print(
            "This is expected if Ollama is not running or QA interface is not fully configured."
        )


def demo_benchmark_statistics():
    """Demo 4: Show benchmark dataset statistics."""
    print("\n" + "=" * 70)
    print("DEMO 4: Benchmark Dataset Statistics")
    print("=" * 70 + "\n")

    loader = BenchmarkLoader()
    all_suites = loader.load_all_suites()

    print(f"Total benchmark suites: {len(all_suites)}\n")

    total_questions = 0
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0}

    for domain, suite in sorted(all_suites.items()):
        print(f"{domain.upper()}:")
        print(f"  Questions: {suite.question_count}")

        easy = len(suite.get_by_difficulty("easy"))
        medium = len(suite.get_by_difficulty("medium"))
        hard = len(suite.get_by_difficulty("hard"))

        print("  Difficulty breakdown:")
        print(f"    Easy:   {easy}")
        print(f"    Medium: {medium}")
        print(f"    Hard:   {hard}")

        # Sample topics
        all_topics = set()
        for q in suite.questions:
            all_topics.update(q.topics)

        print(f"  Topics: {', '.join(sorted(list(all_topics))[:5])}...")
        print()

        total_questions += suite.question_count
        by_difficulty["easy"] += easy
        by_difficulty["medium"] += medium
        by_difficulty["hard"] += hard

    print("OVERALL STATISTICS:")
    print(f"  Total questions: {total_questions}")
    print("  Difficulty distribution:")
    print(
        f"    Easy:   {by_difficulty['easy']} ({by_difficulty['easy']/total_questions*100:.1f}%)"
    )
    print(
        f"    Medium: {by_difficulty['medium']} ({by_difficulty['medium']/total_questions*100:.1f}%)"
    )
    print(
        f"    Hard:   {by_difficulty['hard']} ({by_difficulty['hard']/total_questions*100:.1f}%)"
    )


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("LEGAL QA EVALUATION FRAMEWORK DEMO")
    print("=" * 70)

    try:
        demo_load_benchmarks()
        demo_calculate_metrics()
        demo_benchmark_statistics()
        demo_run_evaluation()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nFor more information, see:")
        print("  - loft/evaluation/README.md (if available)")
        print("  - tests/unit/evaluation/test_*.py (unit tests)")
        print("  - Issue #277 on GitHub")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\nDemo encountered an error: {e}")
        print("This may be expected if benchmark data files are not present.")


if __name__ == "__main__":
    main()
