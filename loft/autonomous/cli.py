"""
CLI Interface for Autonomous Test Harness.

This module provides a Click-based command-line interface for
managing autonomous test runs.

Commands:
- start: Start a new autonomous run
- resume: Resume from checkpoint
- status: Check run status
- report: Generate report from run
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from loft.autonomous.config import AutonomousRunConfig
from loft.autonomous.health import create_health_server
from loft.autonomous.logging_config import (
    create_log_summary,
    setup_autonomous_logging,
)
from loft.autonomous.notifications import create_notification_manager
from loft.autonomous.persistence import create_persistence_manager
from loft.autonomous.runner import AutonomousTestRunner


class LLMCaseProcessorAdapter:
    """
    Adapter that wraps LLMCaseProcessor for use with AutonomousTestRunner.

    The runner expects a harness with process_case(case) -> Dict,
    but LLMCaseProcessor.process_case(case, accumulated_rules) -> CaseResult.
    This adapter bridges that interface gap.
    """

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        """
        Initialize the adapter.

        Args:
            model: LLM model to use for processing
        """
        self._processor: Optional[Any] = None
        self._accumulated_rules: list = []
        self._model = model

    def initialize(self) -> None:
        """Initialize the underlying LLMCaseProcessor (lazy init)."""
        if self._processor is not None:
            return

        from loft.autonomous.llm_processor import LLMCaseProcessor

        self._processor = LLMCaseProcessor(model=self._model)
        self._processor.initialize()

    def process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a case and return results as a dictionary.

        Args:
            case: Case data dictionary

        Returns:
            Dictionary with processing results including 'correct' key
        """
        self.initialize()

        # Process using LLMCaseProcessor
        result = self._processor.process_case(case, self._accumulated_rules)

        # Update accumulated rules if any were generated
        if result.generated_rule_ids:
            self._accumulated_rules.extend(result.generated_rule_ids)

        # Convert CaseResult to dict format expected by runner
        return {
            "case_id": result.case_id,
            "correct": (
                result.prediction_correct if result.prediction_correct is not None else True
            ),
            "domain": case.get("domain", "unknown"),
            "status": result.status.value if result.status else "unknown",
            "rules_generated": result.rules_generated,
            "rules_accepted": result.rules_accepted,
            "rules_rejected": result.rules_rejected,
            "processing_time_ms": result.processing_time_ms,
            "error_message": result.error_message,
            "confidence": result.confidence,
        }

    def get_processor(self) -> Optional[Any]:
        """Get underlying LLMCaseProcessor for metrics access."""
        return self._processor

    def get_failure_patterns(self) -> Dict[str, int]:
        """Get failure patterns from the processor for meta-reasoning."""
        if self._processor is None:
            return {}
        return self._processor.get_failure_patterns()

    def clear_failure_tracking(self) -> None:
        """Clear failure tracking data after meta-reasoning cycle."""
        if self._processor is not None:
            self._processor.clear_failure_tracking()


@dataclass
class VotingRecord:
    """Record of a voting outcome.

    Attributes:
        decision: The winning result
        strategy: Voting strategy used
        vote_count: Number of votes
        confidence: Confidence in decision
    """

    decision: str
    strategy: str
    vote_count: int
    confidence: float


@dataclass
class DisagreementResolution:
    """Record of how a disagreement was resolved.

    Attributes:
        resolution_type: Type of resolution (e.g., 'defer_to_critic')
        original_responses: Count of conflicting responses
        final_decision: The resolved decision
    """

    resolution_type: str
    original_responses: int
    final_decision: str


@dataclass
class ModelPerformance:
    """Performance metrics for a single model.

    Attributes:
        model_name: Name/type of the model
        total_requests: Total number of requests
        successful_requests: Number of successful requests
        average_confidence: Average confidence score
        average_latency_ms: Average response latency
    """

    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    average_confidence: float = 0.0
    average_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


@dataclass
class EnsembleDiagnostics:
    """Diagnostics for ensemble processing.

    Collects metrics about ensemble component performance including
    voting results, disagreement resolutions, and per-model metrics.

    Attributes:
        voting_results: List of voting outcomes
        disagreement_records: List of disagreement resolutions
        model_performance: Per-model performance metrics
        consensus_rate: Fraction of tasks with unanimous agreement
        escalation_count: Number of escalations to human review
        critic_intervention_count: Number of critic interventions
        total_tasks_processed: Total number of tasks processed
        start_time: When diagnostics collection started
    """

    voting_results: List[VotingRecord] = field(default_factory=list)
    disagreement_records: List[DisagreementResolution] = field(default_factory=list)
    model_performance: Dict[str, ModelPerformance] = field(default_factory=dict)
    consensus_rate: float = 0.0
    escalation_count: int = 0
    critic_intervention_count: int = 0
    total_tasks_processed: int = 0
    start_time: float = field(default_factory=time.time)

    def record_voting_result(
        self,
        decision: str,
        strategy: str,
        vote_count: int,
        confidence: float,
        was_unanimous: bool,
    ) -> None:
        """Record a voting outcome."""
        self.voting_results.append(
            VotingRecord(
                decision=decision,
                strategy=strategy,
                vote_count=vote_count,
                confidence=confidence,
            )
        )
        self.total_tasks_processed += 1
        if was_unanimous:
            unanimous_count = sum(1 for v in self.voting_results if v.vote_count == 1) + 1
            self.consensus_rate = unanimous_count / len(self.voting_results)

    def record_disagreement(
        self,
        resolution_type: str,
        original_responses: int,
        final_decision: str,
    ) -> None:
        """Record a disagreement resolution."""
        self.disagreement_records.append(
            DisagreementResolution(
                resolution_type=resolution_type,
                original_responses=original_responses,
                final_decision=final_decision,
            )
        )
        if resolution_type == "escalate":
            self.escalation_count += 1
        if resolution_type == "defer_to_critic":
            self.critic_intervention_count += 1

    def update_model_performance(
        self,
        model_name: str,
        success: bool,
        confidence: float,
        latency_ms: float,
    ) -> None:
        """Update performance metrics for a model."""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = ModelPerformance(model_name=model_name)

        perf = self.model_performance[model_name]
        perf.total_requests += 1
        if success:
            perf.successful_requests += 1
        # Update running averages
        n = perf.total_requests
        perf.average_confidence = (perf.average_confidence * (n - 1) + confidence) / n
        perf.average_latency_ms = (perf.average_latency_ms * (n - 1) + latency_ms) / n

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostics to dictionary for checkpoint serialization."""
        return {
            "consensus_rate": self.consensus_rate,
            "escalation_count": self.escalation_count,
            "critic_intervention_count": self.critic_intervention_count,
            "total_tasks_processed": self.total_tasks_processed,
            "voting_strategy_distribution": self._get_voting_distribution(),
            "disagreement_resolutions": self._get_disagreement_distribution(),
            "model_accuracy_by_task": self._get_model_accuracy(),
            "elapsed_seconds": time.time() - self.start_time,
        }

    def _get_voting_distribution(self) -> Dict[str, int]:
        """Get distribution of voting strategies used."""
        distribution: Dict[str, int] = {}
        for vote in self.voting_results:
            distribution[vote.strategy] = distribution.get(vote.strategy, 0) + 1
        return distribution

    def _get_disagreement_distribution(self) -> Dict[str, int]:
        """Get distribution of disagreement resolution types."""
        distribution: Dict[str, int] = {}
        for record in self.disagreement_records:
            distribution[record.resolution_type] = distribution.get(record.resolution_type, 0) + 1
        return distribution

    def _get_model_accuracy(self) -> Dict[str, Dict[str, float]]:
        """Get accuracy metrics per model."""
        return {
            name: {
                "success_rate": perf.success_rate,
                "average_confidence": perf.average_confidence,
                "average_latency_ms": perf.average_latency_ms,
            }
            for name, perf in self.model_performance.items()
        }


class EnsembleCaseProcessorAdapter:
    """
    Adapter that wraps EnsembleOrchestrator for use with AutonomousTestRunner.

    This adapter integrates Phase 6 ensemble components (LogicGeneratorLLM,
    CriticLLM, TranslatorLLM, MetaReasonerLLM) into the autonomous testing
    infrastructure, providing diagnostics collection and ensemble-aware
    case processing.

    Part of issue #207: Integrate ensemble diagnostics into autonomous testing.
    """

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        """
        Initialize the adapter.

        Args:
            model: LLM model to use for ensemble processing
        """
        self._orchestrator: Optional[Any] = None
        self._diagnostics = EnsembleDiagnostics()
        self._accumulated_rules: List[str] = []
        self._model = model
        self._llm_interface: Optional[Any] = None

    def initialize(self) -> None:
        """Initialize the underlying EnsembleOrchestrator (lazy init)."""
        if self._orchestrator is not None:
            return

        from loft.neural.ensemble import EnsembleOrchestrator, OrchestratorConfig
        from loft.neural.llm_interface import LLMInterface

        self._llm_interface = LLMInterface(model=self._model)
        self._orchestrator = EnsembleOrchestrator(
            config=OrchestratorConfig(),
            llm_interface=self._llm_interface,
        )
        self._diagnostics = EnsembleDiagnostics()

    def process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a case using the ensemble and return results.

        Args:
            case: Case data dictionary

        Returns:
            Dictionary with processing results including 'correct' key
        """
        self.initialize()

        from loft.neural.ensemble import TaskType

        start_time = time.time()

        # Extract relevant info from case
        principle = case.get("facts", case.get("principle", str(case)))
        domain = case.get("domain", "legal")
        case_id = case.get("id", case.get("case_id", "unknown"))

        try:
            # Route through full pipeline: generate -> critique -> refine
            result = self._orchestrator.route_task(
                task_type=TaskType.FULL_PIPELINE,
                input_data={
                    "principle": principle,
                    "domain": domain,
                    "predicates": case.get("predicates", []),
                },
                context={"case_id": case_id},
            )

            processing_time_ms = (time.time() - start_time) * 1000

            # Record diagnostics from orchestration
            self._record_diagnostics_from_result(result, processing_time_ms)

            # Extract generated rules
            final_result = result.final_result
            generated_rule = None
            if hasattr(final_result, "asp_rule"):
                generated_rule = final_result.asp_rule
                self._accumulated_rules.append(generated_rule)

            # Determine if processing was successful
            confidence = 0.0
            if hasattr(final_result, "confidence"):
                confidence = final_result.confidence
            elif result.voting_result:
                confidence = result.voting_result.confidence

            return {
                "case_id": case_id,
                "correct": confidence >= 0.6,
                "domain": domain,
                "status": "success",
                "rules_generated": 1 if generated_rule else 0,
                "rules_accepted": 1 if confidence >= 0.6 else 0,
                "rules_rejected": (0 if confidence >= 0.6 else (1 if generated_rule else 0)),
                "processing_time_ms": processing_time_ms,
                "error_message": None,
                "confidence": confidence,
                "ensemble_metadata": {
                    "models_used": [r.model_type for r in result.model_responses],
                    "voting_used": result.voting_result is not None,
                    "disagreements": len(result.disagreements),
                },
            }

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return {
                "case_id": case_id,
                "correct": False,
                "domain": domain,
                "status": "failure",
                "rules_generated": 0,
                "rules_accepted": 0,
                "rules_rejected": 0,
                "processing_time_ms": processing_time_ms,
                "error_message": str(e),
                "confidence": 0.0,
                "ensemble_metadata": {},
            }

    def _record_diagnostics_from_result(
        self,
        result: Any,
        processing_time_ms: float,
    ) -> None:
        """Record diagnostics from an orchestration result."""
        # Record model performance from responses
        for response in result.model_responses:
            self._diagnostics.update_model_performance(
                model_name=response.model_type,
                success=True,
                confidence=response.confidence,
                latency_ms=response.latency_ms,
            )

        # Record voting result if present
        if result.voting_result:
            voting = result.voting_result
            self._diagnostics.record_voting_result(
                decision=str(voting.decision)[:100],
                strategy=voting.strategy_used.value,
                vote_count=len(voting.participating_models),
                confidence=voting.confidence,
                was_unanimous=len(voting.dissenting_models) == 0,
            )

        # Record disagreements
        for disagreement in result.disagreements:
            self._diagnostics.record_disagreement(
                resolution_type=disagreement.resolution_strategy.value,
                original_responses=len(disagreement.conflicting_responses),
                final_decision=str(disagreement.final_decision)[:100],
            )

    def get_orchestrator(self) -> Optional[Any]:
        """Get underlying EnsembleOrchestrator for direct access."""
        return self._orchestrator

    def get_diagnostics(self) -> EnsembleDiagnostics:
        """Get ensemble diagnostics."""
        return self._diagnostics

    def get_diagnostics_dict(self) -> Dict[str, Any]:
        """Get diagnostics as a dictionary for checkpoint serialization."""
        return self._diagnostics.to_dict()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the orchestrator."""
        if self._orchestrator is None:
            return {}
        metrics = self._orchestrator.get_performance_metrics()
        return {
            name: {
                "total_requests": m.total_requests,
                "success_rate": m.success_rate,
                "average_latency_ms": m.average_latency_ms,
                "average_confidence": m.average_confidence,
            }
            for name, m in metrics.items()
        }

    def get_accumulated_rules(self) -> List[str]:
        """Get list of rules generated across all cases."""
        return self._accumulated_rules.copy()


def setup_logging(
    log_level: str,
    log_file: Optional[Path] = None,
    enable_clingo_filter: bool = True,
    progress_interval_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Configure logging with improved readability.

    Uses the enhanced logging configuration that:
    - Filters Clingo "info" messages that flood logs
    - Provides summarized error logging for LLM failures
    - Standardizes log format across all modules
    - Adds periodic progress indicators

    Args:
        log_level: Logging level string
        log_file: Optional log file path
        enable_clingo_filter: Whether to filter Clingo info messages
        progress_interval_seconds: Seconds between progress logs

    Returns:
        Dictionary with logging components for customization
    """
    return setup_autonomous_logging(
        log_level=log_level,
        log_file=log_file,
        enable_clingo_filter=enable_clingo_filter,
        progress_interval_seconds=progress_interval_seconds,
    )


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """LOFT Autonomous Test Harness.

    Run long-duration autonomous test experiments with meta-reasoning
    integration, checkpointing, and Slack notifications.
    """
    pass


@cli.command()
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="Dataset directory or file path (can specify multiple)",
)
@click.option(
    "--source",
    "-s",
    type=click.Choice(["local", "courtlistener"]),
    default="local",
    help="Data source: 'local' for files, 'courtlistener' for API",
)
@click.option(
    "--search-queries",
    multiple=True,
    help="Search queries for CourtListener (e.g., 'statute of frauds')",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="YAML configuration file",
)
@click.option(
    "--duration",
    "-t",
    default="4h",
    help="Maximum duration (e.g., 4h, 30m, 1.5h)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/autonomous_runs",
    help="Output directory for run data",
)
@click.option(
    "--run-id",
    help="Custom run ID (auto-generated if not provided)",
)
@click.option(
    "--checkpoint-interval",
    default=15,
    type=int,
    help="Minutes between checkpoints",
)
@click.option(
    "--max-cases",
    default=0,
    type=int,
    help="Maximum cases to process (0 = unlimited)",
)
@click.option(
    "--model",
    default="claude-3-5-haiku-20241022",
    help="LLM model to use",
)
@click.option(
    "--slack-webhook",
    envvar="SLACK_WEBHOOK_URL",
    help="Slack webhook URL for notifications",
)
@click.option(
    "--health-port",
    default=8080,
    type=int,
    help="Port for health endpoint",
)
@click.option(
    "--no-health",
    is_flag=True,
    help="Disable health endpoint",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.option(
    "--no-clingo-filter",
    is_flag=True,
    help="Disable filtering of Clingo info messages",
)
@click.option(
    "--progress-interval",
    default=300,
    type=int,
    help="Seconds between progress summary logs (0 to disable)",
)
@click.option(
    "--max-cost",
    type=float,
    default=None,
    help="Maximum LLM API cost in USD before auto-stopping (e.g., 10.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum total LLM tokens before auto-stopping (e.g., 1000000)",
)
@click.option(
    "--budget-warning-threshold",
    type=float,
    default=0.8,
    help="Fraction of budget at which to log warnings (default: 0.8 = 80%%)",
)
@click.option(
    "--enable-llm",
    is_flag=True,
    help="Enable LLM-powered case processing for ASP generation (requires ANTHROPIC_API_KEY)",
)
@click.option(
    "--skip-api-check",
    is_flag=True,
    help="Skip ANTHROPIC_API_KEY validation (for testing)",
)
@click.option(
    "--enable-ensemble",
    is_flag=True,
    help="Enable ensemble orchestrator for multi-LLM processing (requires --enable-llm)",
)
def start(
    dataset: tuple,
    source: str,
    search_queries: tuple,
    config: Optional[str],
    duration: str,
    output: str,
    run_id: Optional[str],
    checkpoint_interval: int,
    max_cases: int,
    model: str,
    slack_webhook: Optional[str],
    health_port: int,
    no_health: bool,
    log_level: str,
    no_clingo_filter: bool,
    progress_interval: int,
    max_cost: Optional[float],
    max_tokens: Optional[int],
    budget_warning_threshold: float,
    enable_llm: bool,
    skip_api_check: bool,
    enable_ensemble: bool,
) -> None:
    """Start a new autonomous test run.

    Examples:

        # Basic run with local dataset
        loft-autonomous start --dataset datasets/contracts/ --duration 4h

        # Run with CourtListener API (real legal cases)
        loft-autonomous start \\
            --source courtlistener \\
            --search-queries "statute of frauds" \\
            --search-queries "adverse possession" \\
            --duration 2h \\
            --max-cases 100

        # Custom configuration with multiple datasets
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --dataset datasets/torts/ \\
            --config config/autonomous.yaml \\
            --duration 2h \\
            --max-cases 500

        # With Slack notifications
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --slack-webhook https://hooks.slack.com/... \\
            --duration 4h

        # With verbose logging (no Clingo filtering)
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --no-clingo-filter \\
            --progress-interval 60 \\
            --duration 1h

        # With budget limits (auto-stop when exceeded)
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --max-cost 10.0 \\
            --max-tokens 1000000 \\
            --budget-warning-threshold 0.8 \\
            --duration 4h

        # With LLM-powered ASP generation (issue #185)
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --enable-llm \\
            --model claude-3-5-haiku-20241022 \\
            --duration 30m \\
            --max-cases 50

        # With ensemble orchestrator for multi-LLM processing (issue #207)
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --enable-llm \\
            --enable-ensemble \\
            --model claude-3-5-haiku-20241022 \\
            --duration 30m \\
            --max-cases 50
    """
    # Set up logging file path
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "run.log" if run_id else None

    # Set up logging with improved readability
    logging_components = setup_logging(
        log_level=log_level,
        log_file=log_file,
        enable_clingo_filter=not no_clingo_filter,
        progress_interval_seconds=float(progress_interval),
    )
    logger = logging.getLogger(__name__)
    progress_indicator = logging_components.get("progress_indicator")
    clingo_filter = logging_components.get("clingo_filter")
    error_summarizer = logging_components.get("error_summarizer")

    # Validate arguments based on source
    if source == "local" and not dataset:
        raise click.UsageError("--dataset is required when using --source local")

    # Pre-flight validation for LLM mode (issue #185)
    if enable_llm and not skip_api_check:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise click.UsageError(
                "--enable-llm requires ANTHROPIC_API_KEY environment variable.\n"
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'\n"
                "Or use --skip-api-check to skip this validation."
            )
        logger.info("ANTHROPIC_API_KEY validated")

    # Validate ensemble mode requires LLM mode (issue #207)
    if enable_ensemble and not enable_llm:
        raise click.UsageError(
            "--enable-ensemble requires --enable-llm.\n"
            "Ensemble mode uses multiple LLMs and requires LLM processing to be enabled."
        )

    duration_hours = _parse_duration(duration)

    if config:
        run_config = AutonomousRunConfig.from_yaml(Path(config))
    else:
        run_config = AutonomousRunConfig()

    run_config.max_duration_hours = duration_hours
    run_config.checkpoint_interval_minutes = checkpoint_interval
    run_config.max_cases = max_cases
    run_config.llm_model = model
    run_config.output_dir = output
    run_config.dataset_paths = list(dataset) if dataset else []

    if slack_webhook:
        run_config.notification.slack_webhook_url = slack_webhook

    run_config.health.enabled = not no_health
    run_config.health.port = health_port

    # Set up data source adapter based on source type
    data_source_adapter = None
    if source == "courtlistener":
        from loft.autonomous.data_sources import create_courtlistener_adapter

        queries = list(search_queries) if search_queries else None
        data_source_adapter = create_courtlistener_adapter(
            search_queries=queries,
            max_cases_per_query=max(50, max_cases // 10) if max_cases > 0 else 50,
        )
        logger.info(f"Using CourtListener API with queries: {queries or 'default'}")
    else:
        logger.info(f"Using local datasets: {dataset}")

    logger.info(f"Starting autonomous run with source: {source}")
    logger.info(f"Max duration: {duration_hours} hours")
    logger.info(f"Output directory: {output}")

    # Log budget limits if configured
    if max_cost is not None:
        logger.info(f"Budget limit: ${max_cost:.2f} USD")
    if max_tokens is not None:
        logger.info(f"Token limit: {max_tokens:,} tokens")

    health_server = None
    notification_manager = None
    metrics_tracker = None
    budget_exceeded_flag = {"exceeded": False}  # Mutable to allow callback to set

    try:
        # Always create LLM metrics tracker for monitoring (issue #165)
        # Even without budget limits, this provides visibility into LLM usage
        from loft.autonomous.llm_metrics import (
            LLMMetricsTracker,
            set_global_metrics_tracker,
        )

        def on_budget_warning(limit_type: str, current: float, limit: float) -> None:
            pct = (current / limit) * 100 if limit > 0 else 0
            logger.warning(
                f"Budget warning: {limit_type} at {pct:.1f}% ({current:.2f}/{limit:.2f})"
            )

        def on_budget_exceeded(limit_type: str, current: float, limit: float) -> None:
            logger.error(
                f"Budget exceeded: {limit_type} limit reached ({current:.2f}/{limit:.2f}). "
                "Run will stop after current operation."
            )
            budget_exceeded_flag["exceeded"] = True

        metrics_tracker = LLMMetricsTracker(
            model=model,
            max_cost_usd=max_cost,
            max_tokens=max_tokens,
            warning_threshold=budget_warning_threshold,
            log_interval_seconds=float(progress_interval),
            on_budget_warning=on_budget_warning if (max_cost or max_tokens) else None,
            on_budget_exceeded=on_budget_exceeded if (max_cost or max_tokens) else None,
        )

        # Set as global tracker so LLMInterface can use it automatically
        set_global_metrics_tracker(metrics_tracker)
        logger.info(
            "LLM metrics tracking enabled"
            + (f" (cost limit: ${max_cost:.2f})" if max_cost else "")
            + (f" (token limit: {max_tokens:,})" if max_tokens else "")
        )

        if run_config.health.enabled:
            health_server = create_health_server(run_config.health)
            if metrics_tracker is not None:
                health_server.metrics_tracker = metrics_tracker
            health_server.start()

        notification_manager = create_notification_manager(run_config.notification)

        runner = AutonomousTestRunner(run_config, output_dir=Path(output))

        # Set data source adapter if using CourtListener
        if data_source_adapter:
            runner.set_data_source(data_source_adapter)

        # Set up LLM case processor adapter if enabled (issue #185, #207)
        llm_processor_adapter = None
        ensemble_adapter = None
        if enable_ensemble:
            # Use EnsembleCaseProcessorAdapter for multi-LLM processing
            logger.info(f"Enabling ensemble orchestrator with model: {model} (issue #207)")
            ensemble_adapter = EnsembleCaseProcessorAdapter(model=model)
            runner.set_batch_harness(ensemble_adapter)
            logger.info("EnsembleCaseProcessorAdapter configured as batch harness")
        elif enable_llm:
            logger.info(f"Enabling LLM-powered case processing with model: {model}")
            llm_processor_adapter = LLMCaseProcessorAdapter(model=model)
            runner.set_batch_harness(llm_processor_adapter)
            logger.info("LLMCaseProcessorAdapter configured as batch harness")
        else:
            logger.warning(
                "LLM processing disabled (use --enable-llm for ASP generation). "
                "Running in stub mode - cases will be marked as successful without processing."
            )

        def on_progress(progress):
            if health_server:
                health_server.update_state(runner.state)

        runner.set_callbacks(on_progress=on_progress)

        notification_manager.notify_started(
            run_id=run_id or "auto",
            config=run_config.to_dict(),
        )

        result = runner.start(dataset_paths=list(dataset), run_id=run_id)

        notification_manager.notify_completion(result)

        # Log final summary
        summary = create_log_summary(
            clingo_filter=clingo_filter,
            error_summarizer=error_summarizer,
            progress_indicator=progress_indicator,
        )
        logger.info("\n" + summary)

        # Log LLM metrics summary if tracker was used (issue #165)
        if metrics_tracker is not None:
            metrics_summary = metrics_tracker.get_metrics_summary()
            logger.info(
                f"\nLLM Metrics Summary:\n"
                f"  Total calls: {metrics_summary['total_calls']}\n"
                f"  Total tokens: {metrics_summary['total_tokens']:,}\n"
                f"  Total cost: ${metrics_summary['total_cost_usd']:.4f}\n"
                f"  Success rate: {metrics_summary['success_rate']:.1%}\n"
                f"  Avg latency: {metrics_summary['avg_latency_seconds']:.2f}s"
            )

        # Log ensemble diagnostics if ensemble adapter was used (issue #207)
        if ensemble_adapter is not None:
            diagnostics = ensemble_adapter.get_diagnostics_dict()
            logger.info(
                f"\nEnsemble Diagnostics (issue #207):\n"
                f"  Tasks processed: {diagnostics['total_tasks_processed']}\n"
                f"  Consensus rate: {diagnostics['consensus_rate']:.1%}\n"
                f"  Escalation count: {diagnostics['escalation_count']}\n"
                f"  Critic interventions: {diagnostics['critic_intervention_count']}\n"
                f"  Elapsed time: {diagnostics['elapsed_seconds']:.1f}s"
            )
            if diagnostics["voting_strategy_distribution"]:
                logger.info(
                    f"  Voting strategies used: {diagnostics['voting_strategy_distribution']}"
                )
            if diagnostics["disagreement_resolutions"]:
                logger.info(
                    f"  Disagreement resolutions: {diagnostics['disagreement_resolutions']}"
                )

        if result.was_successful:
            click.echo("\nRun completed successfully!")
            click.echo(f"Run ID: {result.run_id}")
            click.echo(f"Duration: {result.duration_hours:.2f} hours")
            click.echo(f"Final accuracy: {result.final_metrics.overall_accuracy:.2%}")
            click.echo(f"Report: {Path(output) / result.run_id / 'reports' / 'final_report.md'}")
        else:
            click.echo(f"\nRun ended with status: {result.status.value}")
            if result.error_message:
                click.echo(f"Error: {result.error_message}")
            if result.checkpoint_path:
                click.echo(f"Checkpoint: {result.checkpoint_path}")

    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.exception("Run failed")
        if notification_manager:
            notification_manager.notify_error(
                run_id=run_id or "unknown",
                error=e,
                context={},
            )
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    finally:
        if health_server:
            health_server.stop()


@cli.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint file to resume from",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
def resume(checkpoint: str, log_level: str) -> None:
    """Resume an autonomous run from checkpoint.

    Examples:

        # Resume from latest checkpoint
        loft-autonomous resume --checkpoint data/autonomous_runs/run_001/checkpoints/latest.json

        # Resume from specific checkpoint
        loft-autonomous resume --checkpoint data/autonomous_runs/run_001/checkpoints/checkpoint_0005.json
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    checkpoint_path = Path(checkpoint)
    run_dir = checkpoint_path.parent.parent

    config_path = run_dir / "config.json"
    if not config_path.exists():
        click.echo(f"Config not found at {config_path}", err=True)
        sys.exit(1)

    import json

    with open(config_path) as f:
        config_dict = json.load(f)

    run_config = AutonomousRunConfig(**config_dict)

    logger.info(f"Resuming from checkpoint: {checkpoint}")

    health_server = None

    try:
        if run_config.health.enabled:
            health_server = create_health_server(run_config.health)
            health_server.start()

        runner = AutonomousTestRunner(run_config)

        def on_progress(progress):
            if health_server:
                health_server.update_state(runner.state)

        runner.set_callbacks(on_progress=on_progress)

        result = runner.resume(checkpoint_path)

        if result.was_successful:
            click.echo("\nRun completed successfully!")
            click.echo(f"Run ID: {result.run_id}")
            click.echo(f"Duration: {result.duration_hours:.2f} hours")
            click.echo(f"Final accuracy: {result.final_metrics.overall_accuracy:.2%}")
        else:
            click.echo(f"\nRun ended with status: {result.status.value}")
            if result.checkpoint_path:
                click.echo(f"Checkpoint: {result.checkpoint_path}")

    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.exception("Resume failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    finally:
        if health_server:
            health_server.stop()


@cli.command()
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="Run ID to check status",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/autonomous_runs",
    help="Output directory containing runs",
)
def status(run_id: str, output: str) -> None:
    """Check status of an autonomous run.

    Examples:

        loft-autonomous status --run-id run_20240101_120000_abc123
    """
    run_dir = Path(output) / run_id

    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}", err=True)
        sys.exit(1)

    persistence = create_persistence_manager(Path(output), run_id)

    state = persistence.load_state()
    if state:
        click.echo(f"Run ID: {state.run_id}")
        click.echo(f"Status: {state.status.value}")

        progress = state.progress
        click.echo("\nProgress:")
        click.echo(f"  Cases processed: {progress.cases_processed}/{progress.total_cases}")
        click.echo(f"  Accuracy: {progress.current_accuracy:.2%}")
        click.echo(f"  Elapsed: {progress.elapsed_seconds / 3600:.2f} hours")
        click.echo(f"  Improvement cycles: {progress.total_cycles}")

        if state.last_updated:
            click.echo(f"\nLast updated: {state.last_updated.isoformat()}")

    result = persistence.load_result()
    if result:
        click.echo("\nFinal Result:")
        click.echo(f"  Status: {result.status.value}")
        click.echo(f"  Duration: {result.duration_hours:.2f} hours")
        click.echo(f"  Accuracy: {result.final_metrics.overall_accuracy:.2%}")

    checkpoints = persistence.list_checkpoints()
    if checkpoints:
        click.echo(f"\nCheckpoints: {len(checkpoints)}")
        click.echo(f"  Latest: {checkpoints[-1].name}")


@cli.command()
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="Run ID to generate report for",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/autonomous_runs",
    help="Output directory containing runs",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "markdown", "both"]),
    default="both",
    help="Report format",
)
def report(run_id: str, output: str, format: str) -> None:
    """Generate report from a completed run.

    Examples:

        loft-autonomous report --run-id run_20240101_120000_abc123 --format markdown
    """
    run_dir = Path(output) / run_id

    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}", err=True)
        sys.exit(1)

    persistence = create_persistence_manager(Path(output), run_id)
    result = persistence.load_result()

    if not result:
        click.echo("No result found for this run", err=True)
        sys.exit(1)

    reports_dir = run_dir / "reports"

    if format in ["json", "both"]:
        json_path = reports_dir / "final_report.json"
        click.echo(f"JSON report: {json_path}")

    if format in ["markdown", "both"]:
        md_path = reports_dir / "final_report.md"
        click.echo(f"Markdown report: {md_path}")

        if md_path.exists():
            click.echo("\n" + "=" * 60)
            with open(md_path) as f:
                click.echo(f.read())


@cli.command()
def list_runs() -> None:
    """List all autonomous runs.

    Examples:

        loft-autonomous list-runs
    """
    output_dir = Path("data/autonomous_runs")

    if not output_dir.exists():
        click.echo("No runs found")
        return

    runs = sorted(output_dir.iterdir())
    if not runs:
        click.echo("No runs found")
        return

    click.echo(f"Found {len(runs)} run(s):\n")

    for run_dir in runs:
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        state_file = run_dir / "state.json"

        status_str = "unknown"
        if state_file.exists():
            import json

            with open(state_file) as f:
                state_data = json.load(f)
                status_str = state_data.get("status", "unknown")

        click.echo(f"  {run_id}: {status_str}")


def _parse_duration(duration: str) -> float:
    """Parse duration string to hours.

    Args:
        duration: Duration string (e.g., "4h", "30m", "1.5h")

    Returns:
        Duration in hours
    """
    duration = duration.strip().lower()

    if duration.endswith("h"):
        return float(duration[:-1])
    elif duration.endswith("m"):
        return float(duration[:-1]) / 60
    elif duration.endswith("s"):
        return float(duration[:-1]) / 3600
    else:
        return float(duration)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
