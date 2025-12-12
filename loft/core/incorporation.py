"""
Rule incorporation engine with safety mechanisms.

Safely incorporates LLM-generated rules into the symbolic core with:
- Stratification-based policy enforcement
- Pre-incorporation validation
- Automatic rollback on regression
- Modification tracking and limits
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from loft.neural.rule_schemas import GeneratedRule
from loft.symbolic.stratification import (
    ModificationPolicy,
    StratificationLevel,
)
from loft.validation.validation_schemas import ValidationReport


@dataclass
class IncorporationResult:
    """Result of attempting to incorporate a rule."""

    status: str  # "success", "rejected", "blocked", "deferred", "error"
    reason: str = ""
    requires_human_review: bool = False
    snapshot_id: Optional[str] = None
    modification_number: int = 0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    regression_failures: List[str] = None
    regression_detected: bool = False

    def __post_init__(self):
        if self.regression_failures is None:
            self.regression_failures = []

    def is_success(self) -> bool:
        """Check if incorporation was successful."""
        return self.status == "success"

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.status == "success":
            improvement = self.accuracy_after - self.accuracy_before
            return (
                f"✅ SUCCESS: Rule incorporated (modification #{self.modification_number})\n"
                f"   Accuracy: {self.accuracy_before:.1%} → {self.accuracy_after:.1%} "
                f"({improvement:+.1%})"
            )
        elif self.status == "rejected":
            return f"❌ REJECTED: {self.reason}"
        elif self.status == "blocked":
            return f"🚫 BLOCKED: {self.reason}"
        elif self.status == "deferred":
            return f"⏸️  DEFERRED: {self.reason}"
        else:
            return f"⚠️  ERROR: {self.reason}"


@dataclass
class RollbackEvent:
    """Record of a rollback event."""

    timestamp: datetime
    reason: str
    snapshot_id: str  # Rolled back to this snapshot
    failed_rule_text: str
    regression_details: Optional[Dict[str, Any]] = None


class SimpleVersionControl:
    """
    Simplified version control for ASP core state.

    In production, would use git-based version control or database snapshots.
    """

    def __init__(self):
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.current_snapshot_id: Optional[str] = None

    def create_snapshot(self, state: Dict[str, Any], message: str = "") -> str:
        """
        Create a snapshot of current state.

        Args:
            state: Current state to snapshot
            message: Description of snapshot

        Returns:
            Snapshot ID
        """
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.snapshots[snapshot_id] = {
            "state": state.copy(),
            "message": message,
            "timestamp": datetime.now(),
        }
        self.current_snapshot_id = snapshot_id
        logger.debug(f"Created snapshot {snapshot_id}: {message}")
        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get snapshot by ID."""
        snapshot = self.snapshots.get(snapshot_id)
        return snapshot["state"] if snapshot else None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all snapshots."""
        return [
            {
                "id": sid,
                "message": snap["message"],
                "timestamp": snap["timestamp"],
            }
            for sid, snap in self.snapshots.items()
        ]


class SimpleASPCore:
    """
    Simplified ASP core for demonstration.

    In production, would integrate with actual clingo-based ASP system.
    """

    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.rule_count = 0

    def add_rule(
        self,
        rule_text: str,
        stratification_level: StratificationLevel,
        confidence: float,
        metadata: Dict[str, Any],
    ):
        """Add a rule to the core."""
        self.rules.append(
            {
                "id": f"rule_{self.rule_count}",
                "rule_text": rule_text,
                "stratification_level": stratification_level.value,
                "confidence": confidence,
                "metadata": metadata,
                "added_at": datetime.now(),
            }
        )
        self.rule_count += 1
        logger.info(f"Added rule to {stratification_level.value} layer: {rule_text[:50]}...")

    def get_state(self) -> Dict[str, Any]:
        """Get current state for snapshotting."""
        return {"rules": self.rules.copy(), "rule_count": self.rule_count}

    def restore_state(self, state: Dict[str, Any]):
        """Restore from snapshot state."""
        self.rules = state["rules"].copy()
        self.rule_count = state["rule_count"]
        logger.info("Restored ASP core state from snapshot")

    def get_rules_by_layer(self, layer: StratificationLevel) -> List[Dict[str, Any]]:
        """Get all rules for a specific layer."""
        return [r for r in self.rules if r["stratification_level"] == layer.value]


class SimpleTestSuite:
    """
    Simplified test suite for demonstration.

    In production, would run actual test cases against ASP core.
    """

    def __init__(self, initial_accuracy: float = 0.85):
        self.initial_accuracy = initial_accuracy
        self.current_accuracy = initial_accuracy
        self.test_count = 0

    def run_all(self) -> Dict[str, Any]:
        """
        Run all tests.

        Returns:
            Test results with failures list
        """
        self.test_count += 1
        # Simulate: Always pass for testing (in production, would run real tests)
        import random

        random.seed(42)  # Fixed seed for deterministic testing

        # Tests pass
        self.current_accuracy = min(1.0, self.current_accuracy + random.uniform(0, 0.01))
        return {"passed": True, "failures": [], "accuracy": self.current_accuracy}

    def measure_accuracy(self) -> float:
        """Get current accuracy."""
        return self.current_accuracy


class RuleIncorporationEngine:
    """
    Safely incorporate LLM-generated rules into symbolic core.

    Implements stratified modification with safety checks, rollback,
    and regression testing.
    """

    def __init__(
        self,
        asp_core: Optional[SimpleASPCore] = None,
        version_control: Optional[SimpleVersionControl] = None,
        test_suite: Optional[SimpleTestSuite] = None,
        policies: Optional[Dict[StratificationLevel, ModificationPolicy]] = None,
        regression_threshold: float = 0.02,  # 2% accuracy drop triggers rollback
    ):
        """
        Initialize incorporation engine.

        Args:
            asp_core: ASP core to modify (uses SimpleASPCore if None)
            version_control: Version control system (uses SimpleVersionControl if None)
            test_suite: Test suite for regression testing (uses SimpleTestSuite if None)
            policies: Modification policies (uses defaults if None)
            regression_threshold: Maximum allowed accuracy drop (default 0.02 = 2%)
        """
        self.asp_core = asp_core or SimpleASPCore()
        self.version_control = version_control or SimpleVersionControl()
        self.test_suite = test_suite or SimpleTestSuite()
        self.regression_threshold = regression_threshold

        if policies is None:
            from loft.symbolic.stratification import MODIFICATION_POLICIES

            # Convert to string keys to avoid enum instance mismatch issues
            self.policies = {level.value: policy for level, policy in MODIFICATION_POLICIES.items()}
        else:
            # Assume policies dict already uses appropriate keys
            self.policies = policies

        # Use string values as keys to avoid enum instance mismatch issues
        self.modification_count: Dict[str, int] = {level.value: 0 for level in StratificationLevel}

        self.incorporation_history: List[Dict[str, Any]] = []
        self.rollback_history: List[RollbackEvent] = []

        logger.info(
            f"Initialized RuleIncorporationEngine (regression_threshold={regression_threshold})"
        )

    def incorporate(
        self,
        rule: GeneratedRule,
        target_layer: StratificationLevel,
        validation_report: ValidationReport,
        is_autonomous: bool = True,
    ) -> IncorporationResult:
        """
        Attempt to incorporate rule into symbolic core.

        Args:
            rule: Generated rule to incorporate
            target_layer: Target stratification layer
            validation_report: Validation results for the rule
            is_autonomous: Whether this is autonomous modification (vs human-initiated)

        Returns:
            IncorporationResult indicating success/failure/pending
        """
        logger.info(
            f"Attempting to incorporate rule into {target_layer.value} layer: {rule.asp_rule[:50]}..."
        )

        # 1. Check policy
        if target_layer.value not in self.policies:
            # Fallback: if policies not properly initialized, get from stratification module
            from loft.symbolic.stratification import get_policy

            policy = get_policy(target_layer)
            logger.warning(
                f"Policy for {target_layer.value} not found in engine policies, using default"
            )
        else:
            policy = self.policies[target_layer.value]

        if not policy.autonomous_allowed and is_autonomous:
            logger.warning(f"Autonomous modification not allowed for {target_layer.value} layer")
            return IncorporationResult(
                status="blocked",
                reason=f"Autonomous modification not allowed for {target_layer.value} layer",
                requires_human_review=True,
            )

        # 2. Check confidence threshold
        confidence = rule.confidence
        if confidence < policy.confidence_threshold:
            logger.info(
                f"Confidence {confidence:.2f} below threshold {policy.confidence_threshold}"
            )
            return IncorporationResult(
                status="rejected",
                reason=f"Confidence {confidence:.2f} below threshold {policy.confidence_threshold}",
                requires_human_review=False,
            )

        # 3. Check modification limit
        if self.modification_count[target_layer.value] >= policy.max_modifications_per_session:
            logger.warning(
                f"Reached max modifications ({policy.max_modifications_per_session}) for {target_layer.value}"
            )
            return IncorporationResult(
                status="deferred",
                reason=f"Reached max modifications ({policy.max_modifications_per_session}) for session",
                requires_human_review=False,
            )

        # 4. Snapshot current state (for rollback)
        current_state = self.asp_core.get_state()
        snapshot_id = self.version_control.create_snapshot(
            current_state, message=f"Before incorporating {rule.asp_rule[:30]}..."
        )

        accuracy_before = self.test_suite.measure_accuracy()

        try:
            # 5. Add rule to core
            self.asp_core.add_rule(
                rule_text=rule.asp_rule,
                stratification_level=target_layer,
                confidence=confidence,
                metadata={
                    "source": "llm_generated",
                    "source_type": rule.source_type,
                    "reasoning": rule.reasoning,
                    "validation_confidence": (
                        validation_report.final_decision
                        if hasattr(validation_report, "final_decision")
                        else "unknown"
                    ),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # 6. Run regression tests (if required)
            if policy.regression_test_required:
                logger.debug("Running regression tests...")
                regression_result = self.test_suite.run_all()
                accuracy_after = regression_result["accuracy"]

                # Check for regression via test failures
                if not regression_result["passed"]:
                    # Rollback!
                    logger.warning("Regression test failed, rolling back")
                    self._rollback(
                        current_state=current_state,
                        snapshot_id=snapshot_id,
                        reason="Regression test failures after incorporation",
                        failed_rule=rule,
                        regression_details={"failures": regression_result["failures"]},
                    )

                    return IncorporationResult(
                        status="rejected",
                        reason="Regression test failures after incorporation",
                        regression_failures=regression_result["failures"],
                        requires_human_review=True,
                        regression_detected=True,
                    )

                # Check for accuracy regression via threshold
                accuracy_drop = accuracy_before - accuracy_after
                if accuracy_drop > self.regression_threshold:
                    # Rollback due to accuracy regression!
                    logger.warning(
                        f"Accuracy regression detected: {accuracy_drop:.2%} drop "
                        f"(threshold: {self.regression_threshold:.2%})"
                    )
                    self._rollback(
                        current_state=current_state,
                        snapshot_id=snapshot_id,
                        reason=f"Accuracy regression: {accuracy_drop:.2%} drop",
                        failed_rule=rule,
                        regression_details={
                            "accuracy_before": accuracy_before,
                            "accuracy_after": accuracy_after,
                            "drop": accuracy_drop,
                            "threshold": self.regression_threshold,
                        },
                    )

                    return IncorporationResult(
                        status="rejected",
                        reason=f"Accuracy regression: {accuracy_drop:.2%} drop (threshold: {self.regression_threshold:.2%})",
                        accuracy_before=accuracy_before,
                        accuracy_after=accuracy_after,
                        requires_human_review=True,
                        regression_detected=True,
                    )

            # 7. Success!
            self.modification_count[target_layer.value] += 1
            accuracy_after = self.test_suite.measure_accuracy()

            # Record in history
            self.incorporation_history.append(
                {
                    "rule": rule.asp_rule,
                    "layer": target_layer.value,
                    "confidence": confidence,
                    "snapshot_id": snapshot_id,
                    "accuracy_before": accuracy_before,
                    "accuracy_after": accuracy_after,
                    "timestamp": datetime.now(),
                }
            )

            logger.info(
                f"Successfully incorporated rule into {target_layer.value} layer "
                f"(modification #{self.modification_count[target_layer.value]})"
            )

            return IncorporationResult(
                status="success",
                reason="Rule successfully incorporated",
                snapshot_id=snapshot_id,
                modification_number=self.modification_count[target_layer.value],
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
            )

        except Exception as e:
            # Rollback on any error
            logger.error(f"Error during incorporation: {e}")
            self._rollback(
                current_state=current_state,
                snapshot_id=snapshot_id,
                reason=f"Exception during incorporation: {str(e)}",
                failed_rule=rule,
                regression_details={"exception": str(e)},
            )

            return IncorporationResult(status="error", reason=str(e), requires_human_review=True)

    def _rollback(
        self,
        current_state: Dict[str, Any],
        snapshot_id: str,
        reason: str,
        failed_rule: GeneratedRule,
        regression_details: Optional[Dict[str, Any]] = None,
    ):
        """
        Rollback to previous state and record the event.

        Args:
            current_state: State to restore
            snapshot_id: Snapshot ID for reference
            reason: Reason for rollback
            failed_rule: Rule that caused the rollback
            regression_details: Details about the regression
        """
        self.asp_core.restore_state(current_state)

        # Record rollback event
        rollback_event = RollbackEvent(
            timestamp=datetime.now(),
            reason=reason,
            snapshot_id=snapshot_id,
            failed_rule_text=failed_rule.asp_rule,
            regression_details=regression_details,
        )

        self.rollback_history.append(rollback_event)

        logger.warning(
            f"Rollback complete: {reason}",
            snapshot_id=snapshot_id,
            rollback_count=len(self.rollback_history),
        )

    def reset_session(self):
        """Reset modification counters (e.g., start of new session)."""
        self.modification_count = {level.value: 0 for level in StratificationLevel}
        logger.info("Modification counters reset for new session")

    def get_statistics(self) -> Dict[str, Any]:
        """Get incorporation statistics."""
        total_modifications = sum(self.modification_count.values())
        total_rollbacks = len(self.rollback_history)

        # Calculate success rate
        total_attempts = len(self.incorporation_history) + total_rollbacks
        success_rate = (
            len(self.incorporation_history) / total_attempts if total_attempts > 0 else 0.0
        )

        # modification_count already uses string keys (layer.value)
        by_layer = dict(self.modification_count)

        return {
            "total_modifications": total_modifications,
            "by_layer": by_layer,
            "incorporation_history_count": len(self.incorporation_history),
            "rollback_count": total_rollbacks,
            "success_rate": success_rate,
            "current_accuracy": self.test_suite.measure_accuracy(),
            "regression_threshold": self.regression_threshold,
        }

    def get_history(self, layer: Optional[StratificationLevel] = None) -> List[Dict]:
        """
        Get incorporation history.

        Args:
            layer: Filter by specific layer (None for all)

        Returns:
            List of incorporation records
        """
        if layer is None:
            return self.incorporation_history.copy()

        return [h for h in self.incorporation_history if h["layer"] == layer.value]

    def get_rollback_history(self) -> List[RollbackEvent]:
        """
        Get rollback history.

        Returns:
            List of rollback events
        """
        return self.rollback_history.copy()

    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """
        Rollback to a specific snapshot.

        Args:
            snapshot_id: Snapshot to rollback to

        Returns:
            True if successful
        """
        state = self.version_control.get_snapshot(snapshot_id)
        if state is None:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False

        self.asp_core.restore_state(state)
        logger.info(f"Rolled back to snapshot {snapshot_id}")
        return True
