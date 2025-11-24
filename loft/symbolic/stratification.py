"""
Stratification system for safe rule modification.

Implements stratified modification authority where different layers have
different modification policies based on their importance and stability.

From CLAUDE.md:
    - Constitutional layer: requires human approval (immutable)
    - Strategic layer: requires validation threshold >0.9
    - Tactical layer: requires validation threshold >0.8
    - Operational layer: autonomous modification allowed
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class StratificationLevel(Enum):
    """
    Stratification levels with modification authority.

    Ordered from most stable/important (constitutional) to most dynamic (operational).
    """

    CONSTITUTIONAL = "constitutional"  # Immutable, human-only
    STRATEGIC = "strategic"  # Slow change, high threshold
    TACTICAL = "tactical"  # Frequent updates, medium threshold
    OPERATIONAL = "operational"  # Rapid adaptation, low threshold

    def __lt__(self, other):
        """Enable comparison for stratification levels."""
        if not isinstance(other, StratificationLevel):
            return NotImplemented
        order = [
            StratificationLevel.OPERATIONAL,
            StratificationLevel.TACTICAL,
            StratificationLevel.STRATEGIC,
            StratificationLevel.CONSTITUTIONAL,
        ]
        return order.index(self) < order.index(other)


@dataclass
class ModificationPolicy:
    """
    Policy for modifying a stratification layer.

    Defines what kinds of modifications are allowed, confidence thresholds,
    and safety requirements for each layer.
    """

    level: StratificationLevel
    autonomous_allowed: bool
    confidence_threshold: float
    requires_human_approval: bool
    max_modifications_per_session: int
    regression_test_required: bool
    description: str = ""

    def allows_modification(self, confidence: float, is_autonomous: bool) -> bool:
        """
        Check if a modification is allowed under this policy.

        Args:
            confidence: Confidence score of the proposed rule
            is_autonomous: Whether this is an autonomous modification

        Returns:
            True if modification is allowed
        """
        if is_autonomous and not self.autonomous_allowed:
            return False

        if confidence < self.confidence_threshold:
            return False

        return True

    def summary(self) -> str:
        """Generate human-readable policy summary."""
        return (
            f"{self.level.value.upper()}:\n"
            f"  Autonomous: {'Yes' if self.autonomous_allowed else 'No'}\n"
            f"  Confidence Threshold: {self.confidence_threshold:.2f}\n"
            f"  Human Approval: {'Required' if self.requires_human_approval else 'Not required'}\n"
            f"  Max Modifications: {self.max_modifications_per_session}\n"
            f"  Regression Tests: {'Required' if self.regression_test_required else 'Optional'}"
        )


# Default modification policies for each stratification level
MODIFICATION_POLICIES: Dict[StratificationLevel, ModificationPolicy] = {
    StratificationLevel.CONSTITUTIONAL: ModificationPolicy(
        level=StratificationLevel.CONSTITUTIONAL,
        autonomous_allowed=False,
        confidence_threshold=1.0,  # Impossible to reach autonomously
        requires_human_approval=True,
        max_modifications_per_session=0,
        regression_test_required=True,
        description="Fundamental principles that define system behavior. Immutable without human review.",
    ),
    StratificationLevel.STRATEGIC: ModificationPolicy(
        level=StratificationLevel.STRATEGIC,
        autonomous_allowed=True,
        confidence_threshold=0.90,
        requires_human_approval=False,
        max_modifications_per_session=3,
        regression_test_required=True,
        description="High-level rules guiding domain reasoning. Slow to change, high confidence required.",
    ),
    StratificationLevel.TACTICAL: ModificationPolicy(
        level=StratificationLevel.TACTICAL,
        autonomous_allowed=True,
        confidence_threshold=0.80,
        requires_human_approval=False,
        max_modifications_per_session=10,
        regression_test_required=True,
        description="Domain-specific rules handling common cases. Frequent updates allowed.",
    ),
    StratificationLevel.OPERATIONAL: ModificationPolicy(
        level=StratificationLevel.OPERATIONAL,
        autonomous_allowed=True,
        confidence_threshold=0.70,
        requires_human_approval=False,
        max_modifications_per_session=25,
        regression_test_required=False,  # Optional for performance
        description="Implementation details and optimizations. Rapid adaptation encouraged.",
    ),
}


def get_policy(level: StratificationLevel) -> ModificationPolicy:
    """
    Get modification policy for a stratification level.

    Args:
        level: Stratification level

    Returns:
        Modification policy for that level
    """
    return MODIFICATION_POLICIES[level]


def infer_stratification_level(rule_text: str) -> StratificationLevel:
    """
    Infer stratification level from rule text.

    Args:
        rule_text: ASP rule text

    Returns:
        Inferred stratification level

    Note:
        This is a simplified heuristic. In production, would use more
        sophisticated analysis of rule semantics and dependencies.
    """
    rule_lower = rule_text.lower()

    # Constitutional indicators
    constitutional_keywords = [
        "fundamental",
        "constitutional",
        "core_principle",
        "human_right",
        "due_process",
    ]
    if any(kw in rule_lower for kw in constitutional_keywords):
        return StratificationLevel.CONSTITUTIONAL

    # Strategic indicators
    strategic_keywords = [
        "strategic",
        "policy",
        "shall",
        "must",
        "required",
        "prohibited",
    ]
    if any(kw in rule_lower for kw in strategic_keywords):
        return StratificationLevel.STRATEGIC

    # Operational indicators
    operational_keywords = ["cache", "optimize", "performance", "helper", "utility"]
    if any(kw in rule_lower for kw in operational_keywords):
        return StratificationLevel.OPERATIONAL

    # Default to tactical
    return StratificationLevel.TACTICAL


def print_all_policies():
    """Print all modification policies."""
    print("Modification Policies by Stratification Level:")
    print("=" * 80)
    for level in [
        StratificationLevel.CONSTITUTIONAL,
        StratificationLevel.STRATEGIC,
        StratificationLevel.TACTICAL,
        StratificationLevel.OPERATIONAL,
    ]:
        policy = get_policy(level)
        print()
        print(policy.summary())
        if policy.description:
            print(f"  Description: {policy.description}")
