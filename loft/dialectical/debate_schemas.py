"""
Schemas for multi-agent debate framework (Phase 4.2).

Defines data structures for thesis-antithesis-synthesis dialectical cycles.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from loft.dialectical.critique_schemas import CritiqueReport
from loft.neural.rule_schemas import GeneratedRule


class DebatePhase(Enum):
    """Phase in dialectical cycle."""

    THESIS = "thesis"  # Generator proposes rule
    ANTITHESIS = "antithesis"  # Critic identifies flaws
    SYNTHESIS = "synthesis"  # Synthesizer combines insights
    CONVERGED = "converged"  # Debate has reached stable state


@dataclass
class DebateArgument:
    """
    A single argument in the debate (claim with supporting evidence).
    """

    speaker: str  # "generator", "critic", "synthesizer"
    content: str  # Argument text
    references: List[str] = field(default_factory=list)  # Referenced prior arguments
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5


@dataclass
class DebateRound:
    """
    One complete dialectical cycle: thesis → antithesis → synthesis.
    """

    round_number: int
    thesis: GeneratedRule  # Proposed rule from generator
    thesis_argument: DebateArgument  # Generator's reasoning
    antithesis: CritiqueReport  # Critique from critic
    antithesis_argument: DebateArgument  # Critic's reasoning
    synthesis: Optional[GeneratedRule] = None  # Synthesized rule
    synthesis_argument: Optional[DebateArgument] = None  # Synthesizer's reasoning
    convergence_score: float = 0.0  # How similar is synthesis to thesis (0-1)
    improvement_score: float = 0.0  # How much better is synthesis (0-1)
    phase: DebatePhase = DebatePhase.THESIS


@dataclass
class DialecticalCycleResult:
    """
    Complete result of multi-round dialectical debate.
    """

    initial_proposal: GeneratedRule
    final_rule: GeneratedRule
    debate_rounds: List[DebateRound]
    total_rounds: int
    converged: bool
    convergence_reason: Optional[str] = None
    improvement_score: float = 0.0  # Overall quality improvement
    debate_transcript: List[DebateArgument] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get human-readable summary of debate."""
        return f"""
Dialectical Cycle Summary
========================
Initial Proposal: {self.initial_proposal.asp_rule[:80]}...
Final Rule: {self.final_rule.asp_rule[:80]}...
Total Rounds: {self.total_rounds}
Converged: {self.converged}
Improvement Score: {self.improvement_score:.2f}
Convergence Reason: {self.convergence_reason or "N/A"}
        """.strip()


@dataclass
class DebateContext:
    """
    Context passed through debate rounds.
    """

    knowledge_gap_description: str
    existing_rules: List[str]
    existing_predicates: List[str]
    target_layer: str = "tactical"
    domain: str = "legal"
    constraints: Optional[str] = None
    max_rounds: int = 3
    convergence_threshold: float = 0.85  # Similarity threshold for convergence
