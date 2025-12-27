"""
Rule Accumulation Pipeline.

Continuously processes legal cases to extract, validate, and accumulate rules
in the knowledge database.

Issue #273: Continuous Rule Accumulation Pipeline
"""

from loft.accumulation.conflict_detection import ConflictDetector
from loft.accumulation.pipeline import RuleAccumulationPipeline
from loft.accumulation.schemas import (
    AccumulationResult,
    BatchAccumulationReport,
    CaseData,
    Conflict,
    ConflictResolution,
    RuleCandidate,
)

__all__ = [
    "RuleAccumulationPipeline",
    "ConflictDetector",
    "AccumulationResult",
    "BatchAccumulationReport",
    "CaseData",
    "Conflict",
    "ConflictResolution",
    "RuleCandidate",
]
