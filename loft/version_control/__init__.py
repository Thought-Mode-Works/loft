"""
Version control system for symbolic core states.

Provides git-like operations for managing the evolution of the symbolic core.
"""

from .core_state import (
    CoreState,
    Rule,
    Commit,
    StratificationLevel,
    create_state_id,
    create_commit_id,
)

from .diff import (
    CoreStateDiff,
    RuleChange,
    ConfigChange,
    ChangeType,
    compute_diff,
    detect_conflicts,
)

from .version_control import (
    VersionControl,
    VersionControlError,
    MergeConflictError,
)

from .storage import VersionStorage

__all__ = [
    # Core state
    "CoreState",
    "Rule",
    "Commit",
    "StratificationLevel",
    "create_state_id",
    "create_commit_id",
    # Diff
    "CoreStateDiff",
    "RuleChange",
    "ConfigChange",
    "ChangeType",
    "compute_diff",
    "detect_conflicts",
    # Version control
    "VersionControl",
    "VersionControlError",
    "MergeConflictError",
    # Storage
    "VersionStorage",
]
