"""
Version control system for symbolic core states.

Provides git-like operations for managing the evolution of the symbolic core.
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
from .core_state import CoreState, Commit, create_commit_id, StratificationLevel
from .diff import compute_diff, detect_conflicts, CoreStateDiff
from .storage import VersionStorage
from loguru import logger


class VersionControlError(Exception):
    """Base exception for version control errors."""

    pass


class MergeConflictError(VersionControlError):
    """Raised when merge conflicts are detected."""

    pass


class VersionControl:
    """
    Git-like version control for symbolic core states.

    Provides operations for:
    - Committing core states
    - Computing diffs between versions
    - Rollback to previous states
    - Branching and merging
    - Tagging important milestones
    """

    def __init__(
        self,
        storage_dir: Path = Path(".loft"),
        author: str = "loft-system",
    ):
        """
        Initialize version control.

        Args:
            storage_dir: Directory for version control storage
            author: Default author for commits
        """
        self.storage = VersionStorage(storage_dir)
        self.author = author

        # Initialize if needed
        if not self.storage.get_head():
            self._initialize()

    def _initialize(self) -> None:
        """Initialize a new repository."""
        # Create main branch pointing to None (no commits yet)
        self.storage.set_head("ref: refs/heads/main")
        logger.info("Initialized version control repository")

    def commit(
        self,
        state: CoreState,
        message: str,
    ) -> str:
        """
        Commit a core state to version history.

        Args:
            state: Core state to commit
            message: Commit message

        Returns:
            Commit ID

        Example:
            >>> vc = VersionControl()
            >>> commit_id = vc.commit(core_state, "Initial commit")
        """
        # Get parent commit
        parent_id = self.storage.get_current_commit_id()

        # Get current branch
        current_branch = self.storage.get_current_branch()
        if not current_branch:
            current_branch = "main"

        # Create commit
        commit = Commit(
            commit_id=create_commit_id(),
            parent_id=parent_id,
            state=state,
            message=message,
            author=self.author,
            timestamp=datetime.utcnow().isoformat(),
            branch=current_branch,
        )

        # Save commit
        self.storage.save_commit(commit)

        # Update branch reference
        self.storage.save_branch(current_branch, commit.commit_id)

        logger.info(
            f"Committed {commit.commit_id[:8]}: {message}",
            commit_id=commit.commit_id,
            branch=current_branch,
        )

        return commit.commit_id

    def diff(self, from_version: str, to_version: str) -> CoreStateDiff:
        """
        Compute diff between two versions.

        Args:
            from_version: Commit ID, branch name, or tag name (old)
            to_version: Commit ID, branch name, or tag name (new)

        Returns:
            CoreStateDiff showing changes

        Example:
            >>> diff = vc.diff("abc123", "def456")
            >>> print(diff.format())
        """
        state1 = self._resolve_state(from_version)
        state2 = self._resolve_state(to_version)

        if not state1 or not state2:
            raise VersionControlError(
                f"Could not resolve versions: {from_version}, {to_version}"
            )

        return compute_diff(state1, state2)

    def rollback(self, version: str) -> CoreState:
        """
        Rollback to a previous version.

        This creates a new commit with the old state (doesn't rewrite history).

        Args:
            version: Commit ID, branch name, or tag to rollback to

        Returns:
            Core state that was rolled back to

        Example:
            >>> state = vc.rollback("abc123")
        """
        # Get the state to rollback to
        old_state = self._resolve_state(version)
        if not old_state:
            raise VersionControlError(f"Could not resolve version: {version}")

        # Safety check: prevent rollback if it violates constitutional layer
        current_commit_id = self.storage.get_current_commit_id()
        if current_commit_id:
            current_state = self._resolve_state(current_commit_id)
            if current_state:
                self._validate_rollback(current_state, old_state)

        # Create new commit with old state
        commit_id = self.commit(old_state, f"Rollback to {version[:8]}")

        logger.warning(
            f"Rolled back to {version[:8]}",
            new_commit=commit_id,
            target_version=version,
        )

        return old_state

    def log(self, max_count: int = 10, branch: Optional[str] = None) -> List[Commit]:
        """
        Get commit history.

        Args:
            max_count: Maximum number of commits to return
            branch: Branch to get history for (default: current branch)

        Returns:
            List of commits in reverse chronological order

        Example:
            >>> commits = vc.log(max_count=5)
            >>> for commit in commits:
            ...     print(f"{commit.commit_id[:8]}: {commit.message}")
        """
        # Get starting commit
        if branch:
            commit_id = self.storage.load_branch(branch)
        else:
            commit_id = self.storage.get_current_commit_id()

        if not commit_id:
            return []

        # Walk back through history
        commits: List[Commit] = []
        while commit_id and len(commits) < max_count:
            commit = self.storage.load_commit(commit_id)
            if not commit:
                break

            commits.append(commit)
            commit_id = commit.parent_id

        return commits

    def checkout(self, version: str) -> CoreState:
        """
        Checkout a specific version.

        This updates HEAD to point to the version (detached HEAD if commit ID).

        Args:
            version: Commit ID, branch name, or tag

        Returns:
            Core state of the checked out version

        Example:
            >>> state = vc.checkout("feature-branch")
        """
        # Resolve to commit ID
        commit_id = self._resolve_commit_id(version)
        if not commit_id:
            raise VersionControlError(f"Could not resolve version: {version}")

        commit = self.storage.load_commit(commit_id)
        if not commit:
            raise VersionControlError(f"Commit not found: {commit_id}")

        # Check if version is a branch name
        if version in self.storage.list_branches():
            # Checkout branch
            self.storage.set_head(f"ref: refs/heads/{version}")
            logger.info(f"Checked out branch: {version}")
        else:
            # Detached HEAD
            self.storage.set_head(commit_id)
            logger.warning(f"Checked out commit (detached HEAD): {commit_id[:8]}")

        return commit.state

    def branch(self, branch_name: str, from_version: Optional[str] = None) -> None:
        """
        Create a new branch.

        Args:
            branch_name: Name for the new branch
            from_version: Version to branch from (default: current HEAD)

        Example:
            >>> vc.branch("experiment")
            >>> vc.checkout("experiment")
        """
        # Get commit ID to branch from
        if from_version:
            commit_id = self._resolve_commit_id(from_version)
        else:
            commit_id = self.storage.get_current_commit_id()

        if not commit_id:
            raise VersionControlError("No commits to branch from")

        # Create branch
        self.storage.save_branch(branch_name, commit_id)

        logger.info(f"Created branch: {branch_name} at {commit_id[:8]}")

    def merge(self, branch_name: str, message: Optional[str] = None) -> str:
        """
        Merge another branch into current branch.

        Args:
            branch_name: Branch to merge in
            message: Merge commit message (default: auto-generated)

        Returns:
            Merge commit ID

        Raises:
            MergeConflictError: If conflicts are detected

        Example:
            >>> vc.merge("feature-branch")
        """
        # Get current and other branch states
        current_commit_id = self.storage.get_current_commit_id()
        if not current_commit_id:
            raise VersionControlError("No current commit to merge into")

        other_commit_id = self.storage.load_branch(branch_name)
        if not other_commit_id:
            raise VersionControlError(f"Branch not found: {branch_name}")

        current_state = self._resolve_state(current_commit_id)
        other_state = self._resolve_state(other_commit_id)

        if not current_state or not other_state:
            raise VersionControlError("Could not resolve states for merge")

        # Detect conflicts
        conflicts = detect_conflicts(current_state, other_state)
        if conflicts:
            raise MergeConflictError(
                "Merge conflicts detected:\n" + "\n".join(conflicts)
            )

        # Merge rules (simple union for now)
        merged_rules = list(current_state.rules)
        current_rule_ids = {r.rule_id for r in current_state.rules}

        for rule in other_state.rules:
            if rule.rule_id not in current_rule_ids:
                merged_rules.append(rule)

        # Merge configuration (other branch wins on conflicts)
        merged_config = {**current_state.configuration, **other_state.configuration}

        # Merge metrics (take max for positive metrics, min for negative)
        merged_metrics = {**current_state.metrics, **other_state.metrics}

        # Create merged state
        from .core_state import create_state_id

        merged_state = CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=merged_rules,
            configuration=merged_config,
            metrics=merged_metrics,
        )

        # Commit merged state
        if not message:
            message = f"Merge branch '{branch_name}'"

        commit_id = self.commit(merged_state, message)

        logger.info(f"Merged {branch_name} into current branch", merge_commit=commit_id)

        return commit_id

    def tag(self, tag_name: str, version: Optional[str] = None) -> None:
        """
        Tag a version.

        Args:
            tag_name: Name for the tag
            version: Version to tag (default: current HEAD)

        Example:
            >>> vc.tag("v1.0.0")
        """
        if version:
            commit_id = self._resolve_commit_id(version)
        else:
            commit_id = self.storage.get_current_commit_id()

        if not commit_id:
            raise VersionControlError("No commit to tag")

        self.storage.save_tag(tag_name, commit_id)

        logger.info(f"Tagged {commit_id[:8]} as '{tag_name}'")

    def list_branches(self) -> List[str]:
        """List all branches."""
        return self.storage.list_branches()

    def list_tags(self) -> List[str]:
        """List all tags."""
        return self.storage.list_tags()

    def get_current_branch(self) -> Optional[str]:
        """Get current branch name."""
        return self.storage.get_current_branch()

    def _resolve_commit_id(self, version: str) -> Optional[str]:
        """
        Resolve a version reference to a commit ID.

        Args:
            version: Commit ID, branch name, or tag name

        Returns:
            Commit ID or None if not found
        """
        # Try as commit ID
        if self.storage.load_commit(version):
            return version

        # Try as branch
        branch_commit = self.storage.load_branch(version)
        if branch_commit:
            return branch_commit

        # Try as tag
        tag_commit = self.storage.load_tag(version)
        if tag_commit:
            return tag_commit

        return None

    def _resolve_state(self, version: str) -> Optional[CoreState]:
        """
        Resolve a version reference to a core state.

        Args:
            version: Commit ID, branch name, or tag name

        Returns:
            CoreState or None if not found
        """
        commit_id = self._resolve_commit_id(version)
        if not commit_id:
            return None

        commit = self.storage.load_commit(commit_id)
        if not commit:
            return None

        return commit.state

    def _validate_rollback(
        self, current_state: CoreState, target_state: CoreState
    ) -> None:
        """
        Validate that rollback is safe.

        Args:
            current_state: Current state
            target_state: State to rollback to

        Raises:
            VersionControlError: If rollback would violate safety constraints
        """
        # Check constitutional rules are preserved
        current_const = current_state.get_rules_by_level(
            StratificationLevel.CONSTITUTIONAL
        )
        target_const = target_state.get_rules_by_level(
            StratificationLevel.CONSTITUTIONAL
        )

        current_const_ids = {r.rule_id for r in current_const}
        target_const_ids = {r.rule_id for r in target_const}

        # Constitutional rules should never be removed
        removed_const = current_const_ids - target_const_ids
        if removed_const:
            raise VersionControlError(
                f"Rollback would remove constitutional rules: {removed_const}"
            )
