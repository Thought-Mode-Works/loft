"""
Storage backend for version control system.

Handles persistence of commits and version history to disk.
"""

from pathlib import Path
from typing import List, Optional
from .core_state import Commit


class VersionStorage:
    """
    File-based storage for version control.

    Stores commits as JSON files in a directory structure:
    - .loft/
      - commits/
        - {commit_id}.json
      - refs/
        - heads/
          - {branch_name}  (contains commit_id)
        - tags/
          - {tag_name}  (contains commit_id)
      - HEAD  (contains current branch or commit_id)
    """

    def __init__(self, base_dir: Path = Path(".loft")):
        """
        Initialize storage.

        Args:
            base_dir: Base directory for version control storage
        """
        self.base_dir = base_dir
        self.commits_dir = self.base_dir / "commits"
        self.refs_dir = self.base_dir / "refs"
        self.heads_dir = self.refs_dir / "heads"
        self.tags_dir = self.refs_dir / "tags"
        self.head_file = self.base_dir / "HEAD"

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all necessary directories exist."""
        self.commits_dir.mkdir(parents=True, exist_ok=True)
        self.heads_dir.mkdir(parents=True, exist_ok=True)
        self.tags_dir.mkdir(parents=True, exist_ok=True)

    def save_commit(self, commit: Commit) -> None:
        """
        Save a commit to storage.

        Args:
            commit: Commit to save
        """
        commit_file = self.commits_dir / f"{commit.commit_id}.json"
        with open(commit_file, "w") as f:
            f.write(commit.to_json())

    def load_commit(self, commit_id: str) -> Optional[Commit]:
        """
        Load a commit from storage.

        Args:
            commit_id: ID of commit to load

        Returns:
            Commit if found, None otherwise
        """
        commit_file = self.commits_dir / f"{commit_id}.json"
        if not commit_file.exists():
            return None

        with open(commit_file, "r") as f:
            return Commit.from_json(f.read())

    def list_commits(self) -> List[str]:
        """
        List all commit IDs.

        Returns:
            List of commit IDs
        """
        return [f.stem for f in self.commits_dir.glob("*.json")]

    def save_branch(self, branch_name: str, commit_id: str) -> None:
        """
        Save a branch reference.

        Args:
            branch_name: Name of the branch
            commit_id: Commit ID the branch points to
        """
        branch_file = self.heads_dir / branch_name
        with open(branch_file, "w") as f:
            f.write(commit_id)

    def load_branch(self, branch_name: str) -> Optional[str]:
        """
        Load a branch reference.

        Args:
            branch_name: Name of the branch

        Returns:
            Commit ID if branch exists, None otherwise
        """
        branch_file = self.heads_dir / branch_name
        if not branch_file.exists():
            return None

        with open(branch_file, "r") as f:
            return f.read().strip()

    def list_branches(self) -> List[str]:
        """
        List all branches.

        Returns:
            List of branch names
        """
        return [f.name for f in self.heads_dir.iterdir() if f.is_file()]

    def save_tag(self, tag_name: str, commit_id: str) -> None:
        """
        Save a tag reference.

        Args:
            tag_name: Name of the tag
            commit_id: Commit ID the tag points to
        """
        tag_file = self.tags_dir / tag_name
        with open(tag_file, "w") as f:
            f.write(commit_id)

    def load_tag(self, tag_name: str) -> Optional[str]:
        """
        Load a tag reference.

        Args:
            tag_name: Name of the tag

        Returns:
            Commit ID if tag exists, None otherwise
        """
        tag_file = self.tags_dir / tag_name
        if not tag_file.exists():
            return None

        with open(tag_file, "r") as f:
            return f.read().strip()

    def list_tags(self) -> List[str]:
        """
        List all tags.

        Returns:
            List of tag names
        """
        return [f.name for f in self.tags_dir.iterdir() if f.is_file()]

    def get_head(self) -> Optional[str]:
        """
        Get current HEAD (branch name or commit ID).

        Returns:
            HEAD reference (e.g., "ref: refs/heads/main" or commit ID)
        """
        if not self.head_file.exists():
            return None

        with open(self.head_file, "r") as f:
            return f.read().strip()

    def set_head(self, ref: str) -> None:
        """
        Set HEAD to a branch or commit.

        Args:
            ref: Reference (e.g., "ref: refs/heads/main" or commit ID)
        """
        with open(self.head_file, "w") as f:
            f.write(ref)

    def get_current_branch(self) -> Optional[str]:
        """
        Get name of current branch.

        Returns:
            Branch name if on a branch, None if detached HEAD
        """
        head = self.get_head()
        if head and head.startswith("ref: refs/heads/"):
            return head.replace("ref: refs/heads/", "")
        return None

    def get_current_commit_id(self) -> Optional[str]:
        """
        Get current commit ID.

        Returns:
            Commit ID of current HEAD
        """
        head = self.get_head()
        if not head:
            return None

        if head.startswith("ref: refs/heads/"):
            branch_name = head.replace("ref: refs/heads/", "")
            return self.load_branch(branch_name)
        else:
            # Detached HEAD, direct commit ID
            return head

    def delete_branch(self, branch_name: str) -> bool:
        """
        Delete a branch.

        Args:
            branch_name: Name of branch to delete

        Returns:
            True if deleted, False if didn't exist
        """
        branch_file = self.heads_dir / branch_name
        if branch_file.exists():
            branch_file.unlink()
            return True
        return False

    def delete_tag(self, tag_name: str) -> bool:
        """
        Delete a tag.

        Args:
            tag_name: Name of tag to delete

        Returns:
            True if deleted, False if didn't exist
        """
        tag_file = self.tags_dir / tag_name
        if tag_file.exists():
            tag_file.unlink()
            return True
        return False
