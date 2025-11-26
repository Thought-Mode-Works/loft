"""
ASP Core Persistence Manager with LinkedASP Metadata Support.

Implements file-based persistence for ASP rules with:
- Atomic writes to prevent corruption
- Version history through snapshots
- Backup and recovery mechanisms
- LinkedASP RDF metadata embedding
- Git integration for full history

Designed per docs/MAINTAINABILITY.md for future queryability.
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from loft.symbolic.stratification import StratificationLevel
from loft.symbolic.asp_rule import ASPRule


class PersistenceError(Exception):
    """Base exception for persistence operations."""

    pass


class CorruptedFileError(PersistenceError):
    """Raised when a persisted file is corrupted or invalid."""

    pass


@dataclass
class SnapshotMetadata:
    """Metadata for a rule snapshot."""

    cycle_number: int
    timestamp: datetime
    rules_count: Dict[str, int]  # Count per layer
    total_rules: int
    description: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "rules_count": self.rules_count,
            "total_rules": self.total_rules,
            "description": self.description,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotMetadata":
        """Load from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ASPPersistenceManager:
    """
    Manages persistent storage of ASP rules with version history.

    Features:
    - File-based storage organized by stratification layer
    - Atomic writes to prevent corruption during crashes
    - Snapshot system for version history
    - Backup/restore functionality
    - LinkedASP metadata embedding for future queryability
    - Git integration for full history tracking

    File Structure:
        asp_rules/
        ├── constitutional.lp          # Constitutional layer rules
        ├── strategic.lp               # Strategic layer rules
        ├── tactical.lp                # Tactical layer rules
        ├── operational.lp             # Operational layer rules
        ├── snapshots/                 # Version history
        │   ├── cycle_001/
        │   │   ├── constitutional.lp
        │   │   ├── strategic.lp
        │   │   ├── tactical.lp
        │   │   ├── operational.lp
        │   │   └── metadata.json
        │   └── ...
        ├── backups/                   # Pre-modification backups
        │   ├── 20250126_140000_pre_cycle_5/
        │   └── ...
        └── LIVING_DOCUMENT.md         # Auto-generated documentation
    """

    def __init__(
        self,
        base_dir: str = "./asp_rules",
        enable_git: bool = True,
        snapshot_retention: Optional[int] = None,
    ):
        """
        Initialize persistence manager.

        Args:
            base_dir: Base directory for rule storage
            enable_git: Whether to use git for version control
            snapshot_retention: Number of snapshots to keep (None = keep all)
        """
        self.base_dir = Path(base_dir)
        self.snapshot_dir = self.base_dir / "snapshots"
        self.backup_dir = self.base_dir / "backups"
        self.enable_git = enable_git
        self.snapshot_retention = snapshot_retention

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize git if enabled
        if self.enable_git:
            self._init_git_repo()

        logger.info(f"Initialized ASPPersistenceManager at {self.base_dir}")

    def _init_git_repo(self) -> None:
        """Initialize git repository if not already present."""
        git_dir = self.base_dir / ".git"
        if not git_dir.exists():
            try:
                import subprocess

                subprocess.run(
                    ["git", "init"],
                    cwd=self.base_dir,
                    check=True,
                    capture_output=True,
                )
                logger.info("Initialized git repository for ASP rules")
            except Exception as e:
                logger.warning(f"Could not initialize git repository: {e}")
                self.enable_git = False

    def save_rule(
        self,
        rule: ASPRule,
        layer: StratificationLevel,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a single rule to appropriate layer file.

        Uses atomic write to prevent corruption.

        Args:
            rule: ASP rule to save
            layer: Stratification layer
            metadata: Optional metadata (LinkedASP-compatible)
        """
        layer_file = self.base_dir / f"{layer.value}.lp"

        # Build LinkedASP metadata comment block
        comment_block = self._generate_linkedasp_comment(rule, layer, metadata)

        try:
            with self._atomic_append(layer_file) as f:
                if comment_block:
                    f.write(comment_block)
                    f.write("\n")
                f.write(f"{rule.asp_text}\n\n")

            logger.debug(f"Saved rule to {layer.value}.lp: {rule.rule_id}")

        except Exception as e:
            logger.error(f"Failed to save rule {rule.rule_id}: {e}")
            raise PersistenceError(f"Could not save rule: {e}") from e

    def _generate_linkedasp_comment(
        self,
        rule: ASPRule,
        layer: StratificationLevel,
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """
        Generate LinkedASP-compatible metadata comment block.

        Embeds RDF-style metadata in ASP comments for future LinkedASP integration.

        Args:
            rule: ASP rule
            layer: Stratification layer
            metadata: Additional metadata

        Returns:
            Comment block with LinkedASP metadata
        """
        # Extract head predicate from rule
        head_predicates = rule.new_predicates if rule.new_predicates else ["unknown"]
        head_predicate = head_predicates[0] if head_predicates else "unknown"

        lines = [
            "%%% <!-- LinkedASP Rule Metadata -->",
            f"%%% Rule ID: {rule.rule_id}",
            f"%%% Predicate: {head_predicate}",
            f"%%% Layer: {layer.value}",
            f"%%% Confidence: {rule.confidence}",
            f"%%% Added: {datetime.now().isoformat()}",
            f"%%% Provenance: {rule.metadata.provenance}",
        ]

        if metadata:
            if "source_type" in metadata:
                lines.append(f"%%% Source Type: {metadata['source_type']}")
            if "source_llm" in metadata:
                lines.append(f"%%% Source LLM: {metadata['source_llm']}")
            if "legal_source" in metadata:
                lines.append(f"%%% Legal Source: {metadata['legal_source']}")
            if "genre" in metadata:
                lines.append(f"%%% Genre: {metadata['genre']}")

        lines.append("%%% <!-- End LinkedASP Metadata -->")

        return "\n".join(lines)

    @contextmanager
    def _atomic_append(self, filepath: Path):
        """
        Context manager for atomic file append operations.

        Writes to temporary file, then atomically renames to target.
        Prevents corruption if process crashes during write.

        Args:
            filepath: Target file path

        Yields:
            File object for writing
        """
        # Read existing content if file exists
        existing_content = ""
        if filepath.exists():
            existing_content = filepath.read_text()

        # Create temporary file in same directory
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent, prefix=f".{filepath.name}.", suffix=".tmp"
        )

        try:
            with os.fdopen(temp_fd, "w") as f:
                # Write existing content
                f.write(existing_content)
                # Yield for new content
                yield f

            # Atomic rename
            os.replace(temp_path, filepath)

        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def save_all_rules(
        self,
        rules_by_layer: Dict[StratificationLevel, List[ASPRule]],
        overwrite: bool = False,
    ) -> None:
        """
        Save all rules to disk, organized by layer.

        Args:
            rules_by_layer: Dictionary mapping layers to rule lists
            overwrite: If True, overwrite existing files; if False, append
        """
        for layer, rules in rules_by_layer.items():
            layer_file = self.base_dir / f"{layer.value}.lp"

            # Create backup before overwrite
            if overwrite and layer_file.exists():
                self.create_backup("pre_overwrite")

            try:
                if overwrite:
                    # Write file completely
                    with self._atomic_write(layer_file) as f:
                        self._write_layer_header(f, layer)
                        for rule in rules:
                            comment_block = self._generate_linkedasp_comment(
                                rule, layer, None
                            )
                            if comment_block:
                                f.write(comment_block)
                                f.write("\n")
                            f.write(f"{rule.asp_text}\n\n")
                else:
                    # Append rules
                    for rule in rules:
                        self.save_rule(rule, layer)

                logger.info(
                    f"Saved {len(rules)} rules to {layer.value}.lp (overwrite={overwrite})"
                )

            except Exception as e:
                logger.error(f"Failed to save rules for {layer.value}: {e}")
                raise PersistenceError(f"Could not save layer {layer.value}: {e}") from e

    @contextmanager
    def _atomic_write(self, filepath: Path):
        """
        Context manager for atomic file write operations (overwrite mode).

        Args:
            filepath: Target file path

        Yields:
            File object for writing
        """
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent, prefix=f".{filepath.name}.", suffix=".tmp"
        )

        try:
            with os.fdopen(temp_fd, "w") as f:
                yield f

            # Atomic rename
            os.replace(temp_path, filepath)

        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _write_layer_header(self, f, layer: StratificationLevel) -> None:
        """Write header comment for a layer file."""
        header = f"""%%% ========================================
%%% {layer.value.upper()} LAYER
%%% ========================================
%%%
%%% Stratification Level: {layer.value}
%%% Generated: {datetime.now().isoformat()}
%%%
%%% This file is automatically managed by ASPPersistenceManager.
%%% Manual edits may be overwritten.
%%%
%%% For LinkedASP metadata format, see docs/MAINTAINABILITY.md
%%%

"""
        f.write(header)

    def load_all_rules(self) -> Dict[StratificationLevel, List[ASPRule]]:
        """
        Load all persisted rules from disk.

        Returns:
            Dictionary mapping layers to lists of ASP rules

        Raises:
            CorruptedFileError: If any file is corrupted or invalid
        """
        rules_by_layer = {}

        for layer in StratificationLevel:
            layer_file = self.base_dir / f"{layer.value}.lp"

            if not layer_file.exists():
                logger.debug(f"No persisted rules for {layer.value}")
                rules_by_layer[layer] = []
                continue

            try:
                rules = self._parse_lp_file(layer_file, layer)
                rules_by_layer[layer] = rules
                logger.info(f"Loaded {len(rules)} rules from {layer.value}.lp")

            except Exception as e:
                logger.error(f"Failed to load rules from {layer.value}.lp: {e}")
                raise CorruptedFileError(
                    f"Could not load {layer.value}.lp: {e}"
                ) from e

        return rules_by_layer

    def _parse_lp_file(
        self, filepath: Path, layer: StratificationLevel
    ) -> List[ASPRule]:
        """
        Parse .lp file and extract ASP rules with metadata.

        Args:
            filepath: Path to .lp file
            layer: Stratification layer

        Returns:
            List of ASP rules
        """
        rules = []
        current_metadata = {}
        in_metadata_block = False

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()

                # Detect LinkedASP metadata blocks
                if "<!-- LinkedASP Rule Metadata -->" in line:
                    in_metadata_block = True
                    current_metadata = {}
                    continue

                if "<!-- End LinkedASP Metadata -->" in line:
                    in_metadata_block = False
                    continue

                # Parse metadata lines
                if in_metadata_block and line.startswith("%%%"):
                    # Extract key-value from comment
                    content = line[3:].strip()
                    if ":" in content:
                        key, value = content.split(":", 1)
                        current_metadata[key.strip()] = value.strip()
                    continue

                # Skip empty lines and general comments
                if not line or line.startswith("%"):
                    continue

                # Parse ASP rule
                if line:
                    try:
                        # Create ASPRule from text
                        from loft.symbolic.asp_rule import RuleMetadata

                        # Generate rule ID from hash of text
                        import hashlib

                        rule_id = hashlib.sha256(line.encode()).hexdigest()[:16]

                        # Get provenance from metadata or default to "persisted"
                        provenance = current_metadata.get("Provenance", "persisted")

                        # Create metadata
                        rule_metadata = RuleMetadata(
                            provenance=provenance,
                            timestamp=current_metadata.get("Added", datetime.now().isoformat()),
                            validation_score=float(current_metadata.get("Confidence", 1.0)),
                        )

                        # Create rule
                        rule = ASPRule(
                            rule_id=current_metadata.get("Rule ID", rule_id),
                            asp_text=line,
                            stratification_level=layer,
                            confidence=float(current_metadata.get("Confidence", 1.0)),
                            metadata=rule_metadata,
                        )
                        rules.append(rule)
                        # Reset metadata for next rule
                        current_metadata = {}

                    except Exception as e:
                        logger.warning(f"Could not parse rule '{line}': {e}")
                        continue

        return rules

    def create_snapshot(
        self, cycle_number: int, description: Optional[str] = None
    ) -> Path:
        """
        Create snapshot of current rule state.

        Args:
            cycle_number: Improvement cycle number
            description: Optional description of snapshot

        Returns:
            Path to snapshot directory
        """
        snapshot_path = self.snapshot_dir / f"cycle_{cycle_number:03d}"
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Count rules per layer
        rules_count = {}
        total_rules = 0

        # Copy all layer files
        for layer in StratificationLevel:
            src = self.base_dir / f"{layer.value}.lp"
            dst = snapshot_path / f"{layer.value}.lp"

            if src.exists():
                shutil.copy2(src, dst)

                # Count rules
                rules = self._parse_lp_file(src, layer)
                rules_count[layer.value] = len(rules)
                total_rules += len(rules)
            else:
                rules_count[layer.value] = 0

        # Save snapshot metadata
        metadata = SnapshotMetadata(
            cycle_number=cycle_number,
            timestamp=datetime.now(),
            rules_count=rules_count,
            total_rules=total_rules,
            description=description,
            git_commit=self._get_git_commit() if self.enable_git else None,
        )

        metadata_file = snapshot_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata.to_dict(), indent=2))

        logger.info(f"Created snapshot for cycle {cycle_number}: {snapshot_path}")

        # Git commit if enabled
        if self.enable_git:
            self._git_commit(f"Snapshot for cycle {cycle_number}")

        # Cleanup old snapshots if retention policy set
        if self.snapshot_retention:
            self._cleanup_old_snapshots()

        return snapshot_path

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _git_commit(self, message: str) -> None:
        """Create git commit for current state."""
        try:
            import subprocess

            # Add all .lp files
            subprocess.run(
                ["git", "add", "*.lp", "snapshots/", "LIVING_DOCUMENT.md"],
                cwd=self.base_dir,
                check=True,
                capture_output=True,
            )

            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.base_dir,
                check=True,
                capture_output=True,
            )

            logger.debug(f"Git commit created: {message}")

        except Exception as e:
            logger.warning(f"Could not create git commit: {e}")

    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond retention limit."""
        if not self.snapshot_retention:
            return

        # List all snapshots
        snapshots = sorted(self.snapshot_dir.glob("cycle_*"))

        # Remove oldest if over limit
        if len(snapshots) > self.snapshot_retention:
            to_remove = snapshots[: len(snapshots) - self.snapshot_retention]
            for snapshot_path in to_remove:
                shutil.rmtree(snapshot_path)
                logger.debug(f"Removed old snapshot: {snapshot_path.name}")

    def list_snapshots(self) -> List[SnapshotMetadata]:
        """
        List all available snapshots.

        Returns:
            List of snapshot metadata, sorted by cycle number
        """
        snapshots = []

        for snapshot_dir in sorted(self.snapshot_dir.glob("cycle_*")):
            metadata_file = snapshot_dir / "metadata.json"

            if metadata_file.exists():
                try:
                    data = json.loads(metadata_file.read_text())
                    metadata = SnapshotMetadata.from_dict(data)
                    snapshots.append(metadata)
                except Exception as e:
                    logger.warning(f"Could not load snapshot metadata from {snapshot_dir}: {e}")
                    continue

        return snapshots

    def restore_snapshot(self, cycle_number: int) -> None:
        """
        Restore rules from a snapshot.

        Creates backup before restore for safety.

        Args:
            cycle_number: Cycle number to restore from

        Raises:
            ValueError: If snapshot not found
        """
        snapshot_path = self.snapshot_dir / f"cycle_{cycle_number:03d}"

        if not snapshot_path.exists():
            raise ValueError(f"Snapshot for cycle {cycle_number} not found")

        # Create backup before restore
        self.create_backup(f"pre_restore_cycle_{cycle_number}")

        # Restore files
        for layer in StratificationLevel:
            src = snapshot_path / f"{layer.value}.lp"
            dst = self.base_dir / f"{layer.value}.lp"

            if src.exists():
                shutil.copy2(src, dst)
                logger.debug(f"Restored {layer.value}.lp from snapshot")
            elif dst.exists():
                # Remove file if not in snapshot
                dst.unlink()
                logger.debug(f"Removed {layer.value}.lp (not in snapshot)")

        logger.info(f"Restored snapshot from cycle {cycle_number}")

        # Git commit if enabled
        if self.enable_git:
            self._git_commit(f"Restored snapshot from cycle {cycle_number}")

    def create_backup(self, label: str) -> Path:
        """
        Create backup of current state.

        Args:
            label: Label for backup (e.g., "pre_restore")

        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{timestamp}_{label}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Copy all layer files
        for layer in StratificationLevel:
            src = self.base_dir / f"{layer.value}.lp"
            if src.exists():
                dst = backup_path / f"{layer.value}.lp"
                shutil.copy2(src, dst)

        logger.info(f"Created backup: {backup_path.name}")

        return backup_path

    def get_stats(self) -> Dict[str, Any]:
        """
        Get persistence statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "base_dir": str(self.base_dir),
            "git_enabled": self.enable_git,
            "snapshot_retention": self.snapshot_retention,
            "layers": {},
            "total_rules": 0,
            "snapshots_count": len(list(self.snapshot_dir.glob("cycle_*"))),
            "backups_count": len(list(self.backup_dir.iterdir())),
        }

        # Count rules per layer
        for layer in StratificationLevel:
            layer_file = self.base_dir / f"{layer.value}.lp"
            if layer_file.exists():
                rules = self._parse_lp_file(layer_file, layer)
                count = len(rules)
                stats["layers"][layer.value] = {
                    "count": count,
                    "file_size": layer_file.stat().st_size,
                    "last_modified": datetime.fromtimestamp(
                        layer_file.stat().st_mtime
                    ).isoformat(),
                }
                stats["total_rules"] += count
            else:
                stats["layers"][layer.value] = {
                    "count": 0,
                    "file_size": 0,
                    "last_modified": None,
                }

        return stats
