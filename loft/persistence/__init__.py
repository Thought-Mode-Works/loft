"""
Persistent storage for ASP core with version history and backup.

Provides file-based persistence for ASP rules organized by stratification layers,
with support for snapshots, backups, and LinkedASP metadata export.

Designed for future LinkedASP integration (see docs/MAINTAINABILITY.md):
- Embeds RDF metadata in ASP comment blocks
- Tracks provenance and modification history
- Prepares for genre-based organization
"""

from loft.persistence.asp_persistence import (
    ASPPersistenceManager,
    SnapshotMetadata,
    PersistenceError,
    CorruptedFileError,
)

__all__ = [
    "ASPPersistenceManager",
    "SnapshotMetadata",
    "PersistenceError",
    "CorruptedFileError",
]
