"""
Migration utilities for importing existing ASP files to knowledge database.

Provides tools to migrate file-based ASP rules to the persistent database.

Issue #271: Persistent Legal Knowledge Database
"""

import re
from pathlib import Path
from typing import List, Optional

from loguru import logger

from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.schemas import MigrationResult


class ASPFileMigrator:
    """
    Migrates ASP rules from files to knowledge database.

    Parses ASP files and imports rules with metadata inferred from
    file paths and content.
    """

    def __init__(self, knowledge_db: KnowledgeDatabase):
        """
        Initialize migrator.

        Args:
            knowledge_db: KnowledgeDatabase instance to import into
        """
        self.knowledge_db = knowledge_db

    def migrate_directory(
        self,
        rules_dir: str,
        default_domain: Optional[str] = None,
        default_source_type: str = "imported",
    ) -> MigrationResult:
        """
        Migrate all ASP files in a directory to database.

        Args:
            rules_dir: Directory containing ASP files
            default_domain: Default domain if not inferrable
            default_source_type: Source type for imported rules

        Returns:
            MigrationResult with statistics
        """
        rules_path = Path(rules_dir)

        if not rules_path.exists():
            raise ValueError(f"Rules directory does not exist: {rules_dir}")

        stats = {
            "files_processed": 0,
            "rules_imported": 0,
            "rules_skipped": 0,
            "errors": 0,
            "error_messages": [],
        }

        # Find all .lp files
        asp_files = list(rules_path.glob("**/*.lp"))

        logger.info(f"Found {len(asp_files)} ASP files in {rules_dir}")

        for asp_file in asp_files:
            try:
                result = self.migrate_file(
                    asp_file=asp_file,
                    default_domain=default_domain,
                    default_source_type=default_source_type,
                )

                stats["files_processed"] += 1
                stats["rules_imported"] += result["rules_imported"]
                stats["rules_skipped"] += result["rules_skipped"]

            except Exception as e:
                stats["errors"] += 1
                error_msg = f"Error processing {asp_file}: {e}"
                stats["error_messages"].append(error_msg)
                logger.error(error_msg)

        logger.info(
            f"Migration complete: {stats['rules_imported']} rules imported, "
            f"{stats['rules_skipped']} skipped, {stats['errors']} errors"
        )

        return MigrationResult(**stats)

    def migrate_file(
        self,
        asp_file: Path,
        default_domain: Optional[str] = None,
        default_source_type: str = "imported",
    ) -> dict:
        """
        Migrate a single ASP file to database.

        Args:
            asp_file: Path to ASP file
            default_domain: Default domain if not inferrable
            default_source_type: Source type for imported rules

        Returns:
            dict with import statistics
        """
        stats = {"rules_imported": 0, "rules_skipped": 0}

        # Extract metadata from file path
        metadata = self._extract_metadata_from_path(asp_file)

        # Use default domain if not extracted
        if not metadata.get("domain"):
            metadata["domain"] = default_domain

        # Set source type
        metadata["source_type"] = default_source_type

        # Parse rules from file
        rules = self._parse_asp_file(asp_file)

        logger.debug(f"Parsing {asp_file}: found {len(rules)} rules")

        for rule_data in rules:
            try:
                # Merge file metadata with rule-specific metadata
                rule_metadata = {**metadata, **rule_data}

                # Add to database
                self.knowledge_db.add_rule(
                    asp_rule=rule_metadata["asp_rule"],
                    domain=rule_metadata.get("domain"),
                    jurisdiction=rule_metadata.get("jurisdiction"),
                    doctrine=rule_metadata.get("doctrine"),
                    stratification_level=rule_metadata.get("stratification_level"),
                    source_type=rule_metadata.get("source_type"),
                    confidence=rule_metadata.get("confidence"),
                    reasoning=rule_metadata.get("reasoning"),
                )

                stats["rules_imported"] += 1

            except ValueError as e:
                # Rule already exists
                logger.debug(f"Skipping duplicate rule: {e}")
                stats["rules_skipped"] += 1
            except Exception as e:
                logger.error(f"Error importing rule: {e}")
                raise

        return stats

    def _extract_metadata_from_path(self, file_path: Path) -> dict:
        """
        Extract metadata from file path structure.

        Example paths:
          - asp_rules/tactical.lp → stratification_level=tactical
          - asp_rules/contracts/tactical.lp → domain=contracts, level=tactical
          - datasets/contracts/case_001.lp → domain=contracts

        Args:
            file_path: Path to ASP file

        Returns:
            dict with extracted metadata
        """
        metadata = {}

        # Get filename and parent directories
        filename = file_path.stem  # filename without extension

        # Check for stratification level in filename or path
        stratification_levels = [
            "constitutional",
            "strategic",
            "tactical",
            "operational",
        ]
        for level in stratification_levels:
            if level in filename.lower() or level in str(file_path).lower():
                metadata["stratification_level"] = level
                break

        # Check for domain in path
        legal_domains = [
            "contracts",
            "torts",
            "property",
            "constitutional",
            "criminal",
            "civil_procedure",
        ]
        for domain in legal_domains:
            if domain in str(file_path).lower():
                metadata["domain"] = domain
                break

        return metadata

    def _parse_asp_file(self, file_path: Path) -> List[dict]:
        """
        Parse ASP file and extract rules.

        Handles:
        - Comments (% ...)
        - Multi-line rules
        - Metadata in comments

        Args:
            file_path: Path to ASP file

        Returns:
            List of dicts with rule data
        """
        rules = []
        current_rule = {"asp_rule": "", "reasoning": None}
        current_comment = ""

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Comment line
            if line.startswith("%"):
                # Extract comment text
                comment_text = line[1:].strip()

                # Skip header comments
                if any(
                    keyword in comment_text.lower()
                    for keyword in ["generated:", "total rules:", "layer"]
                ):
                    continue

                # Store as reasoning for next rule
                if comment_text:
                    current_comment = comment_text

                continue

            # Rule line
            if line and not line.startswith("%"):
                # Append to current rule
                current_rule["asp_rule"] += " " + line

                # Check if rule is complete (ends with .)
                if line.endswith("."):
                    # Clean up rule text
                    current_rule["asp_rule"] = current_rule["asp_rule"].strip()

                    # Add reasoning from preceding comment
                    if current_comment:
                        current_rule["reasoning"] = current_comment

                    # Extract confidence if present in comment
                    confidence_match = re.search(
                        r"confidence[:\s]+(\d+\.?\d*)", current_comment, re.IGNORECASE
                    )
                    if confidence_match:
                        current_rule["confidence"] = float(confidence_match.group(1))

                    # Add rule if it's not empty
                    if current_rule["asp_rule"]:
                        rules.append(current_rule.copy())

                    # Reset for next rule
                    current_rule = {"asp_rule": "", "reasoning": None}
                    current_comment = ""

        return rules


def migrate_asp_files_to_database(
    rules_dir: str,
    database_url: str = "sqlite:///legal_knowledge.db",
    default_domain: Optional[str] = None,
) -> MigrationResult:
    """
    Convenience function to migrate ASP files to database.

    Args:
        rules_dir: Directory containing ASP files
        database_url: Database connection URL
        default_domain: Default domain for rules

    Returns:
        MigrationResult with statistics

    Example:
        >>> result = migrate_asp_files_to_database(
        ...     rules_dir="./asp_rules",
        ...     database_url="sqlite:///legal_knowledge.db",
        ...     default_domain="contracts"
        ... )
        >>> print(f"Imported {result.rules_imported} rules")
    """
    # Create database
    db = KnowledgeDatabase(database_url)

    # Create migrator
    migrator = ASPFileMigrator(db)

    # Run migration
    result = migrator.migrate_directory(
        rules_dir=rules_dir,
        default_domain=default_domain,
    )

    # Close database
    db.close()

    return result
