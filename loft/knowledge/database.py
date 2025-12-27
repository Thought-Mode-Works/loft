"""
Persistent knowledge database for legal rules.

Provides CRUD operations and advanced queries for rule storage,
retrieval, and analytics.

Issue #271: Persistent Legal Knowledge Database
Epic #270: Legal Reasoning Accumulation Engine
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import Session, sessionmaker

from loft.knowledge.models import (
    Base,
    KnowledgeCoverage,
    LegalQuestion,
    LegalRule,
    RuleVersion,
)
from loft.knowledge.schemas import (
    DatabaseStats,
    KnowledgeCoverageStats,
)


class KnowledgeDatabase:
    """
    Persistent knowledge database for legal rules.

    Provides CRUD operations and advanced queries for rule storage,
    retrieval, and analytics.

    Example:
        >>> db = KnowledgeDatabase("sqlite:///legal_knowledge.db")
        >>> rule_id = db.add_rule(
        ...     asp_rule="valid_contract(X) :- offer(X), acceptance(X).",
        ...     domain="contracts",
        ...     confidence=0.95
        ... )
        >>> rules = db.search_rules(domain="contracts", min_confidence=0.8)
    """

    def __init__(
        self,
        database_url: str = "sqlite:///legal_knowledge.db",
        echo: bool = False,
    ):
        """
        Initialize database connection.

        Args:
            database_url: SQLAlchemy database URL
            echo: Whether to echo SQL statements (for debugging)
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=echo)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        logger.info(f"Initialized KnowledgeDatabase: {database_url}")

    def add_rule(
        self,
        asp_rule: str,
        domain: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        doctrine: Optional[str] = None,
        stratification_level: Optional[str] = None,
        source_type: Optional[str] = None,
        source_cases: Optional[List[str]] = None,
        generator_model: Optional[str] = None,
        generator_prompt_version: Optional[str] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Add a new rule to the database.

        Args:
            asp_rule: ASP rule text
            domain: Legal domain (contracts, torts, etc.)
            jurisdiction: Jurisdiction (federal, CA, etc.)
            doctrine: Specific legal doctrine
            stratification_level: Stratification level
            source_type: How rule was created
            source_cases: List of source case IDs
            generator_model: LLM model that generated it
            generator_prompt_version: Prompt version used
            confidence: Confidence score (0-1)
            reasoning: Explanation of what rule does
            tags: List of tags
            metadata: Additional metadata dict

        Returns:
            rule_id: UUID of created rule

        Raises:
            ValueError: If rule already exists in domain
        """
        with self.SessionLocal() as session:
            # Check for duplicate
            existing = (
                session.query(LegalRule)
                .filter_by(asp_rule=asp_rule, domain=domain)
                .first()
            )

            if existing:
                raise ValueError(
                    f"Rule already exists in domain '{domain}': {existing.rule_id}"
                )

            # Create rule
            rule = LegalRule(
                asp_rule=asp_rule,
                domain=domain,
                jurisdiction=jurisdiction,
                doctrine=doctrine,
                stratification_level=stratification_level,
                source_type=source_type,
                source_cases=json.dumps(source_cases) if source_cases else None,
                generator_model=generator_model,
                generator_prompt_version=generator_prompt_version,
                confidence=confidence,
                reasoning=reasoning,
                tags=json.dumps(tags) if tags else None,
                rule_metadata=metadata,
            )

            session.add(rule)
            session.commit()
            session.refresh(rule)

            rule_id = rule.rule_id

            # Update coverage stats
            self._update_coverage_stats(session, domain)

            logger.info(f"Added rule {rule_id} to domain '{domain}'")

            return rule_id

    def get_rule(self, rule_id: str) -> Optional[LegalRule]:
        """
        Retrieve a rule by ID.

        Args:
            rule_id: UUID of rule

        Returns:
            LegalRule object or None if not found
        """
        with self.SessionLocal() as session:
            rule = session.query(LegalRule).filter_by(rule_id=rule_id).first()
            if rule:
                # Detach from session
                session.expunge(rule)
            return rule

    def search_rules(
        self,
        domain: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        doctrine: Optional[str] = None,
        stratification_level: Optional[str] = None,
        source_type: Optional[str] = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        is_active: bool = True,
        is_archived: bool = False,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by_confidence: bool = True,
    ) -> List[LegalRule]:
        """
        Search rules by metadata filters.

        Args:
            domain: Filter by legal domain
            jurisdiction: Filter by jurisdiction
            doctrine: Filter by doctrine
            stratification_level: Filter by stratification level
            source_type: Filter by source type
            min_confidence: Minimum confidence score
            max_confidence: Maximum confidence score
            is_active: Filter by active status
            is_archived: Filter by archived status
            tags: Filter by tags (any match)
            limit: Maximum number of results
            offset: Number of results to skip
            order_by_confidence: Order by confidence descending

        Returns:
            List of matching LegalRule objects
        """
        with self.SessionLocal() as session:
            query = session.query(LegalRule)

            # Apply filters
            if domain:
                query = query.filter(LegalRule.domain == domain)
            if jurisdiction:
                query = query.filter(LegalRule.jurisdiction == jurisdiction)
            if doctrine:
                query = query.filter(LegalRule.doctrine == doctrine)
            if stratification_level:
                query = query.filter(
                    LegalRule.stratification_level == stratification_level
                )
            if source_type:
                query = query.filter(LegalRule.source_type == source_type)

            # Confidence range
            if min_confidence > 0.0:
                query = query.filter(LegalRule.confidence >= min_confidence)
            if max_confidence < 1.0:
                query = query.filter(LegalRule.confidence <= max_confidence)

            # Status filters
            query = query.filter(LegalRule.is_active == is_active)
            query = query.filter(LegalRule.is_archived == is_archived)

            # Tag filtering (if tags are provided)
            if tags:
                # For simplicity, checking if any tag is in the JSON tags field
                # In production, consider using PostgreSQL array containment
                for tag in tags:
                    query = query.filter(LegalRule.tags.contains(tag))

            # Ordering
            if order_by_confidence:
                query = query.order_by(desc(LegalRule.confidence))
            else:
                query = query.order_by(desc(LegalRule.created_at))

            # Pagination
            query = query.limit(limit).offset(offset)

            results = query.all()

            # Detach from session
            for rule in results:
                session.expunge(rule)

            return results

    def update_rule(self, rule_id: str, **updates) -> bool:
        """
        Update an existing rule.

        Args:
            rule_id: UUID of rule to update
            **updates: Fields to update

        Returns:
            True if rule was updated, False if not found

        Example:
            >>> db.update_rule(rule_id, confidence=0.98, is_active=True)
        """
        with self.SessionLocal() as session:
            rule = session.query(LegalRule).filter_by(rule_id=rule_id).first()

            if not rule:
                return False

            # Create version before updating
            if "asp_rule" in updates and updates["asp_rule"] != rule.asp_rule:
                version = RuleVersion(
                    rule_id=rule_id,
                    asp_rule=rule.asp_rule,
                    change_reason="Updated via API",
                    changed_by="system",
                )
                session.add(version)

            # Update fields
            for key, value in updates.items():
                if hasattr(rule, key):
                    # Handle list fields that need JSON serialization
                    if key in ["source_cases", "tags"] and isinstance(value, list):
                        value = json.dumps(value)
                    setattr(rule, key, value)

            session.commit()

            logger.info(f"Updated rule {rule_id}")

            return True

    def update_rule_performance(
        self,
        rule_id: str,
        success: bool,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Update rule performance metrics after use.

        Args:
            rule_id: UUID of rule
            success: Whether rule led to correct answer
            timestamp: When rule was used (defaults to now)

        Returns:
            True if updated, False if rule not found
        """
        with self.SessionLocal() as session:
            rule = session.query(LegalRule).filter_by(rule_id=rule_id).first()

            if not rule:
                return False

            # Update counts
            rule.validation_count += 1
            if success:
                rule.success_count += 1
                rule.last_success_date = timestamp or datetime.utcnow()
            else:
                rule.failure_count += 1
                rule.last_failure_date = timestamp or datetime.utcnow()

            rule.last_used = timestamp or datetime.utcnow()

            session.commit()

            return True

    def mark_rule_used(self, rule_id: str) -> bool:
        """
        Mark a rule as recently used (update last_used timestamp).

        Args:
            rule_id: UUID of rule

        Returns:
            True if updated, False if rule not found
        """
        with self.SessionLocal() as session:
            rule = session.query(LegalRule).filter_by(rule_id=rule_id).first()

            if not rule:
                return False

            rule.last_used = datetime.utcnow()
            session.commit()

            return True

    def archive_rule(self, rule_id: str, reason: Optional[str] = None) -> bool:
        """
        Archive a rule (soft delete).

        Args:
            rule_id: UUID of rule
            reason: Reason for archiving

        Returns:
            True if archived, False if rule not found
        """
        with self.SessionLocal() as session:
            rule = session.query(LegalRule).filter_by(rule_id=rule_id).first()

            if not rule:
                return False

            # Create version record
            version = RuleVersion(
                rule_id=rule_id,
                asp_rule=rule.asp_rule,
                change_reason=reason or "Archived",
                changed_by="system",
            )
            session.add(version)

            # Archive
            rule.is_active = False
            rule.is_archived = True

            session.commit()

            logger.info(f"Archived rule {rule_id}")

            return True

    def delete_rule(self, rule_id: str) -> bool:
        """
        Permanently delete a rule (use with caution).

        Args:
            rule_id: UUID of rule

        Returns:
            True if deleted, False if rule not found
        """
        with self.SessionLocal() as session:
            rule = session.query(LegalRule).filter_by(rule_id=rule_id).first()

            if not rule:
                return False

            session.delete(rule)
            session.commit()

            logger.warning(f"Deleted rule {rule_id}")

            return True

    def add_question(
        self,
        question_text: str,
        asp_query: Optional[str] = None,
        answer: Optional[str] = None,
        reasoning: Optional[str] = None,
        rules_used: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        correct: Optional[bool] = None,
        domain: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Record a question that was answered.

        Args:
            question_text: Natural language question
            asp_query: Parsed ASP query
            answer: System answer
            reasoning: Explanation
            rules_used: List of rule IDs used
            confidence: Confidence score
            correct: Whether answer was correct
            domain: Legal domain
            metadata: Additional metadata

        Returns:
            question_id: UUID of created question
        """
        with self.SessionLocal() as session:
            question = LegalQuestion(
                question_text=question_text,
                asp_query=asp_query,
                answer=answer,
                reasoning=reasoning,
                rules_used=json.dumps(rules_used) if rules_used else None,
                confidence=confidence,
                correct=correct,
                domain=domain,
                answered_at=datetime.utcnow() if answer else None,
                rule_metadata=metadata,
            )

            session.add(question)
            session.commit()
            session.refresh(question)

            question_id = question.question_id

            # Update coverage stats
            if domain:
                self._update_coverage_stats(session, domain, question_added=True)

            return question_id

    def get_coverage_stats(self, domain: str) -> Optional[KnowledgeCoverageStats]:
        """
        Get coverage statistics for a domain.

        Args:
            domain: Legal domain

        Returns:
            KnowledgeCoverageStats or None if domain not found
        """
        with self.SessionLocal() as session:
            coverage = session.query(KnowledgeCoverage).filter_by(domain=domain).first()

            if not coverage:
                return None

            return KnowledgeCoverageStats.model_validate(coverage)

    def get_all_coverage_stats(self) -> List[KnowledgeCoverageStats]:
        """
        Get coverage statistics for all domains.

        Returns:
            List of KnowledgeCoverageStats
        """
        with self.SessionLocal() as session:
            coverages = session.query(KnowledgeCoverage).all()
            return [KnowledgeCoverageStats.model_validate(c) for c in coverages]

    def get_database_stats(self) -> DatabaseStats:
        """
        Get overall database statistics.

        Returns:
            DatabaseStats with overall metrics
        """
        with self.SessionLocal() as session:
            total_rules = session.query(func.count(LegalRule.rule_id)).scalar()
            active_rules = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(is_active=True)
                .scalar()
            )
            archived_rules = (
                session.query(func.count(LegalRule.rule_id))
                .filter_by(is_archived=True)
                .scalar()
            )
            total_questions = session.query(
                func.count(LegalQuestion.question_id)
            ).scalar()

            # Get unique domains
            domains_result = session.query(LegalRule.domain).distinct().all()
            domains = [d[0] for d in domains_result if d[0]]

            # Average confidence
            avg_conf = (
                session.query(func.avg(LegalRule.confidence))
                .filter(LegalRule.is_active)
                .scalar()
            )

            # Coverage by domain
            coverage_by_domain = {}
            for domain in domains:
                count = (
                    session.query(func.count(LegalRule.rule_id))
                    .filter_by(domain=domain, is_active=True)
                    .scalar()
                )
                coverage_by_domain[domain] = count

            return DatabaseStats(
                total_rules=total_rules or 0,
                active_rules=active_rules or 0,
                archived_rules=archived_rules or 0,
                total_questions=total_questions or 0,
                domains=domains,
                avg_confidence=float(avg_conf) if avg_conf else None,
                coverage_by_domain=coverage_by_domain,
            )

    def export_to_asp_files(self, output_dir: str) -> dict:
        """
        Export active rules to ASP files organized by stratification layer.

        Args:
            output_dir: Directory to write ASP files

        Returns:
            dict with export statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {"files_created": 0, "rules_exported": 0}

        with self.SessionLocal() as session:
            # Group by stratification level
            levels = (
                session.query(LegalRule.stratification_level)
                .filter(LegalRule.is_active)
                .distinct()
                .all()
            )

            for (level,) in levels:
                if not level:
                    continue

                # Get rules for this level
                rules = (
                    session.query(LegalRule)
                    .filter_by(stratification_level=level, is_active=True)
                    .all()
                )

                # Write to file
                filename = output_path / f"{level}.lp"
                with open(filename, "w") as f:
                    f.write(f"% {level.upper()} Layer Rules\n")
                    f.write(f"% Generated: {datetime.utcnow().isoformat()}\n")
                    f.write(f"% Total rules: {len(rules)}\n\n")

                    for rule in rules:
                        if rule.reasoning:
                            f.write(f"% {rule.reasoning}\n")
                        f.write(f"{rule.asp_rule}\n\n")

                stats["files_created"] += 1
                stats["rules_exported"] += len(rules)

                logger.info(f"Exported {len(rules)} rules to {filename}")

        return stats

    def _update_coverage_stats(
        self,
        session: Session,
        domain: Optional[str],
        question_added: bool = False,
    ) -> None:
        """
        Update coverage statistics for a domain.

        Args:
            session: SQLAlchemy session
            domain: Domain to update
            question_added: Whether a question was added
        """
        if not domain:
            return

        coverage = session.query(KnowledgeCoverage).filter_by(domain=domain).first()

        if not coverage:
            coverage = KnowledgeCoverage(domain=domain)
            session.add(coverage)

        # Update rule count
        coverage.rule_count = (
            session.query(func.count(LegalRule.rule_id))
            .filter_by(domain=domain, is_active=True)
            .scalar()
        )

        # Update question count
        coverage.question_count = (
            session.query(func.count(LegalQuestion.question_id))
            .filter_by(domain=domain)
            .scalar()
        )

        # Calculate accuracy
        correct_count = (
            session.query(func.count(LegalQuestion.question_id))
            .filter_by(domain=domain, correct=True)
            .scalar()
        )
        total_answered = (
            session.query(func.count(LegalQuestion.question_id))
            .filter_by(domain=domain)
            .filter(LegalQuestion.correct.isnot(None))
            .scalar()
        )

        if total_answered and total_answered > 0:
            coverage.accuracy = correct_count / total_answered

        # Average confidence
        avg_conf = (
            session.query(func.avg(LegalRule.confidence))
            .filter_by(domain=domain, is_active=True)
            .scalar()
        )
        coverage.avg_confidence = float(avg_conf) if avg_conf else None

        coverage.last_updated = datetime.utcnow()

        session.commit()

    def close(self):
        """Close database connections."""
        self.engine.dispose()
        logger.info("Closed KnowledgeDatabase")
