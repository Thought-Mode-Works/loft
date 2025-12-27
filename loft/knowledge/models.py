"""
SQLAlchemy ORM models for legal knowledge database.

Defines database schema for storing ASP rules, questions, and metadata.

Issue #271: Persistent Legal Knowledge Database
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import JSON, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class LegalRule(Base):
    """
    Persistent storage for ASP legal rules with metadata.

    Stores rules with rich metadata for categorization, provenance tracking,
    and performance monitoring.
    """

    __tablename__ = "legal_rules"

    # Primary key
    rule_id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=lambda: str(uuid4())
    )

    # Rule content
    asp_rule: Mapped[str] = mapped_column(sa.Text, nullable=False)

    # Categorization
    domain: Mapped[Optional[str]] = mapped_column(
        sa.String(100), index=True
    )  # contracts, torts, property
    jurisdiction: Mapped[Optional[str]] = mapped_column(
        sa.String(50), index=True
    )  # federal, CA, NY
    doctrine: Mapped[Optional[str]] = mapped_column(
        sa.String(100), index=True
    )  # offer-acceptance, negligence
    stratification_level: Mapped[Optional[str]] = mapped_column(
        sa.String(20), index=True
    )  # constitutional, strategic, tactical

    # Provenance
    source_type: Mapped[Optional[str]] = mapped_column(
        sa.String(50)
    )  # case_analysis, llm_generation
    source_cases: Mapped[Optional[str]] = mapped_column(
        sa.Text
    )  # JSON array of case IDs
    generator_model: Mapped[Optional[str]] = mapped_column(sa.String(100))
    generator_prompt_version: Mapped[Optional[str]] = mapped_column(sa.String(50))

    # Quality metrics
    confidence: Mapped[Optional[float]] = mapped_column(sa.Float, index=True)
    validation_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    success_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    failure_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    last_success_date: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)
    last_failure_date: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False
    )
    last_validated: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)
    last_used: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)

    # Additional metadata
    reasoning: Mapped[Optional[str]] = mapped_column(sa.Text)
    tags: Mapped[Optional[str]] = mapped_column(sa.Text)  # JSON array
    rule_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    # Status
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=True, index=True)
    is_archived: Mapped[bool] = mapped_column(sa.Boolean, default=False)

    # Relationships
    versions: Mapped[List["RuleVersion"]] = relationship(
        "RuleVersion", back_populates="rule", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_rules_domain_active", "domain", "is_active"),
        Index("idx_rules_confidence_active", "confidence", "is_active"),
        Index("idx_asp_rule_unique", "asp_rule", "domain", unique=True),
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<LegalRule(id={self.rule_id[:8]}, domain={self.domain}, asp_rule={self.asp_rule[:50]}...)>"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "asp_rule": self.asp_rule,
            "domain": self.domain,
            "jurisdiction": self.jurisdiction,
            "doctrine": self.doctrine,
            "stratification_level": self.stratification_level,
            "source_type": self.source_type,
            "source_cases": self.source_cases,
            "confidence": self.confidence,
            "validation_count": self.validation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "reasoning": self.reasoning,
            "tags": self.tags,
            "is_active": self.is_active,
        }


class LegalQuestion(Base):
    """
    Record of legal questions answered by the system.

    Tracks questions, answers, and which rules were used for performance
    monitoring and feedback loops.
    """

    __tablename__ = "legal_questions"

    # Primary key
    question_id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=lambda: str(uuid4())
    )

    # Question content
    question_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    asp_query: Mapped[Optional[str]] = mapped_column(sa.Text)

    # Answer
    answer: Mapped[Optional[str]] = mapped_column(sa.String(50))  # yes, no, unknown
    reasoning: Mapped[Optional[str]] = mapped_column(sa.Text)
    rules_used: Mapped[Optional[str]] = mapped_column(sa.Text)  # JSON array of rule_ids
    confidence: Mapped[Optional[float]] = mapped_column(sa.Float)

    # Validation
    correct: Mapped[Optional[bool]] = mapped_column(sa.Boolean)

    # Categorization
    domain: Mapped[Optional[str]] = mapped_column(sa.String(100), index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    answered_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)

    # Additional metadata
    rule_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    def __repr__(self) -> str:
        """String representation."""
        return f"<LegalQuestion(id={self.question_id[:8]}, question={self.question_text[:50]}...)>"


class KnowledgeCoverage(Base):
    """
    Aggregated coverage statistics by legal domain.

    Tracks knowledge base growth and performance over time.
    """

    __tablename__ = "knowledge_coverage"

    # Primary key
    domain: Mapped[str] = mapped_column(sa.String(100), primary_key=True)

    # Statistics
    rule_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    question_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    accuracy: Mapped[Optional[float]] = mapped_column(sa.Float)
    avg_confidence: Mapped[Optional[float]] = mapped_column(sa.Float)

    # Timestamp
    last_updated: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<KnowledgeCoverage(domain={self.domain}, rules={self.rule_count}, accuracy={self.accuracy})>"


class RuleVersion(Base):
    """
    Version history for rule changes.

    Tracks modifications to rules over time for auditing and rollback.
    """

    __tablename__ = "rule_versions"

    # Primary key
    version_id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=lambda: str(uuid4())
    )

    # Foreign key to rule
    rule_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("legal_rules.rule_id"), nullable=False, index=True
    )

    # Version content
    asp_rule: Mapped[str] = mapped_column(sa.Text, nullable=False)
    change_reason: Mapped[Optional[str]] = mapped_column(sa.Text)
    changed_by: Mapped[Optional[str]] = mapped_column(
        sa.String(50)
    )  # system, user, feedback_loop

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False
    )

    # Additional metadata
    rule_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    # Relationships
    rule: Mapped["LegalRule"] = relationship("LegalRule", back_populates="versions")

    def __repr__(self) -> str:
        """String representation."""
        return f"<RuleVersion(id={self.version_id[:8]}, rule_id={self.rule_id[:8]})>"
