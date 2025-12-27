"""
Pydantic schemas for knowledge database API.

Provides data validation and serialization for database operations.

Issue #271: Persistent Legal Knowledge Database
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RuleCreate(BaseModel):
    """Schema for creating a new rule."""

    asp_rule: str = Field(..., min_length=1, description="ASP rule text")
    domain: Optional[str] = Field(None, max_length=100, description="Legal domain")
    jurisdiction: Optional[str] = Field(None, max_length=50, description="Jurisdiction")
    doctrine: Optional[str] = Field(None, max_length=100, description="Legal doctrine")
    stratification_level: Optional[str] = Field(
        None, max_length=20, description="Stratification level"
    )
    source_type: Optional[str] = Field(None, max_length=50, description="Source type")
    source_cases: Optional[List[str]] = Field(None, description="Source case IDs")
    generator_model: Optional[str] = Field(None, max_length=100)
    generator_prompt_version: Optional[str] = Field(None, max_length=50)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    tags: Optional[List[str]] = None
    rule_metadata: Optional[dict] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "asp_rule": "valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
                "domain": "contracts",
                "doctrine": "offer-acceptance",
                "stratification_level": "tactical",
                "confidence": 0.95,
                "reasoning": "A valid contract requires three essential elements",
            }
        }
    )


class RuleUpdate(BaseModel):
    """Schema for updating an existing rule."""

    asp_rule: Optional[str] = None
    domain: Optional[str] = None
    jurisdiction: Optional[str] = None
    doctrine: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None
    is_archived: Optional[bool] = None
    rule_metadata: Optional[dict] = None


class RuleFilter(BaseModel):
    """Schema for filtering rules in search queries."""

    domain: Optional[str] = None
    jurisdiction: Optional[str] = None
    doctrine: Optional[str] = None
    stratification_level: Optional[str] = None
    source_type: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_active: Optional[bool] = True
    is_archived: Optional[bool] = False
    tags: Optional[List[str]] = None
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "domain": "contracts",
                "min_confidence": 0.8,
                "is_active": True,
                "limit": 50,
            }
        }
    )


class RuleResponse(BaseModel):
    """Schema for rule retrieval responses."""

    rule_id: str
    asp_rule: str
    domain: Optional[str] = None
    jurisdiction: Optional[str] = None
    doctrine: Optional[str] = None
    stratification_level: Optional[str] = None
    source_type: Optional[str] = None
    source_cases: Optional[List[str]] = None
    confidence: Optional[float] = None
    validation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime
    last_used: Optional[datetime] = None
    reasoning: Optional[str] = None
    tags: Optional[List[str]] = None
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)


class QuestionCreate(BaseModel):
    """Schema for recording a question."""

    question_text: str = Field(..., min_length=1)
    asp_query: Optional[str] = None
    answer: Optional[str] = None
    reasoning: Optional[str] = None
    rules_used: Optional[List[str]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    correct: Optional[bool] = None
    domain: Optional[str] = None
    rule_metadata: Optional[dict] = None


class QuestionResponse(BaseModel):
    """Schema for question retrieval responses."""

    question_id: str
    question_text: str
    asp_query: Optional[str] = None
    answer: Optional[str] = None
    reasoning: Optional[str] = None
    rules_used: Optional[List[str]] = None
    confidence: Optional[float] = None
    correct: Optional[bool] = None
    domain: Optional[str] = None
    created_at: datetime
    answered_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class KnowledgeCoverageStats(BaseModel):
    """Schema for knowledge coverage statistics."""

    domain: str
    rule_count: int = 0
    question_count: int = 0
    accuracy: Optional[float] = None
    avg_confidence: Optional[float] = None
    last_updated: datetime

    model_config = ConfigDict(from_attributes=True)


class RulePerformanceUpdate(BaseModel):
    """Schema for updating rule performance metrics."""

    success: bool
    timestamp: Optional[datetime] = None


class DatabaseStats(BaseModel):
    """Overall database statistics."""

    total_rules: int
    active_rules: int
    archived_rules: int
    total_questions: int
    domains: List[str]
    avg_confidence: Optional[float] = None
    coverage_by_domain: dict


class MigrationResult(BaseModel):
    """Result of ASP file migration."""

    files_processed: int
    rules_imported: int
    rules_skipped: int
    errors: int
    error_messages: List[str] = Field(default_factory=list)
