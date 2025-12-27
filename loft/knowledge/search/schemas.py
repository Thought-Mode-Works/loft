"""
Data schemas for rule search and retrieval.

Defines structures for search queries, results, and scoring.

Issue #275: Rule Retrieval and Search
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from loft.knowledge.models import LegalRule


@dataclass
class SearchQuery:
    """
    Search query for finding relevant rules.

    Supports multiple search modes:
    - Text/keyword search
    - Domain-based filtering
    - Performance-based ranking
    - Semantic similarity (optional)
    """

    query_text: str
    domain: Optional[str] = None
    jurisdiction: Optional[str] = None
    doctrine: Optional[str] = None
    min_confidence: float = 0.0
    max_results: int = 10
    include_archived: bool = False
    boost_performance: bool = True  # Weight by validation/usage
    boost_confidence: bool = True  # Weight by confidence scores
    keywords: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)  # ASP predicates to match

    def __post_init__(self):
        """Extract keywords from query text if not provided."""
        if not self.keywords and self.query_text:
            # Simple keyword extraction
            words = self.query_text.lower().split()
            # Filter out common words
            stopwords = {
                "is",
                "the",
                "a",
                "an",
                "and",
                "or",
                "not",
                "in",
                "of",
                "to",
                "for",
            }
            self.keywords = [w for w in words if w not in stopwords and len(w) > 2]

    def __str__(self) -> str:
        """String representation."""
        parts = [f"'{self.query_text}'"]
        if self.domain:
            parts.append(f"domain={self.domain}")
        if self.doctrine:
            parts.append(f"doctrine={self.doctrine}")
        return f"SearchQuery({', '.join(parts)})"


@dataclass
class SearchResult:
    """
    Single search result with relevance scoring.

    Contains the matched rule and scores explaining why it was retrieved.
    """

    rule: LegalRule
    relevance_score: float  # Overall relevance (0.0-1.0)
    text_match_score: float = 0.0  # Text/keyword match
    domain_match_score: float = 0.0  # Domain/metadata match
    confidence_score: float = 0.0  # Rule confidence
    performance_score: float = 0.0  # Historical performance
    semantic_score: float = 0.0  # Semantic similarity (if available)
    matched_keywords: List[str] = field(default_factory=list)
    matched_predicates: List[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def rule_id(self) -> str:
        """Get rule ID."""
        return self.rule.rule_id

    @property
    def asp_rule(self) -> str:
        """Get ASP rule text."""
        return self.rule.asp_rule

    def get_score_breakdown(self) -> Dict[str, float]:
        """Get breakdown of score components."""
        return {
            "overall": self.relevance_score,
            "text_match": self.text_match_score,
            "domain_match": self.domain_match_score,
            "confidence": self.confidence_score,
            "performance": self.performance_score,
            "semantic": self.semantic_score,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"SearchResult(rule_id={self.rule_id[:8]}..., "
            f"relevance={self.relevance_score:.2f})"
        )


@dataclass
class SearchResults:
    """
    Collection of search results.

    Provides convenient access to results and metadata.
    """

    query: SearchQuery
    results: List[SearchResult]
    total_searched: int = 0
    search_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.results)

    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get highest-scoring result."""
        return self.results[0] if self.results else None

    @property
    def avg_relevance(self) -> float:
        """Average relevance score."""
        if not self.results:
            return 0.0
        return sum(r.relevance_score for r in self.results) / len(self.results)

    def get_rules(self) -> List[LegalRule]:
        """Extract just the rules from results."""
        return [r.rule for r in self.results]

    def filter_by_score(self, min_score: float) -> "SearchResults":
        """Filter results by minimum relevance score."""
        filtered = [r for r in self.results if r.relevance_score >= min_score]
        return SearchResults(
            query=self.query,
            results=filtered,
            total_searched=self.total_searched,
            search_time_ms=self.search_time_ms,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": str(self.query),
            "count": self.count,
            "avg_relevance": self.avg_relevance,
            "search_time_ms": self.search_time_ms,
            "results": [
                {
                    "rule_id": r.rule_id,
                    "asp_rule": r.asp_rule[:100] + "...",
                    "relevance_score": r.relevance_score,
                    "matched_keywords": r.matched_keywords,
                }
                for r in self.results[:5]  # Top 5
            ],
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"SearchResults({self.count} results, "
            f"avg_relevance={self.avg_relevance:.2f})"
        )


@dataclass
class SearchStatistics:
    """
    Statistics about search usage and performance.

    Tracks which rules are being retrieved most often.
    """

    total_searches: int = 0
    avg_results_per_search: float = 0.0
    avg_search_time_ms: float = 0.0
    most_searched_domains: Dict[str, int] = field(default_factory=dict)
    most_retrieved_rules: Dict[str, int] = field(default_factory=dict)
    avg_relevance_score: float = 0.0

    def __str__(self) -> str:
        """String representation."""
        return (
            f"SearchStatistics({self.total_searches} searches, "
            f"{self.avg_search_time_ms:.1f}ms avg)"
        )
