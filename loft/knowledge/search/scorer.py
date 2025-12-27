"""
Relevance scoring for search results.

Computes relevance scores based on multiple factors.

Issue #275: Rule Retrieval and Search
"""

import logging
import re
from typing import List

from loft.knowledge.models import LegalRule
from loft.knowledge.search.schemas import SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Calculate relevance scores for search results.

    Combines multiple scoring factors:
    - Text/keyword matching
    - Domain/metadata matching
    - Confidence scores
    - Performance metrics
    - (Optional) Semantic similarity
    """

    def __init__(
        self,
        text_weight: float = 0.35,
        domain_weight: float = 0.20,
        confidence_weight: float = 0.20,
        performance_weight: float = 0.25,
        semantic_weight: float = 0.0,  # Not used unless embeddings available
    ):
        """
        Initialize scorer with weighting factors.

        Args:
            text_weight: Weight for text matching
            domain_weight: Weight for domain/metadata matching
            confidence_weight: Weight for rule confidence
            performance_weight: Weight for historical performance
            semantic_weight: Weight for semantic similarity
        """
        # Normalize weights to sum to 1.0
        total = (
            text_weight
            + domain_weight
            + confidence_weight
            + performance_weight
            + semantic_weight
        )

        self.text_weight = text_weight / total
        self.domain_weight = domain_weight / total
        self.confidence_weight = confidence_weight / total
        self.performance_weight = performance_weight / total
        self.semantic_weight = semantic_weight / total

    def score(self, rule: LegalRule, query: SearchQuery) -> SearchResult:
        """
        Calculate relevance score for a rule given a query.

        Args:
            rule: Rule to score
            query: Search query

        Returns:
            SearchResult with scores
        """
        # Calculate component scores
        text_score = self._calculate_text_score(rule, query)
        domain_score = self._calculate_domain_score(rule, query)
        confidence_score = self._calculate_confidence_score(rule, query)
        performance_score = self._calculate_performance_score(rule, query)

        # Semantic score would go here if embeddings available
        semantic_score = 0.0

        # Calculate weighted overall score
        overall_score = (
            (text_score * self.text_weight)
            + (domain_score * self.domain_weight)
            + (confidence_score * self.confidence_weight)
            + (performance_score * self.performance_weight)
            + (semantic_score * self.semantic_weight)
        )

        # Extract matched keywords
        matched_keywords = self._get_matched_keywords(rule, query)
        matched_predicates = self._get_matched_predicates(rule, query)

        # Generate explanation
        explanation = self._generate_explanation(
            text_score,
            domain_score,
            confidence_score,
            performance_score,
            matched_keywords,
        )

        return SearchResult(
            rule=rule,
            relevance_score=overall_score,
            text_match_score=text_score,
            domain_match_score=domain_score,
            confidence_score=confidence_score,
            performance_score=performance_score,
            semantic_score=semantic_score,
            matched_keywords=matched_keywords,
            matched_predicates=matched_predicates,
            explanation=explanation,
        )

    def _calculate_text_score(self, rule: LegalRule, query: SearchQuery) -> float:
        """
        Calculate text matching score.

        Based on:
        - Keyword matches in rule text
        - Keyword matches in reasoning
        - Exact phrase matches

        Args:
            rule: Rule to score
            query: Search query

        Returns:
            Score between 0.0 and 1.0
        """
        if not query.keywords:
            return 0.5  # Neutral score if no keywords

        rule_text = (rule.asp_rule or "").lower()
        reasoning_text = (rule.reasoning or "").lower()
        combined_text = f"{rule_text} {reasoning_text}"

        # Count keyword matches
        keyword_matches = 0
        keyword_weights = 0

        for keyword in query.keywords:
            keyword_lower = keyword.lower()

            # Exact match in rule (highest weight)
            if keyword_lower in rule_text:
                keyword_matches += 2.0
            # Match in reasoning (medium weight)
            elif keyword_lower in reasoning_text:
                keyword_matches += 1.0

            keyword_weights += 2.0  # Maximum possible per keyword

        if keyword_weights == 0:
            return 0.0

        # Normalize by maximum possible score
        score = keyword_matches / keyword_weights

        # Bonus for exact phrase match
        query_text_lower = query.query_text.lower()
        if len(query_text_lower) > 5 and query_text_lower in combined_text:
            score = min(score + 0.2, 1.0)

        return min(score, 1.0)

    def _calculate_domain_score(self, rule: LegalRule, query: SearchQuery) -> float:
        """
        Calculate domain/metadata matching score.

        Based on:
        - Domain match
        - Doctrine match
        - Jurisdiction match

        Args:
            rule: Rule to score
            query: Search query

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        factors = 0

        # Domain match (most important)
        if query.domain:
            factors += 2
            if rule.domain and rule.domain.lower() == query.domain.lower():
                score += 2.0
        elif rule.domain:
            # Neutral if no domain specified but rule has one
            score += 0.5
            factors += 1

        # Doctrine match
        if query.doctrine:
            factors += 1
            if rule.doctrine and rule.doctrine.lower() == query.doctrine.lower():
                score += 1.0

        # Jurisdiction match
        if query.jurisdiction:
            factors += 1
            if (
                rule.jurisdiction
                and rule.jurisdiction.lower() == query.jurisdiction.lower()
            ):
                score += 1.0

        if factors == 0:
            return 0.5  # Neutral if no domain criteria

        return min(score / factors, 1.0)

    def _calculate_confidence_score(self, rule: LegalRule, query: SearchQuery) -> float:
        """
        Calculate confidence-based score.

        Args:
            rule: Rule to score
            query: Search query

        Returns:
            Score between 0.0 and 1.0
        """
        if not query.boost_confidence:
            return 0.5  # Neutral if not boosting

        confidence = rule.confidence if rule.confidence is not None else 0.5
        return float(confidence)

    def _calculate_performance_score(
        self, rule: LegalRule, query: SearchQuery
    ) -> float:
        """
        Calculate performance-based score.

        Based on:
        - Validation count (how often tested)
        - Success rate (calculated from success/failure counts)

        Args:
            rule: Rule to score
            query: Search query

        Returns:
            Score between 0.0 and 1.0
        """
        if not query.boost_performance:
            return 0.5  # Neutral if not boosting

        score = 0.0
        factors = 0

        # Calculate success rate from counts
        if rule.validation_count and rule.validation_count > 0:
            success_rate = rule.success_count / rule.validation_count
            # Double weight for success rate
            score += success_rate * 2.0
            factors += 2

            # Validation count bonus (normalized to 10)
            validation_score = min(rule.validation_count / 10.0, 1.0)
            score += validation_score
            factors += 1

        if factors == 0:
            return 0.5  # Neutral if no performance data

        return score / factors

    def _get_matched_keywords(self, rule: LegalRule, query: SearchQuery) -> List[str]:
        """
        Get list of query keywords that matched in rule.

        Args:
            rule: Rule
            query: Search query

        Returns:
            List of matched keywords
        """
        matched = []
        rule_text = (rule.asp_rule or "").lower()
        reasoning_text = (rule.reasoning or "").lower()
        combined_text = f"{rule_text} {reasoning_text}"

        for keyword in query.keywords:
            if keyword.lower() in combined_text:
                matched.append(keyword)

        return matched

    def _get_matched_predicates(self, rule: LegalRule, query: SearchQuery) -> List[str]:
        """
        Get list of query predicates that matched in rule.

        Args:
            rule: Rule
            query: Search query

        Returns:
            List of matched predicates
        """
        if not query.predicates:
            return []

        matched = []
        rule_text = (rule.asp_rule or "").lower()

        for predicate in query.predicates:
            predicate_lower = predicate.lower()
            # Match predicate as whole word followed by opening paren
            pattern = rf"\b{re.escape(predicate_lower)}\s*\("
            if re.search(pattern, rule_text):
                matched.append(predicate)

        return matched

    def _generate_explanation(
        self,
        text_score: float,
        domain_score: float,
        confidence_score: float,
        performance_score: float,
        matched_keywords: List[str],
    ) -> str:
        """
        Generate human-readable explanation of score.

        Args:
            text_score: Text match score
            domain_score: Domain match score
            confidence_score: Confidence score
            performance_score: Performance score
            matched_keywords: Keywords that matched

        Returns:
            Explanation string
        """
        parts = []

        # Keyword matches
        if matched_keywords:
            parts.append(f"Matched keywords: {', '.join(matched_keywords[:3])}")

        # Strong scores
        if text_score > 0.7:
            parts.append("Strong text match")
        if domain_score > 0.7:
            parts.append("Domain/metadata match")
        if confidence_score > 0.8:
            parts.append("High confidence")
        if performance_score > 0.7:
            parts.append("Good performance history")

        # Weak scores (warnings)
        if text_score < 0.3:
            parts.append("Weak text match")
        if confidence_score < 0.5:
            parts.append("Low confidence")

        if not parts:
            parts.append("Moderate relevance")

        return "; ".join(parts)
