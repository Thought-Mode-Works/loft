"""
Rule search engine for retrieving relevant rules.

Combines database queries with relevance scoring to find the best rules
for a given query.

Issue #275: Rule Retrieval and Search
"""

import logging
import time
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import or_

from loft.knowledge.models import LegalRule
from loft.knowledge.search.schemas import SearchQuery, SearchResults
from loft.knowledge.search.scorer import RelevanceScorer

if TYPE_CHECKING:
    from loft.knowledge.database import KnowledgeDatabase

logger = logging.getLogger(__name__)


class RuleSearchEngine:
    """
    Search engine for finding relevant legal rules.

    Combines:
    - Database filtering (domain, jurisdiction, confidence thresholds)
    - Relevance scoring (text, domain, confidence, performance)
    - Result ranking and limiting

    Future enhancements:
    - Semantic embedding search
    - Query expansion
    - Learning from search results
    """

    def __init__(
        self,
        knowledge_db: "KnowledgeDatabase",
        scorer: Optional[RelevanceScorer] = None,
    ):
        """
        Initialize search engine.

        Args:
            knowledge_db: Knowledge database instance
            scorer: Relevance scorer (uses default if not provided)
        """
        self.db = knowledge_db
        self.scorer = scorer or RelevanceScorer()

    def search(self, query: SearchQuery) -> SearchResults:
        """
        Search for relevant rules.

        Args:
            query: Search query with filters and parameters

        Returns:
            SearchResults with scored and ranked results
        """
        start_time = time.time()

        logger.info(f"Searching for rules: {query}")

        # Get candidate rules from database
        candidates = self._retrieve_candidates(query)

        logger.debug(f"Retrieved {len(candidates)} candidate rules")

        # Score each candidate
        scored_results = []
        for rule in candidates:
            result = self.scorer.score(rule, query)
            scored_results.append(result)

        # Filter by minimum confidence if specified
        if query.min_confidence > 0.0:
            scored_results = [
                r for r in scored_results if r.rule.confidence >= query.min_confidence
            ]

        # Sort by relevance score (descending)
        scored_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Limit to max_results
        scored_results = scored_results[: query.max_results]

        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Search complete: {len(scored_results)} results in {search_time_ms:.1f}ms"
        )

        return SearchResults(
            query=query,
            results=scored_results,
            total_searched=len(candidates),
            search_time_ms=search_time_ms,
        )

    def _retrieve_candidates(self, query: SearchQuery) -> List[LegalRule]:
        """
        Retrieve candidate rules from database.

        Applies basic filters before scoring.

        Args:
            query: Search query

        Returns:
            List of candidate rules
        """
        with self.db.SessionLocal() as session:
            # Start with base query
            db_query = session.query(LegalRule)

            # Filter by active/archived status
            if not query.include_archived:
                db_query = db_query.filter(LegalRule.is_archived.is_(False))

            # Filter by domain if specified
            if query.domain:
                db_query = db_query.filter(LegalRule.domain == query.domain)

            # Filter by jurisdiction if specified
            if query.jurisdiction:
                db_query = db_query.filter(
                    LegalRule.jurisdiction == query.jurisdiction
                )

            # Filter by doctrine if specified
            if query.doctrine:
                db_query = db_query.filter(LegalRule.doctrine == query.doctrine)

            # Text search on keywords
            if query.keywords:
                # Build OR conditions for keyword search
                keyword_conditions = []
                for keyword in query.keywords:
                    keyword_lower = keyword.lower()
                    keyword_conditions.append(
                        LegalRule.asp_rule.ilike(f"%{keyword_lower}%")
                    )
                    keyword_conditions.append(
                        LegalRule.reasoning.ilike(f"%{keyword_lower}%")
                    )

                # Apply keyword filter (OR of all conditions)
                db_query = db_query.filter(or_(*keyword_conditions))

            # Predicate search if specified
            if query.predicates:
                predicate_conditions = []
                for predicate in query.predicates:
                    # Search for predicate followed by opening paren
                    pattern = f"%{predicate}(%"
                    predicate_conditions.append(LegalRule.asp_rule.ilike(pattern))

                # Apply predicate filter (OR of all conditions)
                db_query = db_query.filter(or_(*predicate_conditions))

            # Execute query and return results
            candidates = db_query.all()

            # Detach from session to use outside context
            for rule in candidates:
                session.expunge(rule)

            return candidates

    def search_by_text(
        self,
        text: str,
        domain: Optional[str] = None,
        max_results: int = 10,
    ) -> SearchResults:
        """
        Convenience method for simple text search.

        Args:
            text: Search text
            domain: Optional domain filter
            max_results: Maximum number of results

        Returns:
            SearchResults
        """
        query = SearchQuery(
            query_text=text,
            domain=domain,
            max_results=max_results,
        )
        return self.search(query)

    def search_by_predicates(
        self,
        predicates: List[str],
        domain: Optional[str] = None,
        max_results: int = 10,
    ) -> SearchResults:
        """
        Search for rules using specific predicates.

        Args:
            predicates: List of predicate names to search for
            domain: Optional domain filter
            max_results: Maximum number of results

        Returns:
            SearchResults
        """
        query = SearchQuery(
            query_text=" ".join(predicates),  # Combined for explanation
            predicates=predicates,
            domain=domain,
            max_results=max_results,
        )
        return self.search(query)

    def find_similar_rules(
        self,
        rule: LegalRule,
        max_results: int = 5,
    ) -> SearchResults:
        """
        Find rules similar to a given rule.

        Uses the rule's domain, predicates, and keywords.

        Args:
            rule: Rule to find similar rules for
            max_results: Maximum number of results

        Returns:
            SearchResults with similar rules
        """
        # Extract predicates from rule
        import re

        predicates = []
        if rule.asp_rule:
            # Find all predicates (word followed by opening paren)
            matches = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", rule.asp_rule.lower())
            predicates = list(set(matches))  # Remove duplicates

        # Create query
        query = SearchQuery(
            query_text=rule.asp_rule or "",
            domain=rule.domain,
            jurisdiction=rule.jurisdiction,
            doctrine=rule.doctrine,
            predicates=predicates,
            max_results=max_results + 1,  # +1 to account for the rule itself
        )

        # Search
        results = self.search(query)

        # Filter out the original rule
        results.results = [r for r in results.results if r.rule_id != rule.rule_id]

        # Limit to max_results
        results.results = results.results[:max_results]

        return results

    def search_by_domain(
        self,
        domain: str,
        min_confidence: float = 0.7,
        max_results: int = 50,
    ) -> SearchResults:
        """
        Retrieve all rules for a specific domain.

        Args:
            domain: Domain name
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results

        Returns:
            SearchResults for domain
        """
        query = SearchQuery(
            query_text="",  # Empty query to match all rules in domain
            keywords=[],  # No keyword filtering
            domain=domain,
            min_confidence=min_confidence,
            max_results=max_results,
            boost_performance=True,  # Prioritize validated rules
        )
        return self.search(query)

    def get_top_rules(
        self,
        domain: Optional[str] = None,
        max_results: int = 10,
    ) -> List[LegalRule]:
        """
        Get top-performing rules.

        Ranked by performance (success rate, validation count, usage).

        Args:
            domain: Optional domain filter
            max_results: Maximum number of results

        Returns:
            List of top rules
        """
        query = SearchQuery(
            query_text="",  # Empty text, rely on performance scoring
            domain=domain,
            max_results=max_results,
            boost_performance=True,
            boost_confidence=True,
        )

        results = self.search(query)
        return results.get_rules()
