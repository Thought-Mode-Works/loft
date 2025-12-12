"""
CourtListener API client.

Provides access to millions of legal opinions from federal and state courts,
oral arguments, and docket information via the Free Law Project API.

API Documentation: https://www.courtlistener.com/api/rest-info/
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
import requests

from loft.external.base import (
    LegalAPIProvider,
    CaseLawDocument,
    SearchQuery,
    SearchResult,
    CourtLevel,
    DocumentType,
)


class CourtListenerClient(LegalAPIProvider):
    """
    Client for CourtListener API.

    Provides access to:
    - Federal and state court opinions
    - Oral arguments
    - Docket information
    - RECAP Archive (federal court documents)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize CourtListener client.

        Args:
            api_key: CourtListener API key (optional for read-only access)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        base_url = "https://www.courtlistener.com/api/rest/v3"
        super().__init__(api_key, base_url, timeout, max_retries)

        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Token {api_key}"})

    def search(self, query: SearchQuery) -> SearchResult:
        """
        Search CourtListener opinions.

        Args:
            query: Search parameters

        Returns:
            SearchResult with matching opinions

        Raises:
            APIError: If search fails
        """
        start_time = time.time()

        try:
            # Build search parameters
            params = self._build_search_params(query)

            # Make API request
            url = f"{self.base_url}/search/"
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Parse results
            documents = self._parse_search_results(data.get("results", []))

            search_time = (time.time() - start_time) * 1000

            result = SearchResult(
                query=query,
                documents=documents,
                total_results=data.get("count", 0),
                page=query.offset // query.max_results + 1,
                has_more=data.get("next") is not None,
                search_time_ms=search_time,
                source_api="courtlistener",
            )

            logger.info(
                f"CourtListener search returned {len(documents)} results "
                f"(total: {result.total_results})"
            )

            return result

        except Exception as e:
            self._handle_error(e, "during search")

    def get_document(self, document_id: str) -> CaseLawDocument:
        """
        Get full opinion document.

        Args:
            document_id: Opinion ID from CourtListener

        Returns:
            CaseLawDocument with full content
        """
        try:
            url = f"{self.base_url}/opinions/{document_id}/"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            document = self._parse_opinion(data)

            logger.info(f"Retrieved document {document_id}: {document.case_name}")

            return document

        except Exception as e:
            self._handle_error(e, f"retrieving document {document_id}")

    def get_citations(self, document_id: str) -> List[str]:
        """
        Get citations from an opinion.

        Args:
            document_id: Opinion ID

        Returns:
            List of citation strings
        """
        try:
            document = self.get_document(document_id)
            return document.citations

        except Exception as e:
            self._handle_error(e, f"retrieving citations for {document_id}")

    def search_by_citation(self, citation: str) -> Optional[CaseLawDocument]:
        """
        Find case by citation (e.g., "410 U.S. 113").

        Args:
            citation: Legal citation string

        Returns:
            CaseLawDocument if found, None otherwise
        """
        try:
            query = SearchQuery(query_text=f'citation:"{citation}"', max_results=1)
            result = self.search(query)

            if result.documents:
                return result.documents[0]
            return None

        except Exception as e:
            logger.warning(f"Error searching by citation '{citation}': {e}")
            return None

    def get_docket(self, docket_id: str) -> Dict[str, Any]:
        """
        Get docket information.

        Args:
            docket_id: Docket ID from CourtListener

        Returns:
            Dictionary with docket information
        """
        try:
            url = f"{self.base_url}/dockets/{docket_id}/"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            self._handle_error(e, f"retrieving docket {docket_id}")

    def _build_search_params(self, query: SearchQuery) -> Dict[str, Any]:
        """Build API parameters from search query."""
        params = {
            "q": query.query_text,
            "page_size": query.max_results,
            "offset": query.offset,
            "type": "o",  # opinions
        }

        if query.court:
            params["court"] = query.court

        if query.date_from:
            params["filed_after"] = query.date_from.strftime("%Y-%m-%d")

        if query.date_to:
            params["filed_before"] = query.date_to.strftime("%Y-%m-%d")

        if query.cited_case:
            params["cites"] = query.cited_case

        if query.judge:
            params["judge"] = query.judge

        return params

    def _parse_search_results(
        self, results: List[Dict[str, Any]]
    ) -> List[CaseLawDocument]:
        """Parse search results into CaseLawDocuments."""
        documents = []

        for result in results:
            try:
                doc = self._parse_opinion(result)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error parsing search result: {e}")
                continue

        return documents

    def _parse_opinion(self, data: Dict[str, Any]) -> CaseLawDocument:
        """Parse opinion data into CaseLawDocument."""
        # Extract date
        decision_date = None
        if data.get("date_filed"):
            try:
                decision_date = datetime.fromisoformat(
                    data["date_filed"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError, AttributeError):
                pass

        # Extract text
        text = (
            data.get("html", "")
            or data.get("plain_text", "")
            or data.get("html_lawbox", "")
        )

        # Extract citations
        citations = []
        for cite_type in ["citation", "neutral_cite", "lexis_cite"]:
            if data.get(cite_type):
                citations.append(data[cite_type])

        # Determine court level
        court_level = self._determine_court_level(data.get("court", ""))

        document = CaseLawDocument(
            document_id=str(data.get("id", "")),
            source_api="courtlistener",
            url=data.get("absolute_url", ""),
            case_name=data.get("case_name", "Unknown Case"),
            court=data.get("court", ""),
            court_level=court_level,
            docket_number=data.get("docket_number"),
            decision_date=decision_date,
            text=text,
            summary=data.get("summary", ""),
            headnotes=[],
            citations=citations,
            cited_cases=[],
            judges=self._extract_judges(data),
            attorneys=(
                data.get("attorneys", [])
                if isinstance(data.get("attorneys"), list)
                else []
            ),
            jurisdiction=data.get("jurisdiction", ""),
            document_type=DocumentType.OPINION,
            relevance_score=data.get("score", 0.0) if "score" in data else 0.0,
            raw_data=data,
        )

        return document

    def _determine_court_level(self, court: str) -> Optional[CourtLevel]:
        """Determine court level from court identifier."""
        court_lower = court.lower()

        if "scotus" in court_lower or "supreme" in court_lower:
            return CourtLevel.SUPREME
        elif (
            "appellate" in court_lower
            or "circuit" in court_lower
            or court_lower.startswith("ca")
        ):
            return CourtLevel.APPELLATE
        elif "district" in court_lower:
            return CourtLevel.DISTRICT
        elif "bankruptcy" in court_lower:
            return CourtLevel.BANKRUPTCY
        elif "tax" in court_lower:
            return CourtLevel.TAX

        return None

    def _extract_judges(self, data: Dict[str, Any]) -> List[str]:
        """Extract judge names from opinion data."""
        judges = []

        # Try different fields
        if data.get("judges"):
            judges.append(data["judges"])

        if data.get("author_str"):
            judges.append(data["author_str"])

        if data.get("joined_by_str"):
            judges.append(data["joined_by_str"])

        return judges
