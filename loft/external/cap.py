"""
Caselaw Access Project (CAP) API client.

Provides access to all official U.S. case law from 1658-2018 digitized
from Harvard Law Library collection.

API Documentation: https://case.law/api/
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


class CaselawAccessProjectClient(LegalAPIProvider):
    """
    Client for Caselaw Access Project API.

    Provides access to:
    - Historical case law (1658-2018)
    - Full text opinions
    - Citation networks
    - Jurisdiction-specific collections
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize CAP client.

        Args:
            api_key: CAP API key (required for full-text access)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        base_url = "https://api.case.law/v1"
        super().__init__(api_key, base_url, timeout, max_retries)

        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Token {api_key}"})

    def search(self, query: SearchQuery) -> SearchResult:
        """
        Search CAP cases.

        Args:
            query: Search parameters

        Returns:
            SearchResult with matching cases
        """
        start_time = time.time()

        try:
            params = self._build_search_params(query)

            url = f"{self.base_url}/cases/"
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            documents = self._parse_search_results(data.get("results", []))

            search_time = (time.time() - start_time) * 1000

            result = SearchResult(
                query=query,
                documents=documents,
                total_results=data.get("count", 0),
                page=query.offset // query.max_results + 1,
                has_more=data.get("next") is not None,
                search_time_ms=search_time,
                source_api="cap",
            )

            logger.info(
                f"CAP search returned {len(documents)} results (total: {result.total_results})"
            )

            return result

        except Exception as e:
            self._handle_error(e, "during search")

    def get_document(self, document_id: str) -> CaseLawDocument:
        """
        Get full case document.

        Args:
            document_id: CAP case ID

        Returns:
            CaseLawDocument with full content
        """
        try:
            url = f"{self.base_url}/cases/{document_id}/"
            params = {}
            if self.api_key:
                params["full_case"] = "true"

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            document = self._parse_case(data)

            logger.info(f"Retrieved CAP document {document_id}: {document.case_name}")

            return document

        except Exception as e:
            self._handle_error(e, f"retrieving document {document_id}")

    def get_citations(self, document_id: str) -> List[str]:
        """
        Get citations from a case.

        Args:
            document_id: CAP case ID

        Returns:
            List of citation strings
        """
        try:
            url = f"{self.base_url}/cases/{document_id}/citations/"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            return [cite.get("cite", "") for cite in data.get("results", [])]

        except Exception as e:
            self._handle_error(e, f"retrieving citations for {document_id}")

    def get_by_citation(self, citation: str) -> Optional[CaseLawDocument]:
        """
        Find case by citation.

        Args:
            citation: Legal citation (e.g., "1 U.S. 1")

        Returns:
            CaseLawDocument if found
        """
        try:
            query = SearchQuery(query_text=citation, max_results=1)
            result = self.search(query)

            if result.documents:
                return result.documents[0]
            return None

        except Exception as e:
            logger.warning(f"Error searching by citation '{citation}': {e}")
            return None

    def get_jurisdictions(self) -> List[Dict[str, Any]]:
        """
        Get list of available jurisdictions.

        Returns:
            List of jurisdiction metadata
        """
        try:
            url = f"{self.base_url}/jurisdictions/"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            return response.json().get("results", [])

        except Exception as e:
            logger.warning(f"Error retrieving jurisdictions: {e}")
            return []

    def get_courts(self, jurisdiction: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of courts.

        Args:
            jurisdiction: Optional jurisdiction filter

        Returns:
            List of court metadata
        """
        try:
            url = f"{self.base_url}/courts/"
            params = {}
            if jurisdiction:
                params["jurisdiction"] = jurisdiction

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            return response.json().get("results", [])

        except Exception as e:
            logger.warning(f"Error retrieving courts: {e}")
            return []

    def _build_search_params(self, query: SearchQuery) -> Dict[str, Any]:
        """Build API parameters from search query."""
        params = {
            "search": query.query_text,
            "page_size": query.max_results,
            "offset": query.offset,
        }

        if self.api_key:
            params["full_case"] = "true"

        if query.jurisdiction:
            params["jurisdiction"] = query.jurisdiction

        if query.court:
            params["court"] = query.court

        if query.date_from:
            params["decision_date_min"] = query.date_from.strftime("%Y-%m-%d")

        if query.date_to:
            params["decision_date_max"] = query.date_to.strftime("%Y-%m-%d")

        if query.docket_number:
            params["docket_number"] = query.docket_number

        return params

    def _parse_search_results(self, results: List[Dict[str, Any]]) -> List[CaseLawDocument]:
        """Parse search results into CaseLawDocuments."""
        documents = []

        for result in results:
            try:
                doc = self._parse_case(result)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error parsing CAP result: {e}")
                continue

        return documents

    def _parse_case(self, data: Dict[str, Any]) -> CaseLawDocument:
        """Parse case data into CaseLawDocument."""
        # Extract date
        decision_date = None
        if data.get("decision_date"):
            try:
                decision_date = datetime.fromisoformat(data["decision_date"])
            except (ValueError, TypeError):
                pass

        # Extract text (different formats)
        text = ""
        casebody = data.get("casebody", {})
        if casebody.get("data"):
            # Full case data available
            if isinstance(casebody["data"], dict):
                # Extract text from opinions
                opinions = casebody["data"].get("opinions", [])
                text = "\n\n".join([op.get("text", "") for op in opinions])
            elif isinstance(casebody["data"], str):
                text = casebody["data"]

        # Extract headnotes
        headnotes = []
        if casebody.get("data") and isinstance(casebody["data"], dict):
            head_matter = casebody["data"].get("head_matter", "")
            if head_matter:
                headnotes = [head_matter]

        # Extract citations
        citations = []
        for cite in data.get("citations", []):
            if isinstance(cite, dict):
                citations.append(cite.get("cite", ""))
            elif isinstance(cite, str):
                citations.append(cite)

        # Court level
        court_level = self._determine_court_level(data.get("court", {}).get("name", ""))

        document = CaseLawDocument(
            document_id=str(data.get("id", "")),
            source_api="cap",
            url=data.get("url", ""),
            case_name=data.get("name", "Unknown Case"),
            court=data.get("court", {}).get("name", ""),
            court_level=court_level,
            docket_number=data.get("docket_number"),
            decision_date=decision_date,
            text=text,
            summary="",  # CAP doesn't provide summaries
            headnotes=headnotes,
            citations=citations,
            cited_cases=[],  # Would need separate API call
            judges=self._extract_judges(casebody),
            attorneys=[],
            jurisdiction=data.get("jurisdiction", {}).get("name", ""),
            document_type=DocumentType.OPINION,
            raw_data=data,
        )

        return document

    def _determine_court_level(self, court_name: str) -> Optional[CourtLevel]:
        """Determine court level from court name."""
        court_lower = court_name.lower()

        if "supreme" in court_lower:
            return CourtLevel.SUPREME
        elif "appellate" in court_lower or "appeals" in court_lower or "circuit" in court_lower:
            return CourtLevel.APPELLATE
        elif "district" in court_lower or "superior" in court_lower:
            return CourtLevel.DISTRICT
        elif "bankruptcy" in court_lower:
            return CourtLevel.BANKRUPTCY

        return CourtLevel.STATE

    def _extract_judges(self, casebody: Dict[str, Any]) -> List[str]:
        """Extract judge names from casebody."""
        judges = []

        if casebody.get("data") and isinstance(casebody["data"], dict):
            opinions = casebody["data"].get("opinions", [])
            for opinion in opinions:
                author = opinion.get("author")
                if author:
                    judges.append(author)

        return judges
