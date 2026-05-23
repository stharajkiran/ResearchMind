"""SemanticScholarCitationSource — fetch citation edges for the citation graph."""

import logging
import time

import httpx

from researchmind.ingestion.discovery._client import _BASE, headers
from researchmind.ingestion.discovery.interfaces import CitationSource

logger = logging.getLogger(__name__)


class SemanticScholarCitationSource(CitationSource):
    """Fetches citation relationships from Semantic Scholar.

    get_references: papers this paper cites (outbound edges — what it builds on).
    get_citations:  papers that cite this paper (inbound edges — what built on it).

    Both directions are needed to build a meaningful citation graph.
    """

    def get_references(self, paper_id: str) -> list[str]:
        return self._fetch_ids(
            f"{_BASE}/paper/arXiv:{paper_id}/references", key="citedPaper"
        )

    def get_citations(self, paper_id: str) -> list[str]:
        return self._fetch_ids(
            f"{_BASE}/paper/arXiv:{paper_id}/citations", key="citingPaper"
        )

    def _fetch_ids(self, url: str, key: str, max_retries: int = 5) -> list[str]:
        arxiv_ids: list[str] = []
        offset = 0
        limit = 500

        while True:
            for attempt in range(max_retries):
                try:
                    r = httpx.get(
                        url,
                        params={"fields": "externalIds", "limit": limit, "offset": offset},
                        headers=headers(),
                        timeout=30.0,
                    )
                    if r.status_code == 429:
                        wait = int(r.headers.get("Retry-After", 10)) + attempt * 5
                        logger.warning("Rate limited — waiting %ds before retry.", wait)
                        time.sleep(wait)
                        continue
                    if not r.is_success:
                        return arxiv_ids
                    break
                except Exception:
                    logger.warning("SS citation fetch failed at offset=%d url=%s (attempt %d)", offset, url, attempt + 1)
                    time.sleep(2 ** attempt)
            else:
                return arxiv_ids

            batch = r.json().get("data", [])
            if not batch:
                break
            for item in batch:
                paper = item.get(key) or {}
                arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
                if arxiv_id:
                    arxiv_ids.append(arxiv_id.split("v")[0])
            offset += len(batch)
            if len(batch) < limit:
                break
            time.sleep(0.5)

        return arxiv_ids
