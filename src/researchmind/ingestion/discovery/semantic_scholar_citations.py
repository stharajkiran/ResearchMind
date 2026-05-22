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

    def _fetch_ids(self, url: str, key: str) -> list[str]:
        arxiv_ids: list[str] = []
        offset = 0
        limit = 500

        while True:
            try:
                r = httpx.get(
                    url,
                    params={"fields": "externalIds", "limit": limit, "offset": offset},
                    headers=headers(),
                    timeout=30.0,
                )
                if not r.is_success:
                    break
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
            except Exception:
                logger.warning("SS citation fetch failed at offset=%d url=%s", offset, url)
                break

        return arxiv_ids
