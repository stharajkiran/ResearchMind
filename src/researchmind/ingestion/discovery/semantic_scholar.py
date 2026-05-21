import logging
import os
import time
from datetime import date

import httpx

from researchmind.ingestion.discovery.interfaces import CitationSource, PaperEnricher, PaperSource
from researchmind.ingestion.models import RawPaper, S2Author

# forward reference — ArxivSource is imported inside methods to avoid circular imports

logger = logging.getLogger(__name__)

_BASE = "https://api.semanticscholar.org/graph/v1"
_SEARCH_FIELDS = "paperId,externalIds,title,authors,year,publicationDate,fieldsOfStudy,openAccessPdf"
_ENRICH_FIELDS = "paperId,externalIds,citationCount,influentialCitationCount,authors"
_BATCH_SIZE = 500  # SS /paper/batch limit


def _headers() -> dict[str, str]:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return {"x-api-key": key} if key else {}


class SemanticScholarSource(PaperSource):
    """Semantic Scholar implementation of PaperSource.

    fetch_by_query: keyword search — categories are joined as search terms,
                    not arXiv category codes. Useful for domain-specific expansion
                    beyond arXiv category boundaries.
    fetch_by_ids:   fetches paper metadata by arXiv ID via the SS batch endpoint.
                    Note: abstracts may be missing for some papers; prefer ArxivSource
                    when abstracts are required.
    """

    def fetch_by_query(
        self,
        categories: list[str],
        start_date: date,
        end_date: date,
        max_results: int = 1000,
    ) -> list[RawPaper]:
        query = " OR ".join(categories)
        papers: list[RawPaper] = []
        offset = 0
        limit = min(100, max_results)

        while len(papers) < max_results:
            try:
                r = httpx.get(
                    f"{_BASE}/paper/search",
                    params={
                        "query": query,
                        "fields": _SEARCH_FIELDS,
                        "limit": limit,
                        "offset": offset,
                        "year": f"{start_date.year}-{end_date.year}",
                    },
                    headers=_headers(),
                    timeout=30.0,
                )
                r.raise_for_status()
                batch = r.json().get("data", [])
                if not batch:
                    break
                for item in batch:
                    arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
                    if not arxiv_id:
                        continue
                    pub_date = item.get("publicationDate")
                    papers.append(
                        RawPaper(
                            paper_id=arxiv_id,
                            title=item.get("title", ""),
                            authors=[a.get("name", "") for a in item.get("authors", [])],
                            abstract="",  # requires a separate /paper/{id} call
                            categories=item.get("fieldsOfStudy") or [],
                            published=date.fromisoformat(pub_date)
                            if pub_date
                            else date(item.get("year") or 2000, 1, 1),
                            pdf_url=(item.get("openAccessPdf") or {}).get("url") or "",
                        )
                    )
                offset += len(batch)
                if len(batch) < limit:
                    break
                time.sleep(1.0)
            except Exception:
                logger.exception("SemanticScholar search failed at offset=%d", offset)
                break

        logger.info("fetch_by_query: returned %d papers", len(papers))
        return papers[:max_results]

    def fetch_by_ids(self, paper_ids: list[str]) -> list[RawPaper]:
        """Fetch papers by arXiv ID using the SS batch endpoint."""
        if not paper_ids:
            return []
        papers: list[RawPaper] = []
        fetch_fields = f"{_SEARCH_FIELDS},abstract"

        for i in range(0, len(paper_ids), _BATCH_SIZE):
            batch = paper_ids[i : i + _BATCH_SIZE]
            ids = [f"ArXiv:{pid}" for pid in batch]
            try:
                r = httpx.post(
                    f"{_BASE}/paper/batch",
                    params={"fields": fetch_fields},
                    headers=_headers(),
                    json={"ids": ids},
                    timeout=30.0,
                )
                r.raise_for_status()
                for item in (r.json() or []):
                    if not item:
                        continue
                    arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
                    if not arxiv_id:
                        continue
                    pub_date = item.get("publicationDate")
                    papers.append(
                        RawPaper(
                            paper_id=arxiv_id.split("v")[0],
                            title=item.get("title", ""),
                            authors=[a.get("name", "") for a in (item.get("authors") or [])],
                            abstract=item.get("abstract") or "",
                            categories=item.get("fieldsOfStudy") or [],
                            published=date.fromisoformat(pub_date)
                            if pub_date
                            else date(item.get("year") or 2000, 1, 1),
                            pdf_url=(item.get("openAccessPdf") or {}).get("url") or "",
                        )
                    )
                time.sleep(1.0)
            except Exception:
                logger.exception("SS fetch_by_ids batch failed at index %d", i)

        logger.info("fetch_by_ids: returned %d papers for %d requested IDs", len(papers), len(paper_ids))
        return papers


class SemanticScholarCitationSource(CitationSource):
    """Semantic Scholar implementation of CitationSource.

    Uses /paper/{id}/references (outbound) and /paper/{id}/citations (inbound).
    Both directions are required to build a meaningful citation graph.
    """

    def get_references(self, paper_id: str) -> list[str]:
        """Return arXiv IDs of papers this paper cites (outbound edges)."""
        return self._fetch_ids(
            f"{_BASE}/paper/arXiv:{paper_id}/references", key="citedPaper"
        )

    def get_citations(self, paper_id: str) -> list[str]:
        """Return arXiv IDs of papers that cite this paper (inbound edges)."""
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
                    headers=_headers(),
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
                logger.warning("SS citation fetch failed at offset=%d for %s", offset, url)
                break
        return arxiv_ids


class SemanticScholarEnricher(PaperEnricher):
    """Adds SS-specific metadata to existing RawPaper objects.

    Populates: s2_paper_id, citation_count, influential_citation_count, s2_authors.
    Input papers (already fetched from arXiv) are not mutated — enriched copies returned.
    """

    def enrich(self, papers: list[RawPaper]) -> list[RawPaper]:
        if not papers:
            return []

        enriched_map: dict[str, dict] = {}
        for i in range(0, len(papers), _BATCH_SIZE):
            batch = papers[i : i + _BATCH_SIZE]
            ids = [f"ArXiv:{p.paper_id}" for p in batch]
            try:
                r = httpx.post(
                    f"{_BASE}/paper/batch",
                    params={"fields": _ENRICH_FIELDS},
                    headers=_headers(),
                    json={"ids": ids},
                    timeout=30.0,
                )
                r.raise_for_status()
                for item in (r.json() or []):
                    if not item:
                        continue
                    arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
                    if arxiv_id:
                        enriched_map[arxiv_id.split("v")[0]] = item
                time.sleep(1.0)
            except Exception:
                logger.exception("SS enrich batch failed at index %d", i)

        result: list[RawPaper] = []
        enriched_count = 0
        for paper in papers:
            norm_id = paper.paper_id.split("v")[0]
            meta = enriched_map.get(norm_id)
            if meta:
                paper = paper.model_copy(
                    update={
                        "s2_paper_id": meta.get("paperId"),
                        "citation_count": meta.get("citationCount"),
                        "influential_citation_count": meta.get("influentialCitationCount"),
                        "s2_authors": [
                            S2Author(
                                author_id=a.get("authorId") or "",
                                name=a.get("name") or "",
                            )
                            for a in (meta.get("authors") or [])
                        ],
                    }
                )
                enriched_count += 1
            result.append(paper)

        logger.info("enrich: enriched %d / %d papers", enriched_count, len(papers))
        return result


_RECOMMEND_URL = "https://api.semanticscholar.org/recommendations/v1/papers"


class SemanticScholarRecommendationSource(PaperSource):
    """Discovers papers via the SS Recommendations API using seed SS paper IDs.

    fetch_by_ids(seed_ss_ids): seed_ss_ids are Semantic Scholar paper IDs (not arXiv IDs).
                               Calls the recommendations endpoint, collects returned arXiv IDs,
                               then fetches full metadata from arXiv.
    fetch_by_query:            not applicable for this source — logs a warning and returns [].
    """

    def __init__(self, max_recommendations: int = 500) -> None:
        self._max_recommendations = max_recommendations

    def fetch_by_query(
        self,
        categories: list[str],
        start_date: date,
        end_date: date,
        max_results: int = 1000,
    ) -> list[RawPaper]:
        logger.warning(
            "SemanticScholarRecommendationSource does not support fetch_by_query. "
            "Use fetch_by_ids with SS paper IDs as seeds."
        )
        return []

    def fetch_by_ids(self, seed_ss_ids: list[str]) -> list[RawPaper]:
        """Fetch papers recommended by SS based on the given seed SS paper IDs."""
        arxiv_ids = self._get_recommended_arxiv_ids(seed_ss_ids)
        if not arxiv_ids:
            return []
        from researchmind.ingestion.discovery.arxiv import ArxivSource
        return ArxivSource().fetch_by_ids(arxiv_ids)

    def _get_recommended_arxiv_ids(self, seed_ids: list[str]) -> list[str]:
        try:
            r = httpx.post(
                _RECOMMEND_URL,
                headers=_headers(),
                json={"positivePaperIds": seed_ids, "negativePaperIds": []},
                params={"fields": "externalIds", "limit": self._max_recommendations},
                timeout=30.0,
            )
            r.raise_for_status()
            arxiv_ids = [
                arxiv_id
                for paper in r.json().get("recommendedPapers", [])
                if (arxiv_id := (paper.get("externalIds") or {}).get("ArXiv"))
            ]
            logger.info(
                "_get_recommended_arxiv_ids: %d arXiv IDs from %d seeds",
                len(arxiv_ids), len(seed_ids),
            )
            return arxiv_ids
        except Exception:
            logger.exception("SS recommendations fetch failed")
            return []
