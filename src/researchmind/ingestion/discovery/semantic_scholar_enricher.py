"""SemanticScholarEnricher — adds citation metadata to arXiv-fetched papers."""

import logging
import time

import httpx

from researchmind.ingestion.discovery._client import _BASE, _BATCH_SIZE, headers
from researchmind.ingestion.discovery.interfaces import PaperEnricher
from researchmind.ingestion.models import RawPaper, S2Author

logger = logging.getLogger(__name__)

_ENRICH_FIELDS = "paperId,externalIds,citationCount,influentialCitationCount,authors"


class SemanticScholarEnricher(PaperEnricher):
    """Adds SS-specific metadata to papers already fetched from arXiv.

    Populates: s2_paper_id, citation_count, influential_citation_count, s2_authors.
    Input papers are not mutated — enriched copies are returned.
    """

    def enrich(self, papers: list[RawPaper]) -> list[RawPaper]:
        if not papers:
            return []

        enriched_map: dict[str, dict] = {}
        for i in range(0, len(papers), _BATCH_SIZE):
            batch = papers[i : i + _BATCH_SIZE]
            try:
                r = httpx.post(
                    f"{_BASE}/paper/batch",
                    params={"fields": _ENRICH_FIELDS},
                    headers=headers(),
                    json={"ids": [f"ARXIV:{p.paper_id.split('v')[0]}" for p in batch]},
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
            meta = enriched_map.get(paper.paper_id.split("v")[0])
            if meta:
                paper = paper.model_copy(update={
                    "s2_paper_id": meta.get("paperId"),
                    "citation_count": meta.get("citationCount"),
                    "influential_citation_count": meta.get("influentialCitationCount"),
                    "s2_authors": [
                        S2Author(author_id=a.get("authorId") or "", name=a.get("name") or "")
                        for a in (meta.get("authors") or [])
                    ],
                })
                enriched_count += 1
            result.append(paper)

        logger.info("enrich: enriched %d / %d papers", enriched_count, len(papers))
        return result
