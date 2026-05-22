"""SemanticScholarSearch — discover arXiv IDs via SS keyword search."""

import logging
import time

import httpx
from tqdm import tqdm

from researchmind.ingestion.discovery._client import _BASE, headers
from researchmind.ingestion.discovery.interfaces import PaperIDSource

logger = logging.getLogger(__name__)

_SEARCH_FIELDS = "externalIds,title,year"
_PAGE_SIZE = 100


class SemanticScholarSearch(PaperIDSource):
    """Discovers arXiv IDs via Semantic Scholar keyword search.

    Takes a list of query strings, runs each separately with token-based
    pagination, and returns deduplicated arXiv IDs. Callers then use
    ArxivSource.fetch_by_ids to get full paper metadata with abstracts.

    collect_ids(queries): each string is a separate SS search query.
                          Results are pooled and deduplicated across queries.
    """

    def collect_ids(
        self,
        identifiers: list[str],
        max_per_query: int = 200,
    ) -> list[str]:
        """Return deduplicated arXiv IDs for the given keyword queries.

        Args:
            identifiers: keyword query strings (e.g. ["anomaly detection", "OOD"]).
            max_per_query: target arXiv IDs per query (not SS result count).
        """
        all_ids: set[str] = set()

        for query in identifiers:
            logger.info("SS search: %r (target %d IDs)", query, max_per_query)
            token = None
            ids_this_query = 0

            with tqdm(total=max_per_query, desc=f"SS: {query}") as pbar:
                while ids_this_query < max_per_query:
                    params: dict = {
                        "query": query,
                        "fields": _SEARCH_FIELDS,
                        "limit": _PAGE_SIZE,
                    }
                    if token:
                        params["token"] = token

                    try:
                        r = httpx.get(
                            f"{_BASE}/paper/search",
                            params=params,
                            headers=headers(),
                            timeout=30.0,
                        )
                        r.raise_for_status()
                        data = r.json()
                    except Exception:
                        logger.exception("SS search failed for query %r", query)
                        break

                    papers = data.get("data", [])
                    if not papers:
                        break

                    for paper in papers:
                        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
                        if arxiv_id and arxiv_id not in all_ids:
                            all_ids.add(arxiv_id)
                            ids_this_query += 1
                            pbar.update(1)
                        if ids_this_query >= max_per_query:
                            break

                    token = data.get("token")
                    if not token:
                        break

                    time.sleep(1.0)

            logger.info("Query %r: %d IDs collected", query, ids_this_query)

        logger.info("collect_ids: %d unique arXiv IDs across %d queries", len(all_ids), len(identifiers))
        return list(all_ids)
