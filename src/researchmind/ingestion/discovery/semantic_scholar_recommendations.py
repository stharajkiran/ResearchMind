"""SemanticScholarRecommendationSource — discover arXiv IDs via SS recommendations."""

import logging

import httpx

from researchmind.ingestion.discovery._client import _RECOMMEND_BASE, headers
from researchmind.ingestion.discovery.interfaces import PaperIDSource

logger = logging.getLogger(__name__)


class SemanticScholarRecommendationSource(PaperIDSource):
    """Discovers arXiv IDs via the SS Recommendations API.

    collect_ids(seed_ss_ids): takes Semantic Scholar paper IDs as seeds (not arXiv IDs),
                              returns arXiv IDs of recommended papers. Callers then use
                              ArxivSource.fetch_by_ids to get full paper metadata.
    """

    def __init__(self, max_recommendations: int = 500) -> None:
        self._max_recommendations = max_recommendations

    def collect_ids(self, identifiers: list[str]) -> list[str]:
        """Return arXiv IDs recommended by SS based on the given SS paper IDs (seeds).

        Args:
            identifiers: Semantic Scholar paper IDs used as positive seeds.
        """
        try:
            r = httpx.post(
                f"{_RECOMMEND_BASE}/papers",
                headers=headers(),
                json={"positivePaperIds": identifiers, "negativePaperIds": []},
                params={"fields": "externalIds", "limit": self._max_recommendations},
                timeout=30.0,
            )
            r.raise_for_status()
            arxiv_ids = [
                aid
                for paper in r.json().get("recommendedPapers", [])
                if (aid := (paper.get("externalIds") or {}).get("ArXiv"))
            ]
            logger.info(
                "collect_ids: %d arXiv IDs from %d seeds",
                len(arxiv_ids), len(identifiers),
            )
            return arxiv_ids
        except Exception:
            logger.exception("SS recommendations fetch failed")
            return []
