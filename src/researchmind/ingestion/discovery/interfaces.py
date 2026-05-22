from abc import ABC, abstractmethod
from datetime import date

from researchmind.ingestion.models import RawPaper


class PaperSource(ABC):
    """Fetch paper metadata from an external source."""

    @abstractmethod
    def fetch_by_query(
        self,
        categories: list[str],
        start_date: date,
        end_date: date,
        max_results: int = 1000,
    ) -> list[RawPaper]:
        """Fetch papers matching categories/keywords within a date range."""
        ...

    @abstractmethod
    def fetch_by_ids(self, paper_ids: list[str]) -> list[RawPaper]:
        """Fetch specific papers by their arXiv IDs."""
        ...


class CitationSource(ABC):
    """Fetch citation relationships for a paper."""

    @abstractmethod
    def get_references(self, paper_id: str) -> list[str]:
        """Return arXiv IDs of papers this paper cites (outbound edges)."""
        ...

    @abstractmethod
    def get_citations(self, paper_id: str) -> list[str]:
        """Return arXiv IDs of papers that cite this paper (inbound edges)."""
        ...


class PaperEnricher(ABC):
    """Add source-specific metadata to existing RawPaper objects."""

    @abstractmethod
    def enrich(self, papers: list[RawPaper]) -> list[RawPaper]:
        """Return enriched copies of the input papers. Input papers are not mutated."""
        ...


class PaperIDSource(ABC):
    """Discovers arXiv IDs from an external source without fetching full metadata.

    Returns IDs only so multiple sources can be combined before a single
    ArxivSource.fetch_by_ids call:

        ids = set()
        ids.update(ss_search.collect_ids(queries))
        ids.update(ss_recs.collect_ids(seed_ids))
        papers = ArxivSource().fetch_by_ids(list(ids))

    The meaning of `identifiers` differs per implementation:
      SemanticScholarSearch       — list of keyword query strings
      SemanticScholarRecommendationSource — list of SS paper IDs (seeds)
    """

    @abstractmethod
    def collect_ids(self, identifiers: list[str]) -> list[str]:
        """Return deduplicated arXiv IDs discovered from the given identifiers."""
        ...
