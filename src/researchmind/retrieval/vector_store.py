"""Abstract interface for corpus retrieval services."""

from abc import ABC, abstractmethod

from researchmind.ingestion.models import Chunk


class VectorStore(ABC):

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "standard",
        filters: dict | None = None,
        recency_decay_rate: float | None = None,
    ) -> list[Chunk]: ...

    @abstractmethod
    def get_chunks_for_papers(
        self, paper_ids: list[str], max_per_paper: int | None = None
    ) -> list[Chunk]: ...

    @abstractmethod
    def get_relevant_chunks_for_papers(
        self,
        paper_ids: list[str],
        query: str,
        max_per_paper: int = 2,
    ) -> list[Chunk]: ...

    @abstractmethod
    def resolve_paper_id_by_title(
        self, title: str, author_hint: str | None = None
    ) -> str | None:
        """Resolve a unique corpus paper ID from an exact normalized title."""
