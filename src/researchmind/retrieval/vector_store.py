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
    def get_chunks_for_papers(self, paper_ids: list[str]) -> list[Chunk]: ...
