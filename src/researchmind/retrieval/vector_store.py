from abc import ABC, abstractmethod

from researchmind.embedding.models import MPNetEncoder
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.chroma_store import ChromaStore
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.rrf import reciprocal_rank_fusion
from researchmind.retrieval.temporal import apply_recency_decay


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
