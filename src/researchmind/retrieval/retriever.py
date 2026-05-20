import json
import logging
from pathlib import Path

from researchmind.embedding.models import BaseResearchEncoder
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.interfaces import DenseIndex, FilteredStore, SparseIndex
from researchmind.retrieval.query_intelligence import QueryTransformer
from researchmind.retrieval.rrf import reciprocal_rank_fusion
from researchmind.retrieval.temporal import apply_recency_decay
from researchmind.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrieverService(VectorStore):
    def __init__(
        self,
        dense: DenseIndex,
        sparse: SparseIndex,
        filtered: FilteredStore,
        encoder: BaseResearchEncoder,
        chunks_path: Path,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._filtered = filtered
        self._encoder = encoder
        self._query_transformer = QueryTransformer()
        self._chunk_dict = self._load_chunk_dict(chunks_path)
        logger.info("RetrieverService initialised with %d chunks.", len(self._chunk_dict))

    def _load_chunk_dict(self, chunks_path: Path) -> dict[str, dict]:
        if not chunks_path.exists():
            logger.warning("Chunks file not found at %s.", chunks_path)
            return {}
        chunk_dict: dict[str, dict] = {}
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                chunk_dict.setdefault(c["chunk_id"], c)
        logger.info("Loaded %d chunks from %s.", len(chunk_dict), chunks_path)
        return chunk_dict

    # ── VectorStore interface ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "standard",
        filters: dict | None = None,
        recency_decay_rate: float | None = None,
    ) -> list[Chunk]:
        if filters:
            return self._filtered.query(query, k, filters)

        bm25_results = self._sparse.search(query, k=k)

        if mode == "rewrite":
            query = self._query_transformer.rewrite(query)
        elif mode == "hyde":
            query = self._query_transformer.hyde(query)

        q_embedding = self._encoder.encode([query])
        faiss_results = self._dense.search(q_embedding, k=k)

        rrf_results = reciprocal_rank_fusion(faiss_results, bm25_results)[:k]

        if recency_decay_rate is not None:
            rrf_results = apply_recency_decay(
                rrf_results, self._chunk_dict, recency_decay_rate
            )

        return [
            Chunk(**self._chunk_dict[chunk_id])
            for chunk_id in rrf_results
            if chunk_id in self._chunk_dict
        ]

    def get_chunks_for_papers(self, paper_ids: list[str]) -> list[Chunk]:
        return [
            Chunk(**c)
            for c in self._chunk_dict.values()
            if c["paper_id"] in set(paper_ids)
        ]

    # ── Properties used by api/app.py and guardrails ─────────────────────────

    @property
    def lookup_paper_metadata(self) -> dict[str, dict]:
        metadata: dict[str, dict] = {}
        for chunk in self._chunk_dict.values():
            pid = chunk["paper_id"]
            if pid not in metadata:
                metadata[pid] = {
                    "title": chunk.get("title", ""),
                    "authors": chunk.get("authors", []),
                    "year": chunk.get("year", 0),
                }
        return metadata

    @property
    def corpus_paper_ids(self) -> set[str]:
        return {chunk["paper_id"] for chunk in self._chunk_dict.values()}

    @property
    def encoder(self) -> BaseResearchEncoder:
        return self._encoder
