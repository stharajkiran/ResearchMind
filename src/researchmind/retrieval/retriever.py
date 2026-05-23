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
        logger.info(
            "RetrieverService initialised with %d chunks.", len(self._chunk_dict)
        )

    def _load_chunk_dict(self, chunks_path: Path) -> dict[str, dict]:
        if not chunks_path.exists():
            logger.warning("Chunks file not found at %s.", chunks_path)
            return {}
        chunk_dict: dict[str, dict] = {}
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                # Normalize to unversioned arXiv ID ("2301.07041v2" → "2301.07041")
                # so paper_ids match citation graph nodes, which are built without version suffix.
                c["paper_id"] = c["paper_id"].split("v")[0]
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

    def get_chunks_for_papers(
        self, paper_ids: list[str], max_per_paper: int | None = None
    ) -> list[Chunk]:
        """Return chunks for the given paper IDs, optionally capping the number of chunks per paper."""
        seen: dict[str, list[Chunk]] = {}
        for c in self._chunk_dict.values():
            if c["paper_id"] in set(paper_ids):
                seen.setdefault(c["paper_id"], []).append(Chunk(**c))
        result = []
        for pid in paper_ids:
            result.extend(seen.get(pid, [])[:max_per_paper])
        return result

    def get_relevant_chunks_for_papers(
        self,
        paper_ids: list[str],
        query: str,
        max_per_paper: int = 2,
    ) -> list[Chunk]:
        """Return the most query-relevant chunks for the given papers.

        Encodes all candidate chunks in one batch (not per-paper) then ranks
        by cosine similarity within each paper.
        """
        paper_id_set = set(paper_ids)
        q_vec = self._encoder.encode([query], normalize_embeddings=True)  # (1, dim)

        candidates = [c for c in self._chunk_dict.values() if c["paper_id"] in paper_id_set]
        if not candidates:
            return []

        chunk_vecs = self._encoder.encode(
            [c["text"] for c in candidates], normalize_embeddings=True
        )  # (N, dim)
        scores = (chunk_vecs @ q_vec.T).ravel()  # (N,) — ravel handles N=1 without scalar

        paper_scored: dict[str, list[tuple[float, dict]]] = {}
        for score, chunk in zip(scores, candidates):
            paper_scored.setdefault(chunk["paper_id"], []).append((float(score), chunk))

        result = []
        for pid in paper_ids:
            top = sorted(paper_scored.get(pid, []), key=lambda x: -x[0])[:max_per_paper]
            result.extend(Chunk(**c) for _, c in top)
        return result


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
