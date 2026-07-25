"""Hybrid retrieval service for ResearchMind's configured paper corpus."""

import json
import logging
import re
from pathlib import Path

import numpy as np

from researchmind.embedding.models import BaseResearchEncoder
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.interfaces import DenseIndex, FilteredStore, SparseIndex
from researchmind.retrieval.query_intelligence import QueryTransformer
from researchmind.retrieval.rrf import reciprocal_rank_fusion
from researchmind.retrieval.temporal import apply_recency_decay
from researchmind.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _normalize_paper_title(title: str) -> str:
    """Normalize a paper title for deterministic exact-title matching."""
    normalized = re.sub(r"[^a-z0-9]+", " ", title.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_author_name(author: str) -> str:
    """Normalize an author name for a conservative equality check."""
    return re.sub(r"[^a-z0-9]+", "", author.casefold())


class RetrieverService(VectorStore):
    """Combine sparse and dense retrieval and apply corpus relevance policy.

    Args:
        dense: Dense vector-search backend.
        sparse: Sparse keyword-search backend.
        filtered: Metadata-filtered retrieval backend.
        encoder: Encoder used for query and relevance embeddings.
        chunks_path: JSONL file containing indexed chunk metadata.
        min_relevance_score: Optional cosine-similarity floor for returned chunks.
    """

    def __init__(
        self,
        dense: DenseIndex,
        sparse: SparseIndex,
        filtered: FilteredStore,
        encoder: BaseResearchEncoder,
        chunks_path: Path,
        min_relevance_score: float | None = None,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._filtered = filtered
        self._encoder = encoder
        self._query_transformer = QueryTransformer()
        self._chunk_dict = self._load_chunk_dict(chunks_path)
        self._min_relevance_score = min_relevance_score
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

    def _filter_by_relevance(
        self, chunk_ids: list[str], query_embedding: np.ndarray
    ) -> list[str]:
        """Remove hybrid candidates below the configured cosine-similarity floor.

        The fusion backends use different native score semantics, so this method
        evaluates only the fused candidates with normalized embeddings. A null
        threshold retains the existing behavior unchanged.

        Args:
            chunk_ids: Ranked candidate IDs produced by hybrid retrieval.
            query_embedding: Normalized embedding for the effective search query.

        Returns:
            Candidate IDs whose cosine similarity meets the configured floor.
        """
        if self._min_relevance_score is None or not chunk_ids:
            return chunk_ids

        candidates = [
            (chunk_id, self._chunk_dict[chunk_id]["text"])
            for chunk_id in chunk_ids
            if chunk_id in self._chunk_dict
        ]
        if not candidates:
            return []

        candidate_embeddings = self._encoder.encode(
            [text for _, text in candidates], normalize_embeddings=True
        )
        query_vector = np.asarray(query_embedding[0])
        relevance_scores = candidate_embeddings @ query_vector
        logger.info(
            "Relevance gate: candidates=%d max_score=%.3f threshold=%.3f",
            len(candidates),
            float(np.max(relevance_scores)),
            self._min_relevance_score,
        )
        relevant_ids = [
            chunk_id
            for (chunk_id, _), score in zip(candidates, relevance_scores)
            if float(score) >= self._min_relevance_score
        ]

        if not relevant_ids:
            logger.info(
                "No candidates met relevance threshold %.2f.",
                self._min_relevance_score,
            )
        return relevant_ids

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
        rrf_results = self._filter_by_relevance(rrf_results, q_embedding)

        if recency_decay_rate is not None:
            rrf_results = apply_recency_decay(
                rrf_results, self._chunk_dict, recency_decay_rate
            )

        return [
            Chunk(**self._chunk_dict[chunk_id])
            for chunk_id in rrf_results
            if chunk_id in self._chunk_dict
        ]

    def search_scored(
        self,
        query: str,
        k: int = 10,
        mode: str = "standard",
        recency_decay_rate: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Like :meth:`search` but also returns a normalized relevance score per chunk.

        The score is the fused RRF weight normalized to ``[0, 1]`` against the top
        hit in this result set, so it reflects the retriever's own ranking. Only the
        ``/search`` endpoint uses this; the core :meth:`search` path is unchanged so
        agent, RAG, and MCP callers keep receiving plain ``list[Chunk]``.
        """
        bm25_results = self._sparse.search(query, k=k)

        if mode == "rewrite":
            query = self._query_transformer.rewrite(query)
        elif mode == "hyde":
            query = self._query_transformer.hyde(query)

        q_embedding = self._encoder.encode([query])
        faiss_results = self._dense.search(q_embedding, k=k)

        scored = reciprocal_rank_fusion(
            faiss_results, bm25_results, return_scores=True
        )[:k]
        score_map = dict(scored)
        ranked_ids = [doc_id for doc_id, _ in scored]

        ranked_ids = self._filter_by_relevance(ranked_ids, q_embedding)

        if recency_decay_rate is not None:
            ranked_ids = apply_recency_decay(
                ranked_ids, self._chunk_dict, recency_decay_rate
            )

        # Normalize against the strongest surviving hit so the top result reads as 1.0.
        surviving = [score_map[cid] for cid in ranked_ids if cid in score_map]
        top = max(surviving) if surviving else 0.0

        results: list[tuple[Chunk, float]] = []
        for chunk_id in ranked_ids:
            if chunk_id not in self._chunk_dict:
                continue
            raw = score_map.get(chunk_id, 0.0)
            norm = (raw / top) if top > 0 else 0.0
            results.append((Chunk(**self._chunk_dict[chunk_id]), norm))
        return results

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

    def resolve_paper_id_by_title(
        self, title: str, author_hint: str | None = None
    ) -> str | None:
        """Resolve one corpus paper ID from an exact normalized title.

        Args:
            title: Paper title extracted from the user's citation request.
            author_hint: Optional full author name used to reject a conflicting
                title match.

        Returns:
            The unique matching paper ID, or ``None`` when the title is absent,
            ambiguous, or conflicts with the supplied author hint.
        """
        normalized_title = _normalize_paper_title(title)
        if not normalized_title:
            return None

        matches = [
            (paper_id, metadata)
            for paper_id, metadata in self.lookup_paper_metadata.items()
            if _normalize_paper_title(metadata.get("title", ""))
            == normalized_title
        ]
        if author_hint:
            normalized_author = _normalize_author_name(author_hint)
            matches = [
                (paper_id, metadata)
                for paper_id, metadata in matches
                if any(
                    _normalize_author_name(author) == normalized_author
                    for author in metadata.get("authors", [])
                )
            ]

        return matches[0][0] if len(matches) == 1 else None

    @property
    def corpus_paper_ids(self) -> set[str]:
        return {chunk["paper_id"] for chunk in self._chunk_dict.values()}

    @property
    def encoder(self) -> BaseResearchEncoder:
        return self._encoder
