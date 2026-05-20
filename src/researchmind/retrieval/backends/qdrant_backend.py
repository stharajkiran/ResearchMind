import logging
import os

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from researchmind.embedding.models import BaseResearchEncoder
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.interfaces import DenseIndex, FilteredStore

logger = logging.getLogger(__name__)


class QdrantBackend(DenseIndex, FilteredStore):
    """
    Single Qdrant collection implementing both DenseIndex and FilteredStore.
    Replaces FAISS (dense search) + ChromaDB (filtered search) with one system.

    Usage in api/app.py:
        backend = QdrantBackend("researchmind", encoder=encoder, dimension=encoder.dim)
        retriever = RetrieverService(dense=backend, sparse=sparse, filtered=backend, ...)
    """

    def __init__(
        self,
        collection_name: str,
        encoder: BaseResearchEncoder,
        dimension: int,
        url: str | None = None,
    ):
        self.collection_name = collection_name
        self.encoder = encoder
        self.dimension = dimension
        self._client = QdrantClient(
            url=url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        )

    # ── DenseIndex ────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        self._client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"chunk_id": ids[i]},
            )
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)
        logger.info(
            "Built Qdrant collection '%s' with %d vectors.", self.collection_name, len(ids)
        )

    def load(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in collections:
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' not found. "
                "Run 'make indexes' to build it first."
            )
        logger.info("Qdrant collection '%s' ready.", self.collection_name)

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[str]:
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vec[0].tolist(),
            limit=k,
        )
        return [r.payload["chunk_id"] for r in results]

    # ── FilteredStore ─────────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> None:
        embeddings = self.encoder.encode([c.text for c in chunks])
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunks[i].chunk_id,
                    "paper_id": chunks[i].paper_id,
                    "section": chunks[i].section,
                    "title": chunks[i].title,
                    "year": chunks[i].year,
                    "authors": chunks[i].authors,
                    "text": chunks[i].text,
                },
            )
            for i in range(len(chunks))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Upserted %d chunks into Qdrant.", len(chunks))

    def query(self, query: str, k: int = 10, filters: dict | None = None) -> list[Chunk]:
        query_vec = self.encoder.encode([query])[0].tolist()
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            query_filter=self._build_filter(filters) if filters else None,
            limit=k,
        )
        return [
            Chunk(
                chunk_id=r.payload["chunk_id"],
                paper_id=r.payload["paper_id"],
                section=r.payload["section"],
                text=r.payload["text"],
                year=r.payload["year"],
                authors=r.payload["authors"],
                title=r.payload["title"],
                page=None,
            )
            for r in results
        ]

    def _build_filter(self, filters: dict) -> Filter:
        return Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
        )
