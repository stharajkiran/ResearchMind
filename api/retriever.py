import json
import logging
import os
from pathlib import Path

from typing import Optional
from api.models import SearchResult
from researchmind.embedding.models import MPNetEncoder
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.rrf import reciprocal_rank_fusion
from researchmind.retrieval.query_intelligence import QueryTransformer
from researchmind.retrieval.temporal import apply_recency_decay

from openai import OpenAI
import ollama

logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(self, project_root: Path, artifact_dir: Path | None = None):
        logger.info("Initializing Retriever service...")
        self.project_root = project_root
        # INDEX_PHASE env var selects which artifact dir to load.
        # Defaults to root (Phase 1 backward compat). Set INDEX_PHASE=phase2 for combined corpus.
        # self.artifact_dir = self.project_root / "artifacts" / "indexes" / "phase2"
        self.artifact_dir = artifact_dir

    def load(self, chunks_path: Path, model_name:Optional[str] = None) -> None:
        """Load the retriever components from disk.
        This includes:
        - Loading the encoder model
        - Initializing the Faiss and BM25 retrievers
        - Loading the Faiss and BM25 indexes from disk
        - Loading the papers metadata into a dictionary for ID to metadata mapping during search

        Args:
            chunks_path (Path): The path to the chunks metadata file, used for mapping chunk IDs to their corresponding metadata during search.
        """
        logger.info("Loading retriever service...")
        # load encoder, faiss index, bm25 index, papers dict
        self.encoder = MPNetEncoder()
        # Initialize retrievers
        logger.info(
            "Initializing Faiss and BM25 retrievers from artifact dir: %s",
            self.artifact_dir,
        )
        self.faissRetriver = FaissIndexBuilder(
            dimension=self.encoder.dim, artifact_dir=self.artifact_dir
        )
        self.bm25Retriver = BM25IndexBuilder(artifact_dir=self.artifact_dir)
        # Load indexes from disk
        self.faissRetriver.load_index("HNSW32")
        self.bm25Retriver.load_index()
        # Load papers dict for ID to metadata mapping during search
        self.chunk_dict = self._load_chunk_dict(chunks_path)
        # Initialize query transformer
        ollama_client = ollama.Client()
        self.query_transformer = QueryTransformer(
            client=ollama_client, model=model_name
        )

        logger.info("Retriever service loaded successfully.")

    def set_query_transformer_model(self, model_name: str):
        """Dynamically update the model used for query transformations."""
        self.query_transformer.set_model(model_name)
        logger.info("Query transformer model updated to: %s", model_name)

    def _load_chunk_dict(self, chunks_path: Path) -> dict[str, dict]:
        """Load chunk metadata into a dictionary for easy retrieval during search.

        Returns a dict mapping chunk_id to chunk metadata.
        {
            "chunk_id_1": {
                "chunk_id": "chunk_id_1",
                "paper_id": "paper_id_1",
                "text": "chunk text...",
                ...
            },
            ...
        }
        """
        # processed = self.project_root / "data" / "processed"
        # chunks_path = processed / "cleaned_chunks.jsonl"
        logger.info("Loading chunks metadata from source: %s", chunks_path)

        chunk_dict: dict[str, dict] = {}
        if not chunks_path.exists():
            logger.warning("Chunks metadata file not found at %s", chunks_path)
            return chunk_dict
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                chunk_dict.setdefault(c["chunk_id"], c)
        logger.info("Loaded %d chunks into dictionary.", len(chunk_dict))
        return chunk_dict

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "standard",
        recency_decay_rate: float | None = None,
    ) -> list[Chunk]:
        """Search for relevant chunks given a query.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.

        Returns:
            list[Chunk]: A list of chunks for the top k results.
        """
        # encode → (rewrite/hyde/no change)-> faiss.search → bm25.search → rrf → map IDs to paper metadata
        logger.info("--------------------------------")
        logger.info("Received search query: %s", query)
        # original query for bm25 search, since it doesn't use embeddings
        bm25_results = self.bm25Retriver.search(query, k=k)

        if mode == "rewrite":
            logger.info("Applying query rewrite transformation.")
            query = self.query_transformer.rewrite(query)
            logger.info("Rewritten query: %s", query)
        elif mode == "hyde":
            logger.info("Applying HyDE query transformation.")
            query = self.query_transformer.hyde(query)
            logger.info("HyDE-generated abstract: %s", query)
        else:
            logger.info("Using standard query without transformation.")
        # transformed or original query for faiss search
        q_embedding = self.encoder.encode([query])
        faiss_results = self.faissRetriver.search(q_embedding, k=k)
        # final chunk ids after fusion
        rrf_results = reciprocal_rank_fusion(faiss_results, bm25_results)[:k]
        if recency_decay_rate is not None:
            rrf_results = apply_recency_decay(
                rrf_results, self.chunk_dict, recency_decay_rate
            )
        # map paper IDs to metadata
        # convert each dict into SearchResult model
        search_results = [
            Chunk(**self.chunk_dict[chunk_id])
            for chunk_id in rrf_results
            if chunk_id in self.chunk_dict
        ]
        logger.info("*****Search results retrieved successfully for query******")

        return search_results
