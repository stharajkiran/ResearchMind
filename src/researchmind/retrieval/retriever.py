import json
import logging
from pathlib import Path

from researchmind.embedding.models import MPNetEncoder
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.chroma_store import ChromaStore
from researchmind.retrieval.chroma_store import ChromaStore
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.rrf import reciprocal_rank_fusion
from researchmind.retrieval.query_intelligence import QueryTransformer
from researchmind.retrieval.temporal import apply_recency_decay
from researchmind.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrieverService(VectorStore):
    def __init__(
        self,
        artifact_dir: Path | None = None,
        collection_name: str | None = None,
        chunks_path: Path | None = None,
    ) -> None:
        """Initialize the RetrieverService.

        Args:
            artifact_dir (Path | None): The directory containing retriever artifacts.
            collection_name (str | None): The name of the Chroma collection.
            chunks_path (Path | None): The path to the chunks metadata file.
        """
        logger.info("Initializing Retriever service...")
        # INDEX_PHASE env var selects which artifact dir to load.
        # Defaults to root (Phase 1 backward compat). Set INDEX_PHASE=phase2 for combined corpus.
        # self.artifact_dir = self.project_root / "artifacts" / "indexes" / "phase2"
        self.artifact_dir = artifact_dir
        self._chroma_collection_name = collection_name
        self.IsChunksLoaded = False
        if chunks_path is not None:
            self.load(chunks_path)
            self.IsChunksLoaded = True
        else:
            logger.warning(
                "No chunks_path provided. RetrieverService will not be fully initialized."
            )

    def load(self, chunks_path: Path) -> None:
        """Load the retriever components from disk.
        This includes:
        - Loading the encoder model
        - Initializing the Faiss and BM25 retrievers
        - Loading the Faiss and BM25 indexes from disk
        - Loading the papers metadata into a dictionary for ID to metadata mapping during search

        Args:
            chunks_path (Path): The path to the chunks metadata file, used for mapping chunk IDs to their corresponding metadata during search.
        """
        if self.IsChunksLoaded:
            logger.info("Chunks already loaded. Skipping load process.")
            return
        if chunks_path is None:
            logger.error(
                "chunks_path is required to load the retriever. Aborting load."
            )
            return
        # confirm the chunks_path exists before proceeding
        if not chunks_path.exists():
            logger.error(
                "Chunks metadata file not found at %s. Aborting load.", chunks_path
            )
            return
        try:
            logger.info("Loading retriever service...")
            # load encoder, faiss index, bm25 index, papers dict
            self._encoder = MPNetEncoder()
            # Initialize retrievers
            logger.info(
                "Initializing Faiss and BM25 retrievers from artifact dir: %s",
                self.artifact_dir,
            )
            self._faissRetriver = FaissIndexBuilder(
                dimension=self._encoder.dim, artifact_dir=self.artifact_dir
            )
            self._bm25Retriver = BM25IndexBuilder(artifact_dir=self.artifact_dir)
            # Load indexes from disk
            self._faissRetriver.load_index("HNSW32")
            self._bm25Retriver.load_index()
            # Load papers dict for ID to metadata mapping during search
            self._chunk_dict = self._load_chunk_dict(chunks_path)
            # Initialize query transformer
            self._query_transformer = QueryTransformer()
            self._chroma = ChromaStore(
                self._chroma_collection_name, encoder=self._encoder
            )

            logger.info("Retriever service loaded successfully.")
        except Exception as e:
            logger.error("Failed to load retriever service: %s", e)
            raise RuntimeError(f"Failed to load retriever service: {e}")

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

    def get_chunks_for_papers(self, paper_ids: list[str]) -> list[Chunk]:
        """Helper method to retrieve all chunks associated with a list of paper IDs."""
        return [
            Chunk(**c)
            for c in self._chunk_dict.values()
            if c["paper_id"] in set(paper_ids)
        ]

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "standard",
        filters: dict | None = None,
        recency_decay_rate: float | None = None,
    ) -> list[Chunk]:
        """Search for relevant chunks given a query.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.
            mode (str): The retrieval mode, which can be "standard", "rewrite", or "hyde". Determines how the query is transformed before retrieval.
            filters (dict | None): Optional metadata filters to apply when searching with ChromaDB.
            recency_decay_rate (float | None): Optional decay rate to apply to search results based on recency. Higher values will favor more recent chunks.

        Returns:
            list[Chunk]: A list of chunks for the top k results.
        """
        if not hasattr(self, "_encoder"):
            raise RuntimeError("RetrieverService.load() must be called before search()")
        # encode → (rewrite/hyde/no change)-> faiss.search → bm25.search → rrf → map IDs to paper metadata
        logger.info("--------------------------------")
        logger.info("Received search query: %s", query)

        if filters:
            return self._chroma.query_collection(query, k, where=filters)
        # original query for bm25 search, since it doesn't use embeddings
        bm25_results = self._bm25Retriver.search(query, k=k)

        if mode == "rewrite":
            logger.info("Applying query rewrite transformation.")
            query = self._query_transformer.rewrite(query)
            logger.info("Rewritten query: %s", query)
        elif mode == "hyde":
            logger.info("Applying HyDE query transformation.")
            query = self._query_transformer.hyde(query)
            logger.info("HyDE-generated abstract: %s", query)
        else:
            logger.info("Using standard query without transformation.")
        # transformed or original query for faiss search
        q_embedding = self._encoder.encode([query])
        faiss_results = self._faissRetriver.search(q_embedding, k=k)
        # final chunk ids after fusion
        rrf_results = reciprocal_rank_fusion(faiss_results, bm25_results)[:k]
        if recency_decay_rate is not None:
            rrf_results = apply_recency_decay(
                rrf_results, self._chunk_dict, recency_decay_rate
            )
        # map paper IDs to metadata
        # convert each dict into SearchResult model
        search_results = [
            Chunk(**self._chunk_dict[chunk_id])
            for chunk_id in rrf_results
            if chunk_id in self._chunk_dict
        ]
        logger.info("*****Search results retrieved successfully for query******")

        return search_results
