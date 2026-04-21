import json
import logging
import os
from pathlib import Path

from researchmind.embedding.models import MPNetEncoder 
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(self, project_root: Path):
        logger.info("Initializing Retriever service...")
        self.project_root = project_root
        # INDEX_PHASE env var selects which artifact dir to load.
        # Defaults to root (Phase 1 backward compat). Set INDEX_PHASE=phase2 for combined corpus.
        phase = os.environ.get("INDEX_PHASE", "")
        if phase:
            self.artifact_dir = self.project_root / "artifacts" / "indexes" / phase
        else:
            self.artifact_dir = self.project_root / "artifacts" / "indexes"
    
    def load(self) -> None:
        logger.info("Loading retriever service...")
        # load encoder, faiss index, bm25 index, papers dict
        self.encoder = MPNetEncoder ()
        # Initialize retrievers
        self.faissRetriver = FaissIndexBuilder(dimension=self.encoder.dim, artifact_dir=self.artifact_dir)
        self.bm25Retriver = BM25IndexBuilder(artifact_dir=self.artifact_dir)
        # Load indexes from disk
        self.faissRetriver.load_index("HNSW32")
        self.bm25Retriver.load_index()
        # Load papers dict for ID to metadata mapping during search
        self.papers_dict = self._load_papers_dict()
        logger.info("Retriever service loaded successfully.")
        
    def search(self, query: str, k: int = 10) -> list[dict]:
        logger.info("Received search query: %s", query)
        # encode → faiss.search → bm25.search → rrf → map IDs to paper metadata
        q_embedding = self.encoder.encode([query])
        faiss_results = self.faissRetriver.search(q_embedding, k=k)
        bm25_results = self.bm25Retriver.search(query, k=k)
        rrf_results = reciprocal_rank_fusion(faiss_results, bm25_results)[:k]
        # map paper IDs to metadata
        search_results = [self.papers_dict[corpus_id] for corpus_id in rrf_results if corpus_id in self.papers_dict]
        logger.info("Search results: %s", search_results)
        return search_results

    def _load_papers_dict(self) -> dict[str, dict]:
        logger.info("Loading papers metadata into dictionary...")
        processed = self.project_root / "data" / "processed"
        sources = [processed / "papers.jsonl", processed / "papers_foundational.jsonl"]
        papers_dict: dict[str, dict] = {}
        for path in sources:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line)
                    papers_dict.setdefault(p["paper_id"], p)
        logger.info("Loaded %d papers into dictionary.", len(papers_dict))
        return papers_dict