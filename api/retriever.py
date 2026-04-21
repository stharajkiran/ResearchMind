from pathlib import Path
import json
import logging

from researchmind.embedding.models import BGEEncoder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)  

class RetrieverService:
    def __init__(self, project_root: Path): 
        logger.info("Initializing Retriever service...")
        self.project_root = project_root
        self.artifact_dir = self.project_root / "artifacts" / "indexes"
    
    def load(self) -> None:
        logger.info("Loading retriever service...")
        # load encoder, faiss index, bm25 index, papers dict
        self.encoder = BGEEncoder()
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
        # load papers metadata into a dict for easy retrieval during search
        papers_path = self.project_root / "data" / "processed" / "papers.jsonl"
        with papers_path.open("r", encoding="utf-8") as f:
            papers = [json.loads(line) for line in f]
        papers_dict = {p["paper_id"]: p for p in papers}
        logger.info("Loaded %d papers into dictionary.", len(papers_dict))
        return papers_dict