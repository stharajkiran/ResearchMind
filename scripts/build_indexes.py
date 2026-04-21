from pathlib import Path
import json
import logging
from datetime import datetime
from numpy import ndarray

from researchmind.embedding.models import BGEEncoder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.utils.logging import configure_logging
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)


class IndexBuilderService:
    def __init__(self, project_root: Path):
        logger.info("Initializing Index Builder service...")
        self.project_root = project_root
        self.artifact_dir = self.project_root / "artifacts" / "indexes"
        self.encoder = None
        self.faissIndexBuilder = None
        self.bm25IndexBuilder = None
        self.papers_dict = None

    def _load_encoder(self):
        logger.info("Loading encoder model...")
        self.encoder = BGEEncoder()
        logger.info("Encoder model loaded successfully.")

    def _load_IndexBuilders(self):
        logger.info("Initializing Index Builders...")
        self.faissIndexBuilder = FaissIndexBuilder(
            dimension=self.encoder.dim, artifact_dir=self.artifact_dir
        )
        self.bm25IndexBuilder = BM25IndexBuilder(artifact_dir=self.artifact_dir)
        logger.info("Index Builders initialized successfully.")

    def _load_papers_dict(self):
        logger.info("Loading papers metadata into dictionary...")
        # load papers metadata into a dict for easy retrieval during search
        papers_path = self.project_root / "data" / "processed" / "papers.jsonl"
        with papers_path.open("r", encoding="utf-8") as f:
            papers = [json.loads(line) for line in f]
        papers_dict = {p["paper_id"]: p for p in papers}
        self.papers_dict = papers_dict
        logger.info("Loaded %d papers into dictionary.", len(papers_dict))

    def build_faiss_index(
        self, corpus_embeddings: ndarray, corpus_ids: list[str], index_type: str
    ):
        logger.info(
            "Building Faiss index of type '%s' with %d embeddings...",
            index_type,
            len(corpus_embeddings),
        )
        self.faissIndexBuilder.build_index(
            corpus_embeddings, corpus_ids, index_type=index_type
        )
        logger.info("Faiss index built and saved successfully.")

    def build_bm25_index(self, corpus_texts: list[str], corpus_ids: list[str]):
        logger.info("Building BM25 index with %d documents...", len(corpus_texts))
        self.bm25IndexBuilder.build_index(corpus_texts, corpus_ids)
        logger.info("BM25 index built and saved successfully.")

    def run(self):
        self._load_encoder()
        self._load_IndexBuilders()
        self._load_papers_dict()
        corpus_ids = list(self.papers_dict.keys())
        corpus_texts = [self.papers_dict[pid]["abstract"] for pid in corpus_ids]
        corpus_embeddings = self.encoder.encode(corpus_texts)
        self.build_faiss_index(corpus_embeddings, corpus_ids, index_type="HNSW32")
        self.build_bm25_index(corpus_texts, corpus_ids)


if __name__ == "__main__":
    project_root = find_project_root()
    logs_dir = project_root / "logs" / "build_indexes"
    # Session log captures cross-model orchestration messages in one place.
    session_log_path = (
        logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    configure_logging(session_log_path, logger)

    service = IndexBuilderService(project_root)
    service.run()
