import argparse
from pathlib import Path
import json
import logging
from datetime import datetime
from numpy import ndarray

from researchmind.embedding.models import MPNetEncoder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.utils.logging import configure_logging
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)


class IndexBuilderService:
    def __init__(
        self,
        artifact_dir: Path | None = None,
        papers_path: Path | None = None,
    ):
        logger.info("Initializing Index Builder service...")
        self.artifact_dir = artifact_dir 
        self.papers_path = papers_path
        self.encoder = None
        self.faissIndexBuilder = None
        self.bm25IndexBuilder = None
        self.papers_dict = None

    def _load_encoder(self):
        logger.info("Loading encoder model...")
        self.encoder = MPNetEncoder()
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
        with self.papers_path.open("r", encoding="utf-8") as f:
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

    def run(self, index_type: str):
        self._load_encoder()
        self._load_IndexBuilders()

        chunks_path = self.papers_path
        with chunks_path.open() as f:
            chunks = [json.loads(line) for line in f]
    
        corpus_ids = [c["chunk_id"] for c in chunks]
        corpus_texts = [c["text"] for c in chunks]

        logger.info("Encoding corpus texts into embeddings...")
        corpus_embeddings = self.encoder.encode(corpus_texts, batch_size= 256)
        logger.info("Corpus encoding completed successfully.")

        self.build_faiss_index(corpus_embeddings, corpus_ids, index_type=index_type)
        self.build_bm25_index(corpus_texts, corpus_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS and BM25 indexes from processed papers metadata."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=find_project_root(),
        help="Project root path. Defaults to auto-detected repository root.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(find_project_root() / "artifacts" / "indexes" / "semantic"),
        help="Output directory for generated indexes. Defaults to <project-root>/artifacts/indexes.",
    )
    parser.add_argument(
        "--papers-path",
        type=Path,
        default=Path(find_project_root() / "data" / "processed" / "cleaned_semantic_chunks.jsonl"),
        help="Path to papers JSONL input file. Defaults to <project-root>/data/processed/cleaned_chunk.jsonl.",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="HNSW32",
        choices=["Flat", "IVF", "HNSW32"],
        help='FAISS index type to build: "Flat", "IVF", or "HNSW32".',
    )
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    project_root = cli_args.project_root
    logs_dir = project_root / "logs" / "build_indexes"
    # Session log captures cross-model orchestration messages in one place.
    session_log_path = (
        logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    configure_logging(session_log_path, logger)
    logger.info("Starting Index Builder service with arguments: %s", cli_args)

    service = IndexBuilderService(
        artifact_dir=cli_args.artifact_dir,
        papers_path=cli_args.papers_path,
    )
    service.run(index_type=cli_args.index_type)