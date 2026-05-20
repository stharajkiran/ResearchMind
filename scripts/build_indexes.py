import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from researchmind.embedding.models import BaseResearchEncoder, MPNetEncoder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.interfaces import DenseIndex, SparseIndex
from researchmind.utils.config import load_phase_config
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class IndexBuilderService:
    def __init__(
        self,
        encoder: BaseResearchEncoder,
        dense: DenseIndex,
        sparse: SparseIndex,
        chunks_path: Path,
    ):
        self.encoder = encoder
        self.dense = dense
        self.sparse = sparse
        self.chunks_path = chunks_path

    def run(self) -> None:
        logger.info("Loading chunks from %s...", self.chunks_path)
        with self.chunks_path.open(encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        corpus_ids = [c["chunk_id"] for c in chunks]
        corpus_texts = [c["text"] for c in chunks]

        logger.info("Encoding %d chunks...", len(corpus_texts))
        corpus_embeddings = self.encoder.encode(corpus_texts, batch_size=256)

        logger.info("Building dense index...")
        self.dense.build(corpus_embeddings, corpus_ids)

        logger.info("Building sparse index...")
        self.sparse.build(corpus_texts, corpus_ids)

        logger.info("Done.")


if __name__ == "__main__":
    project_root = find_project_root()
    cfg = load_phase_config(project_root)

    # Existence guard — skip if already built
    if cfg.artifact_dir.exists() and any(cfg.artifact_dir.iterdir()):
        print(f"Indexes already exist at {cfg.artifact_dir} — skipping. Delete to rebuild.")
        sys.exit(0)

    logs_dir = project_root / "logs" / "build_indexes"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", logger
    )
    logger.info("Building indexes for phase=%s, backend=%s", cfg.phase, cfg.vector_backend)

    encoder = MPNetEncoder()
    sparse = BM25IndexBuilder(artifact_dir=cfg.artifact_dir)

    if cfg.vector_backend == "qdrant":
        from researchmind.retrieval.backends.qdrant_backend import QdrantBackend
        dense = QdrantBackend(
            collection_name=f"researchmind_{cfg.phase}",
            encoder=encoder,
            dimension=encoder.dim,
        )
    else:
        dense = FaissIndexBuilder(
            dimension=encoder.dim,
            artifact_dir=cfg.artifact_dir,
            index_type=cfg.index_type,
        )

    IndexBuilderService(
        encoder=encoder,
        dense=dense,
        sparse=sparse,
        chunks_path=cfg.chunks_path,
    ).run()
