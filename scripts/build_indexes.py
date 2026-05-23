import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from researchmind.embedding.models import BaseResearchEncoder, BGEEncoder, MPNetEncoder, SPECTER2Encoder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.interfaces import DenseIndex, SparseIndex
from researchmind.utils.config import load_phase_config
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# Short alias → encoder class
ENCODER_MAP: dict[str, type[BaseResearchEncoder]] = {
    "mpnet":    MPNetEncoder,
    "bge":      BGEEncoder,
    "specter2": SPECTER2Encoder,
    # Full model names also accepted
    "sentence-transformers/all-mpnet-base-v2": MPNetEncoder,
    "BAAI/bge-small-en-v1.5":                 BGEEncoder,
    "allenai/specter2_base":                   SPECTER2Encoder,
}


def build_encoder(name: str) -> BaseResearchEncoder:
    cls = ENCODER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown encoder '{name}'. Choose: {list(ENCODER_MAP.keys())}"
        )
    return cls()


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

        logger.info("Encoding %d chunks with %s...", len(corpus_texts), self.encoder.model_name)
        corpus_embeddings = self.encoder.encode(corpus_texts, batch_size=256)

        logger.info("Building dense index...")
        self.dense.build(corpus_embeddings, corpus_ids)

        logger.info("Building sparse index...")
        self.sparse.build(corpus_texts, corpus_ids)

        logger.info("Done.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes from chunks.")
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Encoder to use. Overrides model.embedding in config. "
             "Choices: mpnet, bge, specter2 (or full HuggingFace model name).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    project_root = find_project_root()
    cfg = load_phase_config(project_root)

    # Existence guard — skip if already built
    if cfg.index.artifact_dir.exists() and any(cfg.index.artifact_dir.iterdir()):
        print(f"Indexes already exist at {cfg.index.artifact_dir} — skipping. Delete to rebuild.")
        sys.exit(0)

    logs_dir = project_root / "logs" / "build_indexes"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", logger
    )

    # --encoder overrides YAML; falls back to cfg.model.embedding
    encoder_name = args.encoder or cfg.model.embedding
    logger.info(
        "Building indexes | phase=%s backend=%s encoder=%s",
        cfg.name, cfg.index.vector_backend, encoder_name,
    )

    encoder = build_encoder(encoder_name)
    sparse = BM25IndexBuilder(artifact_dir=cfg.index.artifact_dir)

    if cfg.index.vector_backend == "qdrant":
        from researchmind.retrieval.backends.qdrant_backend import QdrantBackend
        dense = QdrantBackend(
            collection_name=f"researchmind_{cfg.name}",
            encoder=encoder,
            dimension=encoder.dim,
        )
    else:
        dense = FaissIndexBuilder(
            dimension=encoder.dim,
            artifact_dir=cfg.index.artifact_dir,
            index_type=cfg.index.index_type,
        )

    IndexBuilderService(
        encoder=encoder,
        dense=dense,
        sparse=sparse,
        chunks_path=cfg.index.chunks_path,
    ).run()

    logger.info("All done. Index artifacts stored in %s", cfg.index.artifact_dir)
