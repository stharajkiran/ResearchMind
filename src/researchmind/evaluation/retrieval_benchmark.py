import json
from pathlib import Path
from time import time
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
 
import mlflow
from researchmind.embedding.models import MPNetEncoder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.utils.logging import configure_logging
from researchmind.retrieval.rrf import reciprocal_rank_fusion
from researchmind.utils.datatypes import RetrieverMetrics


from dotenv import load_dotenv
load_dotenv()

def run_benchmark(
    encoder: MPNetEncoder,
    queries: list[dict],
    query_embeddings: np.ndarray,
    corpus_ids: list[str],
    corpus_texts: list[str],
    corpus_embeddings: np.ndarray,
    logger: logging.Logger,
    logs_dir: Path | None = None,
    experiment_name: str = "retrieval_benchmark",
) -> dict:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=experiment_name):
        run_id = mlflow.active_run().info.run_id
        per_run_handler: logging.FileHandler | None = None
        per_run_log_path: Path | None = None

        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            safe_experiment_name = experiment_name.lower().replace(" ", "_")
            # Keep one log file per MLflow run so logs are traceable by run_id.
            per_run_log_path = logs_dir / f"{safe_experiment_name}_{run_id}.log"
            per_run_handler = logging.FileHandler(per_run_log_path, encoding="utf-8")
            per_run_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            if logger:
                logger.addHandler(per_run_handler)

        try:
            # ------------------------------------------------------------------------------
            logger.info(
                "Started benchmark run for %s (run_id=%s)", experiment_name, run_id
            )

            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("corpus_size", len(corpus_ids))

            # Initialize FaissIndexBuilder and BM25IndexBuilder
            faissRetriver = FaissIndexBuilder(dimension=encoder.dim)
            bm25Retriver = BM25IndexBuilder()

            faissRetriver.build_index(
                corpus_embeddings, corpus_ids, index_type="HNSW32"
            )
            bm25Retriver.build_index(corpus_texts, corpus_ids)

            recalls = defaultdict(RetrieverMetrics)
            logger.info(
                "Running %d queries and %d query embeddings...",
                len(queries),
                len(query_embeddings),
            )
            for q, query_embedding in zip(queries, query_embeddings):
                faiss_ids = faissRetriver.search(query_embedding.reshape(1, -1), k=10)
                bm25_ids = bm25Retriver.search(q["query"], k=10)
                hybrid_ids = reciprocal_rank_fusion(faiss_ids, bm25_ids)[:10]

                found_faiss = any(pid in faiss_ids for pid in q["relevant_paper_ids"])
                found_bm25 = any(pid in bm25_ids for pid in q["relevant_paper_ids"])
                found_rrf = any(pid in hybrid_ids for pid in q["relevant_paper_ids"])

                # Update for each retriever
                for name, found in [
                    ("faiss", found_faiss),
                    ("bm25", found_bm25),
                    ("rrf", found_rrf),
                ]:
                    recalls[name].update_found(q, found)

            faiss_semantic_recall = recalls["faiss"].semantic_recall
            faiss_technical_recall = recalls["faiss"].technical_recall
            bm25_semantic_recall = recalls["bm25"].semantic_recall
            bm25_technical_recall = recalls["bm25"].technical_recall
            rrf_semantic_recall = recalls["rrf"].semantic_recall
            rrf_technical_recall = recalls["rrf"].technical_recall
            metrics = {
                "faiss_semantic_recall": faiss_semantic_recall,
                "faiss_technical_recall": faiss_technical_recall,
                "bm25_semantic_recall": bm25_semantic_recall,
                "bm25_technical_recall": bm25_technical_recall,
                "rrf_semantic_recall": rrf_semantic_recall,
                "rrf_technical_recall": rrf_technical_recall,
            }
            # ------------------------------------------------------------------------------

            logger.info(metrics)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("run_log_artifact", "logs")
            return metrics

        except Exception:
            # Log full traceback so it is captured in both the session and per-run log files.
            logger.exception("Benchmark failed for experiment '%s'", experiment_name)
            raise
        finally:
            # Ensure file handlers are always released, even if benchmarking fails.
            if per_run_handler is not None:
                logger.removeHandler(per_run_handler)
                per_run_handler.close()
            # Persist local run logs in MLflow artifacts for remote inspection.
            if per_run_log_path is not None and per_run_log_path.exists():
                mlflow.log_artifact(str(per_run_log_path), artifact_path="logs")


if __name__ == "__main__":
    experiment_name = "retrieval_benchmark"

    project_root = Path(__file__).resolve().parents[3]
    logs_dir = project_root / "logs" / experiment_name
    # Session log captures cross-model orchestration messages in one place.
    session_log_path = (
        logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger(__name__)
    configure_logging(session_log_path, logger)

    logger.info("Starting embedding benchmark evaluation...")
    # ------------------------------------------------------------------------------
    papers_path = project_root / "data" / "processed" / "papers.jsonl"
    with papers_path.open("r", encoding="utf-8") as f:
        papers = [json.loads(line) for line in f]
        logger.info("Loaded %d papers from %s", len(papers), papers_path)

    queries_path = project_root / "experiments" / "60_query_test_set.json"
    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)
        logger.info("Loaded %d queries from %s", len(queries), queries_path)

    # ------------------------------------------------------------------------------
    logger.info("Initializing encoder...")
    encoder = MPNetEncoder()

    # we need corpus_embeddings to build the index, and corpus_ids for mapping back search results to paper IDs
    papers_dict = {p["paper_id"]: p for p in papers}
    corpus_ids = list(papers_dict.keys())
    corpus_texts = [papers_dict[pid]["abstract"] for pid in corpus_ids]
    corpus_embeddings = encoder.encode(corpus_texts)
    query_embeddings = encoder.encode([q["query"] for q in queries])
    logger.info("Encoder initialized successfully.")

    # ------------------------------------------------------------------------------
    # Implement run_benchmark and call it for each index type

    logger.info("Running benchmark for %s...", experiment_name)

    metrics = run_benchmark(
        encoder,
        queries,
        query_embeddings,
        corpus_ids,
        corpus_texts,
        corpus_embeddings,
        logger=logger,
        logs_dir=logs_dir,
    )
    logger.info("Completed benchmark for %s. Metrics: %s", experiment_name, metrics)
