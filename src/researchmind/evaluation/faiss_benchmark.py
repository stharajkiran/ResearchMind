import json
from pathlib import Path
from time import time
import logging
from datetime import datetime
import numpy as np

import mlflow
from researchmind.embedding.models import BGEEncoder, BaseResearchEncoder
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.utils.logging import configure_logging

from dotenv import load_dotenv

load_dotenv()




def run_benchmark(
    faissIndex: FaissIndexBuilder,
    index_name: str,
    queries: list[dict],
    query_embeddings: np.ndarray,
    corpus_ids: list[str],
    logger: logging.Logger,
    logs_dir: Path | None = None,
    experiment_name: str = "faiss_benchmark",
) -> dict:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=index_name):
        run_id = mlflow.active_run().info.run_id
        per_run_handler: logging.FileHandler | None = None
        per_run_log_path: Path | None = None

        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            safe_index_name = index_name.lower().replace(" ", "_")
            # Keep one log file per MLflow run so logs are traceable by run_id.
            per_run_log_path = logs_dir / f"{safe_index_name}_{run_id}.log"
            per_run_handler = logging.FileHandler(per_run_log_path, encoding="utf-8")
            per_run_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            if logger:
                logger.addHandler(per_run_handler)

        try:
            logger.info("Started benchmark run for %s (run_id=%s)", index_name, run_id)

            mlflow.log_param("index_name", index_name)
            mlflow.log_param("corpus_size", len(corpus_ids))

            # Build the FAISS index and measure build time
            start = time()
            faissIndex.build_index(corpus_embeddings, corpus_ids, index_type=index_name)
            build_time = time() - start

            latencies = []
            recalls = []
            logger.info(
                "Running %d queries and %d query embeddings...",
                len(queries),
                len(query_embeddings),
            )
            for q, query_embedding in zip(queries, query_embeddings):
                start_time = time()
                top10_ids = faissIndex.search(query_embedding.reshape(1, -1), k=10)
                latencies.append(time() - start_time)
                # Check if any of the relevant paper IDs are in the top 10 results
                found = any(pid in top10_ids for pid in q["relevant_paper_ids"])
                recalls.append(1.0 if found else 0.0)

            mean_recall = float(np.mean(recalls))
            p50_latency_ms = float(np.percentile(latencies, 50) * 1000)
            p95_latency_ms = float(np.percentile(latencies, 95) * 1000)
            metrics = {
                "recall_10": mean_recall,
                "p50_latency": p50_latency_ms,
                "p95_latency": p95_latency_ms,
                "build_time": build_time,
            }

            logger.info(
                "Benchmark results - Recall_10: %.2f, P50 Latency: %.2fms, P95 Latency: %.2fms, Build Time: %.2fs",
                mean_recall,
                p50_latency_ms,
                p95_latency_ms,
                build_time,
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tag("index_name", index_name)
            mlflow.set_tag("run_log_artifact", "logs")
            return metrics

        except Exception:
            # Log full traceback so it is captured in both the session and per-run log files.
            logger.exception("Benchmark failed for index '%s'", index_name)
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
    project_root = Path(__file__).resolve().parents[3]
    logs_dir = project_root / "logs" / "faiss_benchmark"
    # Session log captures cross-model orchestration messages in one place.
    session_log_path = (
        logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger(__name__)
    configure_logging(session_log_path, logger)

    logger.info("Starting embedding benchmark evaluation...")

    papers_path = project_root / "data" / "processed" / "papers.jsonl"
    with papers_path.open("r", encoding="utf-8") as f:
        papers = [json.loads(line) for line in f]
        logger.info("Loaded %d papers from %s", len(papers), papers_path)

    queries_path = project_root / "experiments" / "60_query_test_set.json"
    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)
        logger.info("Loaded %d queries from %s", len(queries), queries_path)

    logger.info("Initializing encoder...")
    encoder = BGEEncoder()

    # we need corpus_embeddings to build the index, and corpus_ids for mapping back search results to paper IDs
    papers_dict = {p["paper_id"]: p for p in papers}
    corpus_ids = list(papers_dict.keys())
    corpus_texts = [papers_dict[pid]["abstract"] for pid in corpus_ids]
    corpus_embeddings = encoder.encode(corpus_texts)
    query_embeddings = encoder.encode([q["query"] for q in queries])
    logger.info("Encoder initialized successfully.")

    # Initialize FaissIndexBuilder with the correct dimension from the encoder
    faissIndex = FaissIndexBuilder(dimension=encoder.dim)
    # Implement run_benchmark and call it for each index type
    results = []
    for index_name in [
        "Flat",
        "IVF100",
        "HNSW32",
    ]:
        logger.info("Running benchmark for %s...", index_name)
        # build the index — measure build time here

        metrics = run_benchmark(
            faissIndex,
            index_name,
            queries,
            query_embeddings,
            corpus_ids,
            logger=logger,
            logs_dir=logs_dir,
        )
        logger.info("Completed benchmark for %s. Metrics: %s", index_name, metrics)
        results.append((index_name, metrics))
