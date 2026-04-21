from collections import defaultdict
import json
from pathlib import Path
from time import time
import logging
from datetime import datetime
import numpy as np
import mlflow
from researchmind.embedding.models import (
    BGEEncoder,
    BaseResearchEncoder,
    MPNetEncoder,
    SPECTER2Encoder,
)
from researchmind.utils.logging import configure_logging
from researchmind.utils.datatypes import RetrieverMetrics

from dotenv import load_dotenv

load_dotenv()


def run_benchmark(
    encoder: BaseResearchEncoder,
    model_name: str,
    papers: list[dict],
    queries: list[dict],
    logger: logging.Logger,
    logs_dir: Path | None = None,
    experiment_name: str = "embedding_benchmark",
) -> dict:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        run_id = mlflow.active_run().info.run_id
        per_run_handler: logging.FileHandler | None = None
        per_run_log_path: Path | None = None

        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            safe_model_name = model_name.lower().replace(" ", "_")
            # Keep one log file per MLflow run so logs are traceable by run_id.
            per_run_log_path = (
                logs_dir
                / f"{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            per_run_handler = logging.FileHandler(per_run_log_path, encoding="utf-8")
            per_run_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            if logger:
                logger.addHandler(per_run_handler)

        try:
            logger.info("Started benchmark run for %s (run_id=%s)", model_name, run_id)

            # match paper IDs to abstracts for easy retrieval during evaluation
            papers_dict = {p["paper_id"]: p for p in papers}

            # Embed all abstracts
            corpus_ids = list(papers_dict.keys())
            corpus_texts = [papers_dict[pid]["abstract"] for pid in corpus_ids]

            start = time()
            if isinstance(encoder, SPECTER2Encoder):
                corpus_embeddings = encoder.encode_corpus(papers)
            else:
                corpus_embeddings = encoder.encode(corpus_texts)
            elapsed = time() - start
            throughput = len(corpus_texts) / max(elapsed, 1e-9)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("corpus_size", len(papers))
            mlflow.log_param("vector_dimension", int(corpus_embeddings.shape[1]))

            latencies = []
            recalls = []
            recalls_cat = RetrieverMetrics()
            logger.info("Running %d queries...", len(queries))
            for q in queries:
                start_time = time()
                if isinstance(encoder, SPECTER2Encoder):
                    query_embedding = encoder.encode_queries([q["query"]])
                else:
                    query_embedding = encoder.encode([q["query"]])
                scores = corpus_embeddings @ query_embedding.T
                top10_indices = np.argsort(scores.flatten())[-10:][::-1]
                top10_ids = [corpus_ids[i] for i in top10_indices]
                latencies.append(time() - start_time)
                found = any(pid in top10_ids for pid in q["relevant_paper_ids"])
                recalls.append(1.0 if found else 0.0)
                recalls_cat.update_found(q, found)

            semantic_recall = recalls_cat.semantic_recall
            technical_recall = recalls_cat.technical_recall
            mean_recall = float(np.mean(recalls))
            p95_latency_ms = float(np.percentile(latencies, 95) * 1000)
            metrics = {
                "recall_10": mean_recall,
                "semantic_recall": semantic_recall,
                "technical_recall": technical_recall,
                "p95_latency": p95_latency_ms,
                "throughput": throughput,
            }

            logger.info(
                "Benchmark results - Recall_10: %.2f, Semantic Recall: %.2f, Technical Recall: %.2f, P95 Latency: %.2fms, Throughput: %.2f docs/sec",
                mean_recall,
                semantic_recall,
                technical_recall,
                p95_latency_ms,
                throughput,
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("run_log_artifact", "logs")
            mlflow.set_tag("experiment_name", experiment_name)
            return metrics

        except Exception:
            # Log full traceback so it is captured in both the session and per-run log files.
            logger.exception("Benchmark failed for model '%s'", model_name)
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
    
    experiment_name = "embedding_benchmark_final"
    queries_path = "60_query_test_set.json"

    project_root = Path(__file__).resolve().parents[3]
    logs_dir = project_root / "logs" / experiment_name
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

    queries_path = project_root / "experiments" / queries_path
    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)
        logger.info("Loaded %d queries from %s", len(queries), queries_path)

    logger.info("Initializing models...")
    # Run benchmark for all three models
    specter_encoder = SPECTER2Encoder()
    mpnet_encoder = MPNetEncoder()
    bge_encoder = BGEEncoder()
    logger.info("Models initialized successfully.")

    # Implement run_benchmark and call it for each model, then print results in a table format
    results = []
    for model, model_name in [
        (mpnet_encoder, "MPNet"),
        (bge_encoder, "BGE"),
        (specter_encoder, "SPECTER2")
    ]:
        logger.info("Running benchmark for %s...", model_name)
        metrics = run_benchmark(
            model,
            model_name,
            papers,
            queries,
            logger=logger,
            logs_dir=logs_dir,
            experiment_name=experiment_name,
        )
        logger.info("Completed benchmark for %s. Metrics: %s", model_name, metrics)
        results.append((model_name, metrics))

    # Print results table
    logger.info(
        "%-15s %-10s %-12s %-12s %-12s %-12s", "Model", "Recall_10", "Semantic Recall", "Technical Recall", "Throughput", "P95 Latency"
    )
    for model_name, metrics in results:
        logger.info(
            "%-15s %-10.2f %-12.2f %-12.2f %-12.2f %-12.2f",
            model_name,
            metrics["recall_10"],
            metrics["semantic_recall"],
            metrics["technical_recall"],
            metrics["throughput"],
            metrics["p95_latency"],
        )

    logger.info(
        "All benchmarks completed. Session logs written to %s", session_log_path
    )
