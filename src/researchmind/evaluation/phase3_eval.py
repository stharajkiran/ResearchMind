import atexit
import json
import logging
from datetime import datetime
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import mlflow
from api.retriever import RetrieverService
from researchmind.evaluation.test_set_generator import TestQuery
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging, configure_logging_root

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Config:
    project_root = find_project_root()
    experiment_name = "phase3_eval"
    logs_dir = project_root / "logs" / experiment_name
    queries_path = project_root / "data" / "processed" / "200_test_queries_set.json"
    artifact_dir = project_root / "artifacts" / "indexes" / "phase2"
    chunks_path = project_root / "data" / "processed" / "cleaned_chunks.jsonl"
    decay_factor = 0.9
    rewrite_model = "qwen3.5:9b"
    hyde_model = "qwen3.6:27b"


def cleanup_vram():
    print("Automatic VRAM Cleanup initiated...")
    os.system("ollama stop qwen3.6:9b")
    os.system("ollama stop qwen3.6:27b")


def run_benchmark(
    mode: str,
    retriever: RetrieverService,
    queries: list[TestQuery],
    experiment_name: str = Config.experiment_name,
    recency_decay_rate: float | None = None,
) -> dict:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=mode):
        logger.info(
            "Started benchmark run for mode=%s decay=%s", mode, recency_decay_rate
        )
        mlflow.log_param("mode", mode)
        mlflow.log_param("recency_decay_rate", recency_decay_rate)

        category_indices = defaultdict(list)
        for i, query in enumerate(queries):
            category_indices[query.category].append(i)

        total_queries = sum(len(v) for v in category_indices.values())
        category_recalls = {}
        # track progress with nested tqdm: outer for categories
        outer = tqdm(
            category_indices.items(),
            desc=f"[{mode}] categories",
            unit="cat",
            total=len(category_indices),
        )
        # out of all queries
        query_counter = tqdm(
            total=total_queries, desc="  queries", unit="q", leave=False
        )

        for cat, indices in outer: # each category and its query indices
            cat_queries = [queries[i] for i in indices]
            recalls = []
            for query in cat_queries:
                try:
                    search_results = retriever.search(
                        query.question,
                        k=10,
                        mode=mode,
                        recency_decay_rate=recency_decay_rate,
                    )
                    topk_ids = [r.chunk_id for r in search_results]
                    recalls.append(float(query.chunk_id in topk_ids))
                except Exception as e:
                    logger.warning(
                        "Search failed for query '%s': %s", query.question[:50], e
                    )
                    recalls.append(0.0)
                query_counter.update(1)
                query_counter.set_postfix({"running": f"{np.mean(recalls):.3f}"})

            category_recalls[cat] = float(np.mean(recalls))
            mlflow.log_metric(f"{cat}_recall", category_recalls[cat])
            outer.set_postfix({cat: f"{category_recalls[cat]:.3f}"})
            logger.info("Category '%s' recall@10: %.3f", cat, category_recalls[cat])

        query_counter.close()
        outer.close()

        overall = float(np.mean(list(category_recalls.values())))
        mlflow.log_metric("overall_recall", overall)
        logger.info("Overall recall@10: %.3f", overall)

    return category_recalls


def temporal_run(
    run_name: str,
    retriever: RetrieverService,
    queries: list[TestQuery],
    recency_decay_rate: float | None,
    experiment_name: str = Config.experiment_name,
) -> dict:
    temporal_queries = [q for q in queries if q.category == "Temporal"]
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        logger.info("Started temporal run: %s (decay=%s)", run_name, recency_decay_rate)
        mlflow.log_param("mode", "standard")
        mlflow.log_param("recency_decay_rate", recency_decay_rate)

        recalls = []
        with tqdm(
            temporal_queries, desc=run_name, unit="q", postfix={"recall": "?"}
        ) as pbar:
            for query in pbar:
                try:
                    search_results = retriever.search(
                        query.question,
                        k=10,
                        mode="standard",
                        recency_decay_rate=recency_decay_rate,
                    )
                    topk_ids = [r.chunk_id for r in search_results]
                    recalls.append(float(query.chunk_id in topk_ids))
                except Exception as e:
                    logger.warning(
                        "Search failed for query '%s': %s", query.question[:50], e
                    )
                    recalls.append(0.0)
                pbar.set_postfix({"recall": f"{np.mean(recalls):.3f}"})

        recall = float(np.mean(recalls))
        mlflow.log_metric("Temporal_recall", recall)
        logger.info("Temporal recall@10 (%s): %.3f", run_name, recall)

    return {"Temporal": recall}


if __name__ == "__main__":
    logger.info("Starting Phase 3 Evaluation...")
    Config.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Logs directory ensured at %s", Config.logs_dir)
    session_log_path = (
        Config.logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    configure_logging_root(session_log_path)
    logger.info("Session log initialized at %s", session_log_path)

    logger.info("Loading test queries...")
    queries = []
    try:
        with open(Config.queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            queries.append(TestQuery.model_validate(item))
    except FileNotFoundError:
        logger.error("Test set not found at %s", Config.queries_path)
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s", e)
        raise SystemExit(1)

    logger.info("Loaded %d queries.", len(queries))

    logger.info("Initializing retriever...")
    try:
        retriever = RetrieverService(Config.project_root, Config.artifact_dir)
        retriever.load(Config.chunks_path)
        logger.info("Retriever loaded with chunks metadata from %s", Config.chunks_path)
    except Exception as e:
        logger.error("Failed to initialize retriever: %s", e)
        raise SystemExit(1)
    logger.info("Retriever ready.")

    # Mode A/B: standard vs rewrite vs hyde on all 200 queries
    modes = ["rewrite", "standard",  "hyde"]
    for mode in tqdm(modes, desc="Benchmark modes", unit="mode"):
        if mode == "rewrite":
            retriever.set_query_transformer_model(Config.rewrite_model)
            logger.info("Set retriever to rewrite mode with model %s", Config.rewrite_model)
        elif mode == "hyde":
            retriever.set_query_transformer_model(Config.hyde_model)
            logger.info("Set retriever to hyde mode with model %s", Config.hyde_model)
        else:
            logger.info("Set retriever to standard mode with no query transformation")
        logger.info("Running benchmark: mode=%s", mode)
        metrics = run_benchmark(mode, retriever, queries)
        logger.info("Completed benchmark for mode=%s. Metrics: %s", mode, metrics)

    logger.info("Running temporal A/B test with and without recency decay...")
    # Temporal A/B: standard with vs without recency decay on 40 temporal queries
    temporal_run("temporal_no_decay", retriever, queries, recency_decay_rate=None)
    temporal_run(
        "temporal_with_decay",
        retriever,
        queries,
        recency_decay_rate=Config.decay_factor,
    )

    # Register this function to run when the script/kernel exits
    atexit.register(cleanup_vram)
    logger.info("Benchmarking complete. VRAM cleanup will run on exit.")
