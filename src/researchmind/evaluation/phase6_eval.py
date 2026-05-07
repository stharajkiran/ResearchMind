import logging
import time
import httpx
import numpy as np
import mlflow
from dotenv import load_dotenv

from researchmind.utils.logging import configure_logging
from researchmind.utils.find_root import find_project_root

load_dotenv()

project_root = find_project_root()
log_dir = project_root / "logs" / "phase_6" / "phase6_eval.log"
logger = logging.getLogger(__name__)
configure_logging(log_dir, logger)

API_BASE = "http://localhost:8000"


def run_latency_benchmark(
    queries: list[tuple[str, str]], session_id: str
) -> list[tuple[float, str]]:
    logger.info(
        "Starting benchmark: session_id=%s, queries=%d", session_id, len(queries)
    )
    latencies = []
    with httpx.Client(timeout=120) as client:
        for query, query_type in queries:
            logger.debug("Querying [%s]: %s", query_type, query)
            start = time.perf_counter()
            client.post(
                f"{API_BASE}/agent",
                json={"query": query, "session_id": session_id},
            )
            elapsed = time.perf_counter() - start
            latencies.append((elapsed, query_type))
            logger.debug("[%s] %.2fs", query_type, elapsed)
    logger.info(
        "Benchmark complete: session_id=%s, avg=%.2fs",
        session_id,
        sum(l for l, _ in latencies) / len(latencies),
    )
    return latencies


def main():
    queries: list[tuple[str, str]] = [
        # search (8)
        ("What are the main approaches to out-of-distribution detection?", "search"),
        ("How does DINO use self-supervised learning?", "search"),
        ("What are the limitations of contrastive learning?", "search"),
        ("How does batch normalization stabilize training in deep networks?", "search"),
        ("What is the mathematical formulation of the attention mechanism?", "search"),
        ("How does knowledge distillation transfer model capacity?", "search"),
        (
            "What is the role of data augmentation in self-supervised learning?",
            "search",
        ),
        ("How do sparse mixture-of-experts models reduce inference cost?", "search"),
        # compare (4)
        ("How do vision transformers compare to CNNs?", "compare"),
        (
            "How does LoRA compare to full fine-tuning for large language models?",
            "compare",
        ),
        (
            "What are the differences between BERT and GPT pre-training objectives?",
            "compare",
        ),
        (
            "How does AdamW compare to SGD with momentum for transformer training?",
            "compare",
        ),
        # gap_detection (3)
        (
            "What open problems remain in adversarial robustness for vision models?",
            "gap_detection",
        ),
        (
            "What are the unresolved challenges in multi-modal alignment?",
            "gap_detection",
        ),
        (
            "What theoretical gaps exist in understanding grokking in neural networks?",
            "gap_detection",
        ),
        # recent (3)
        ("What are the latest advances in diffusion model efficiency?", "recent"),
        ("What is the state-of-the-art for long-context language modeling?", "recent"),
        ("What are the newest methods for test-time compute scaling?", "recent"),
        # citation (2)
        (
            "What papers have built on the original Transformer architecture?",
            "citation",
        ),
        ("Which works cite the original GAN paper in generative modeling?", "citation"),
    ]

    logger.info("Starting phase 6 Redis latency evaluation")
    mlflow.set_experiment("phase6_redis_latency")
    with mlflow.start_run(run_name="phase6_redis_latency"):
        # cold run — fresh session, no cache hits
        logger.info("Running cold benchmark (no cache)")
        cold_latencies = run_latency_benchmark(queries, session_id="eval_session")
        cold_avg = sum(latency for latency, _ in cold_latencies) / len(cold_latencies)
        cold_p95 = float(np.percentile([latency for latency, _ in cold_latencies], 95))

        # warm run — same session, Redis has prior context
        logger.info("Running warm benchmark (Redis cache populated)")
        warm_latencies = run_latency_benchmark(queries, session_id="eval_session")
        warm_avg = sum(latency for latency, _ in warm_latencies) / len(warm_latencies)
        warm_p95 = float(np.percentile([latency for latency, _ in warm_latencies], 95))

        logger.info(
            "Cold: avg=%.2fs p95=%.2fs | Warm: avg=%.2fs p95=%.2fs",
            cold_avg,
            cold_p95,
            warm_avg,
            warm_p95,
        )
        logger.info("Latency reduction: %.1f%%", (cold_avg - warm_avg) / cold_avg * 100)

        # global metrics — logged once each
        mlflow.log_metric("cold_avg_latency", cold_avg)
        mlflow.log_metric("cold_p95_latency", cold_p95)
        mlflow.log_metric("warm_avg_latency", warm_avg)
        mlflow.log_metric("warm_p95_latency", warm_p95)
        mlflow.log_metric(
            "latency_reduction_pct", (cold_avg - warm_avg) / cold_avg * 100
        )

        # per-intent metrics — cold and warm logged separately
        logger.info("Logging per-intent metrics")
        for intent in ["search", "compare", "gap_detection", "recent", "citation"]:
            cold_intent = [l for l, i in cold_latencies if i == intent]
            warm_intent = [l for l, i in warm_latencies if i == intent]

            if cold_intent:
                cold_intent_avg = sum(cold_intent) / len(cold_intent)
                cold_intent_p95 = float(np.percentile(cold_intent, 95))
                mlflow.log_metric(f"cold_avg_{intent}", cold_intent_avg)
                mlflow.log_metric(f"cold_p95_{intent}", cold_intent_p95)
                logger.info(
                    "  cold %-14s avg=%.2fs p95=%.2fs",
                    intent,
                    cold_intent_avg,
                    cold_intent_p95,
                )

            if warm_intent:
                warm_intent_avg = sum(warm_intent) / len(warm_intent)
                warm_intent_p95 = float(np.percentile(warm_intent, 95))
                mlflow.log_metric(f"warm_avg_{intent}", warm_intent_avg)
                mlflow.log_metric(f"warm_p95_{intent}", warm_intent_p95)
                logger.info(
                    "  warm %-14s avg=%.2fs p95=%.2fs",
                    intent,
                    warm_intent_avg,
                    warm_intent_p95,
                )

        logger.info(
            "Evaluation complete. Cold avg: %.2fs | Warm avg: %.2fs | Reduction: %.1f%%",
            cold_avg,
            warm_avg,
            (cold_avg - warm_avg) / cold_avg * 100,
        )


if __name__ == "__main__":
    main()
