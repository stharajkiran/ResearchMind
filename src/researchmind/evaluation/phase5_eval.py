from dataclasses import dataclass
import logging
import mlflow
from tqdm import tqdm
from datetime import datetime

from researchmind.agent.graph import build_graph
from researchmind.agent.tracing import configure_tracing
from researchmind.graph.citation_graph import load_graph
from researchmind.guardrails.pipeline import ValidatorPipeline
from researchmind.utils.find_root import find_project_root
from researchmind.utils.llm_client import ResearchMindLLM
from researchmind.retrieval.retriever import RetrieverService
from researchmind.mcp_server.config import Config
from researchmind.utils.logging import configure_logging


project_root = find_project_root()
# build logger for the API module
logger = logging.getLogger("phase_5_eval")

TEST_QUERIES = [
    # search — 20 queries
    ("What is knowledge distillation?", "search"),
    ("How does batch normalization work?", "search"),
    (
        "What is the mathematical formulation of the cross-entropy loss function?",
        "search",
    ),
    (
        "How does batch normalization stabilize training in deep neural networks?",
        "search",
    ),
    ("What are the key architectural components of the Transformer model?", "search"),
    ("How is the F1 score calculated for imbalanced classification tasks?", "search"),
    (
        "What is the role of the attention mechanism in sequence-to-sequence models?",
        "search",
    ),
    (
        "How does dropout regularization prevent overfitting in neural networks?",
        "search",
    ),
    (
        "What is the definition of the BLEU metric in machine translation evaluation?",
        "search",
    ),
    (
        "How does Adam optimizer adapt learning rates for different parameters?",
        "search",
    ),
    

    # gap_detection — 20 queries
    ("What problems in OOD detection remain unsolved?", "gap_detection"),
    ("What are the open challenges in anomaly detection?", "gap_detection"),
    (
        "What are the primary theoretical gaps in understanding the generalization bounds of overparameterized neural networks?",
        "gap_detection",
    ),
    (
        "Which aspects of out-of-distribution detection remain unresolved in current self-supervised learning frameworks?",
        "gap_detection",
    ),
    (
        "Where do current diffusion models fall short in terms of computational efficiency for real-time generation?",
        "gap_detection",
    ),
    (
        "What open problems exist in the robustness of vision transformers against adversarial perturbations?",
        "gap_detection",
    ),
    (
        "Which theoretical limitations hinder the scalability of reinforcement learning algorithms in continuous control tasks?",
        "gap_detection",
    ),
    (
        "What are the unresolved challenges in aligning multi-modal models with human ethical guidelines?",
        "gap_detection",
    ),
    (
        "Where does the current literature lack empirical evidence regarding the interpretability of attention mechanisms?",
        "gap_detection",
    ),
    (
        "What are the critical gaps in benchmarking fairness metrics for deep learning models in healthcare applications?",
        "gap_detection",
    ),
]


@dataclass
class ValidatorStats:
    name: str
    total_passed: int = 0
    total_score: float = 0.0
    count: int = 0


class StatsDict(dict):
    def __missing__(self, key):
        # This runs automatically when a key isn't found
        new_stats = ValidatorStats(name=key)
        self[key] = new_stats
        return new_stats


def run_eval(agent):
    intents_list = ["search", "gap_detection"]

    logger.info("Starting phase5 routing evaluation with %d queries", len(TEST_QUERIES))

    mlflow.set_experiment("Phase 5 Evaluation")
    with mlflow.start_run(run_name=f"phase5_routing_eval"):
        # Log Hyperparameters
        mlflow.log_params(
            {
                "model": "qwen3.5:9b",
                "temp": 0.0,
                "tier": "fast",
                "num_test_queries": len(TEST_QUERIES),
            }
        )
        metrics = StatsDict()
        total_queries = 0
        total_blocked = 0
        blocked_queries = []
        for i, (query, intent) in enumerate(
            tqdm(TEST_QUERIES, desc="Routing eval", total=len(TEST_QUERIES))
        ):
            # keep the search or gap_detection queries only
            if intent.lower() not in intents_list:
                continue
            state = {
                "query": query,
                "intent": "",
                "retrieved_chunks": [],
                "compared_chunks": None,
                "tool_call_history": [],
                "session_id": "",
                "final_answer": None,
                "validation_result": None,
            }

            try:
                # invoke the agent
                result = agent.invoke(state)
                # read the list of validation result (list[ValidationResult])
                validation_results = result.get("validation_result", None)

                # accumulate the scores from the validation result for each category
                if validation_results is not None:
                    status = "BLOCKED" if validation_results.blocked else "PASSED"
                    tqdm.write(f"[{status}] {query[:60]}")
                    for v in validation_results.results:
                        tqdm.write(
                            f"  {v.validator}: passed={v.passed} score={v.score:.3f}"
                        )
                    if validation_results.blocked:
                        total_blocked += 1
                    total_queries += 1

                    for v in validation_results.results:
                        category = v.validator
                        metrics[category].total_passed += int(v.passed)
                        metrics[category].total_score += v.score
                        metrics[category].count += 1

            except ValueError as e:
                tqdm.write(f"[BLOCKED] {query[:60]} — {str(e)}")
                # add the blocked query to mlflow artifacts
                blocked_queries.append(
                    {"intent": intent, "query": query, "reason": str(e)}
                )
                total_blocked += 1
                total_queries += 1
                continue
            except Exception as e:
                logger.error("Error processing query '%s': %s", query, str(e))
                continue

        # After processing all queries, log the aggregated metrics in mlflow
        mlflow.log_dict({"blocked": blocked_queries}, "blocked_queries.json")
        total_blocks_rate = total_blocked / total_queries if total_queries > 0 else 0.0
        mlflow.log_metric("total_blocks_rate", total_blocks_rate)
        for category, stats in metrics.items():
            pass_rate = stats.total_passed / stats.count if stats.count > 0 else 0.0
            avg_score = stats.total_score / stats.count if stats.count > 0 else 0.0

            mlflow.log_metric(f"{category}_pass_rate", pass_rate)
            mlflow.log_metric(f"{category}_avg_score", avg_score)
            logger.info(
                "Validator: %s | Pass Rate: %.2f%% | Average Score: %.4f ",
                category,
                pass_rate * 100,
                avg_score,
            )

    return metrics


if __name__ == "__main__":
    # Set up experiment logging
    experiment_name = "phase5_routing_eval"
    logs_dir = project_root / "logs" / experiment_name
    session_log_path = (
        logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    configure_logging(session_log_path, logger)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Load retriever, citation graph, and build agent
    project_root = find_project_root()
    configure_tracing()

    retriever = RetrieverService(
        Config.artifact_dir,
        collection_name="researchmind",
        chunks_path=Config.chunks_path,
    )
    retriever.load(Config.chunks_path)
    citation_graph = load_graph(project_root / "artifacts" / "citation_graph.pkl")
    pipeline = ValidatorPipeline(
        corpus_paper_ids=retriever.corpus_paper_ids, encoder=retriever.encoder
    )
    client = ResearchMindLLM()
    agent = build_graph(
        retriever=retriever,
        llm=client,
        citation_graph=citation_graph,
        pipeline=pipeline,
    )

    # Run evaluation
    stats = run_eval(agent)
    logger.info("Final evaluation stats: %s", stats)
