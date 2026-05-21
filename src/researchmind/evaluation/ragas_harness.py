import os
import argparse

from pathlib import Path
import logging
import httpx
from datasets import Dataset
import json
import numpy as np
from researchmind.utils.experiment_logger import ExperimentLogger, MLflowLogger
from researchmind.ingestion.models import RAGResponse
from researchmind.evaluation.test_set_generator import TestQuery
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from dotenv import load_dotenv
from collections import defaultdict
from anthropic import Anthropic
from researchmind.utils.find_root import find_project_root
from langchain_anthropic import ChatAnthropic
from researchmind.utils.logging import configure_logging
from datetime import datetime
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import pandas as pd

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("RAGAS_Harness")
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_retries=5,
    timeout=60,
)


def load_queries(test_set_path: Path) -> list[TestQuery]:
    """Load test queries from a JSON file. Each dictionary should be a JSON object matching the TestQuery schema."""
    queries = []
    try:
        with open(test_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            queries.append(TestQuery(**item))
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s", e)
    return queries


def query_rag(query: str, api_url: str, retries: int = 3) -> RAGResponse:
    for attempt in range(retries):
        try:
            response = httpx.post(api_url, json={"query": query}, timeout=120.0)
            response.raise_for_status()
            return RAGResponse(**response.json())
        except Exception as exc:
            if attempt == retries - 1:
                raise
            logger.warning("Retry %d for query %s: %s", attempt + 1, query[:50], exc)


def build_ragas_dataset(
    queries: list[TestQuery], responses: list[RAGResponse]
) -> Dataset:
    if len(queries) != len(responses):
        raise ValueError(
            f"Mismatched input lengths: {len(queries)} queries vs {len(responses)} responses"
        )

    return Dataset.from_dict(
        {
            "question": [q.question for q in queries],
            "answer": [r.response for r in responses],
            "contexts": [r.contexts for r in responses],  # trancuate
            "ground_truth": [q.ground_truth for q in queries],
        }
    )


def evaluate_ragas(dataset: Dataset, batch_size: int) -> dict[str, float]:

    results = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating RAGAS batches"):
        batch_dataset = dataset.select(range(i, min(i + batch_size, len(dataset))))

        try:
            batch_result = evaluate(
                batch_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=llm,
                show_progress=True,
            )
            results.append(batch_result.to_pandas())
            tqdm.write(f"Processed batch {i // batch_size + 1}")

        except Exception as e:
            logger.error("Batch %d failed: %s", i // batch_size, e)

    combined = pd.concat(results, ignore_index=True)
    METRIC_NAMES = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    return {
        col: combined[col].tolist() for col in METRIC_NAMES if col in combined.columns
    }


def main(args: argparse.Namespace):
    from researchmind.utils.config import load_phase_config
    project_root = find_project_root()
    cfg = load_phase_config(project_root)

    test_set_path = args.test_set_path or cfg.evaluation.test_set_path
    responses_output_path = args.responses_output_path or cfg.evaluation.ragas_responses_path

    logger.info("Loading test queries from %s...", test_set_path)
    queries = load_queries(test_set_path)

    if not responses_output_path.exists():
        try:
            logger.info("Querying RAG system for each test query...")
            responses = []
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task_id = progress.add_task(
                    f"[cyan]Processing {len(queries)} RAG queries...",
                    total=len(queries),
                )
                for q in queries:
                    progress.update(
                        task_id,
                        description=f"[cyan]Querying: {q.question[:60]}...",
                    )
                    try:
                        responses.append(query_rag(q.question, args.api_url))
                    except Exception as e:
                        logger.error("Failed query: %s — %s", q.question[:50], e)
                        responses.append(None)
                    finally:
                        progress.advance(task_id)

            responses_output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(responses_output_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump(mode="json") for r in responses], f, indent=2)
            logger.info("Saved RAG responses to %s", responses_output_path)

        except Exception as e:
            logger.error("Failed to save/load RAG responses: %s", e)
    else:
        logger.info(
            "loading RAG responses from saved file for evaluation from %s",
            responses_output_path,
        )
        # load  the responses from processed/RAGA_response_sample.json
        with open(responses_output_path, "r", encoding="utf-8") as f:
            responses_data = json.load(f)
        responses = [RAGResponse(**r) for r in responses_data]

    logger.info("Building category index list ...")
    # group indices by category
    category_indices = defaultdict(list)
    for i, query in enumerate(queries):
        category_indices[query.category].append(i)

    exp_logger: ExperimentLogger = MLflowLogger()
    # evaluate per category
    logger.info("Evaluating RAGAS metrics for each category...")
    with exp_logger.start_run("RAGAS_Evaluation_semantic", experiment_name="RAGAS_Evaluation"):
        for category, indices in tqdm(
            category_indices.items(), desc="Evaluating categories", unit="category"
        ):
            cat_queries = [queries[i] for i in indices]
            cat_responses = [responses[i] for i in indices]
            cat_dataset = build_ragas_dataset(cat_queries, cat_responses)
            logger.info(
                "Evaluating category '%s' with %d queries...",
                category,
                len(cat_queries),
            )
            cat_scores = evaluate_ragas(cat_dataset, args.batch_size)
            logger.info(
                "Category '%s' evaluation completed. Scores: %s", category, cat_scores
            )
            avg_scores = {k: float(np.nanmean(v)) for k, v in cat_scores.items()}
            exp_logger.log_metrics({f"{category}/{k}": v for k, v in avg_scores.items()})
            logger.info(
                "Logged category '%s' metrics to MLflow: %s", category, avg_scores
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Harness")
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("BACKEND_URL", "http://localhost:8000") + "/rag",
        help="RAG API endpoint URL. Defaults to BACKEND_URL env var + /rag, or http://localhost:8000/rag",
    )
    # Optional path overrides — config values are used when not supplied
    parser.add_argument("--test-set-path", type=Path, default=None,
                        help="Overrides evaluation.test_set from config.")
    parser.add_argument("--responses-output-path", type=Path, default=None,
                        help="Overrides evaluation.ragas_responses from config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    from researchmind.utils.config import load_phase_config
    cfg = load_phase_config(find_project_root())
    session_log_path = (
        cfg.evaluation.log_dir
        / f"session_{cfg.evaluation.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    configure_logging(session_log_path, logger)
    main(args)
