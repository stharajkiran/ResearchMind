import os
import argparse

from pathlib import Path
import logging
import httpx
from datasets import Dataset
import json
import numpy as np
import mlflow
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

# client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
# llm = llm_factory("claude-sonnet-4-6", client=client)

# llm = ChatAnthropic(
#     model="claude-haiku-4-5-20251001", api_key=os.environ.get("ANTHROPIC_API_KEY")
# )

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
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning("Retry %d for query: %s", attempt + 1, query[:50])


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
    METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    return {col: combined[col].tolist() for col in METRIC_NAMES if col in combined.columns}


def main(args: argparse.Namespace):
    project_root = find_project_root()
    test_set_path = project_root / args.test_set_path
    logger.info("Loading test queries...")
    queries = load_queries(test_set_path)

    # save the responses to a JSON file for later analysis and reproducibility
    responses_output_path = project_root / args.responses_output_path

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

    mlflow.set_experiment("RAGAS_Evaluation")
    # evaluate per category
    logger.info("Evaluating RAGAS metrics for each category...")
    with mlflow.start_run(run_name="RAGAS_Evaluation_semantic") as run:
        for category, indices in tqdm(category_indices.items(), desc="Evaluating categories", unit="category"):
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
            # log to MLflow with category prefix
            avg_scores = {k: float(np.nanmean(v)) for k, v in cat_scores.items()}
            mlflow.log_metrics({f"{category}/{k}": v for k, v in avg_scores.items()})
            logger.info(
                "Logged category '%s' metrics to MLflow: %s", category, avg_scores
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Harness")
    parser.add_argument(
        "--test-set-path",
        type=Path,
        default=Path("data/processed/200_test_queries_set.json"),
        help="Path to the JSON file containing the test queries. Defaults to data/processed/200_test_queries_set.json",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/rag",
        help="URL of the RAG API endpoint to query. Defaults to http://localhost:8000/rag",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=find_project_root(),
        help="Root directory of the project. Defaults to auto-detected repository root.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/ragas_evaluation"),
        help="Directory to save evaluation logs. Defaults to logs/ragas_evaluation",
    )
    parser.add_argument(
        "--responses-output-path",
        type=Path,
        default=Path("data/processed/RAGA_responses.json"),
        help="Path to save the raw RAG responses for later analysis. Defaults to data/processed/RAGA_responses.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="batch size of data to send for evaluation to RAGAS. Defaults to 20",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="RAGAS_Evaluation_Fixed",
        help="MLflow experiment name for logging results. Defaults to RAGAS_Evaluation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logs_dir = args.root_dir / "logs" / "ragas_evaluation"
    session_log_path = (
        logs_dir / f"session_{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    configure_logging(session_log_path, logger)

    main(args)
