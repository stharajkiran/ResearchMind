import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from researchmind.ingestion.discovery import (
    ArxivSource,
    SemanticScholarRecommendationSource,
)
from researchmind.utils.find_root import find_project_root

load_dotenv()
project_root = find_project_root()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
TASKS = ["anomaly-detection", "out-of-distribution-detection"]
# Suppress noisy third-party loggers
# logging.getLogger("arxiv").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_RECOMMEND_URL = "https://api.semanticscholar.org/recommendations/v1/papers"
S2_VENUES = "CVPR,ICCV,ECCV,NeurIPS,ICML"

OOD_SEED_IDS = [
    "6ff2a434578ff2746b9283e45abf296887f48a2d",
    "547c854985629cfa9404a5ba8ca29367b5f8c25f",
    "d03ca175e2b2745126e792fdc31dfadae4c63afa",
    "35b966347dae2f0d496ea713edf03a68211838a5",
    "7f9760a76e9cf424da0b72d42f75594cefc4a329",
    "4e9e30f4702f64af5aacbb5791172c5b37510dc3",
    "23ad8fc48530ce366f8192dfb48d0f7df1dba277",
    "2e8d62277e40d465343e8dfb32ecc246f320540e",
    "9277dc70c74bcadf80dab11c28ead83fd085deec",
    "78d80c343d36baaf89f18e12d325cf6309fb6c8f",
    "a23c0a89bd21bd2481fedcdd6d1ac891c6c06bdc",
    "aa207668318fec38d60b79f407fb64982e46fce9",
    "60264ea13394896aeb7bb514f761a287cab1a54d",
]




def fetch_arxiv_ids_from_s2(
    queries: list[str],
    target_per_query: int = 200,  # target arxiv IDs, not S2 results
) -> list[str]:
    """
    Fetches arXiv IDs from Semantic Scholar based on search queries.

    Args:
        queries (list[str]): List of search queries.
        target_per_query (int, optional): Target number of arXiv IDs per query. Defaults to 200.

    Returns:
        list[str]: List of unique arXiv IDs.
    """
    api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    headers = {"x-api-key": api_key}
    all_ids = set()

    for query in queries:
        logger.info("S2 query: %r", query)
        token = None
        ids_this_query = 0

        with tqdm(total=target_per_query, desc=f"S2: {query}") as pbar:
            while ids_this_query < target_per_query:
                params = {
                    "query": query,
                    "fields": "externalIds,title,year",
                    "limit": 100,
                }
                if token:
                    params["token"] = token

                response = httpx.get(
                    S2_SEARCH_URL, headers=headers, params=params, timeout=30
                )
                response.raise_for_status()
                data = response.json()

                papers = data.get("data", [])
                if not papers:
                    logger.info("No more results for query: %r", query)
                    break

                for paper in papers:
                    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
                    if arxiv_id and arxiv_id not in all_ids:
                        all_ids.add(arxiv_id)
                        ids_this_query += 1
                        pbar.update(1)
                    if ids_this_query >= target_per_query:
                        break

                token = data.get("token")
                if not token:
                    logger.info("Pagination exhausted for query: %r", query)
                    break

                time.sleep(1.0)

        logger.info("Query %r done: %d IDs collected", query, ids_this_query)

    logger.info("Total unique arxiv IDs: %d", len(all_ids))
    return list(all_ids)




def run_arxiv_id_fetch_and_save():
    logger.info("run_arxiv_id_fetch_and_save: starting")
    from researchmind.utils.config import load_phase_config
    cfg = load_phase_config(project_root)
    papers = ArxivSource().fetch_by_query(
        categories=cfg.corpus.categories,
        start_date=date.fromisoformat(cfg.corpus.date_from),
        end_date=date.fromisoformat(cfg.corpus.date_to),
        max_results=cfg.corpus.max_results,
        keywords=cfg.corpus.keywords or None,
    )
    output_path = project_root / "data" / "processed" / "demo" / "arxiv_demo_papers_cv.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info("run_arxiv_id_fetch_and_save: saved %d papers to %s", len(papers), output_path)


def run_ssd_fetch_and_save():
    logger.info("run_ssd_fetch_and_save: starting")
    arxiv_ids = fetch_arxiv_ids_from_s2(queries=["out-of-distribution", "anomaly detection"])
    papers = ArxivSource().fetch_by_ids(arxiv_ids)
    output_path = project_root / "data" / "processed" / "demo" / "ssd_demo_papers.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info("run_ssd_fetch_and_save: saved %d papers to %s", len(papers), output_path)


def run_ss2_recommendation_fetch_and_save():
    logger.info("run_ss2_recommendation_fetch_and_save: starting")
    papers = SemanticScholarRecommendationSource().fetch_by_ids(OOD_SEED_IDS)
    output_path = project_root / "data" / "processed" / "demo" / "ss2_recommendation_demo_papers.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info("run_ss2_recommendation_fetch_and_save: saved %d papers to %s", len(papers), output_path)


if __name__ == "__main__":
    # run_arxiv_id_fetch_and_save()
    run_ss2_recommendation_fetch_and_save()
    # run_ssd_fetch_and_save()
