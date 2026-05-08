import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import arxiv
import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from researchmind.ingestion.arxiv_client import fetch_papers_by_ids
from researchmind.ingestion.models import RawPaper
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
PWC_BASE = "https://paperswithcode.com/api/v1"
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

def fetch_arxiv_ids_from_s2_recommendations(
    seed_ids: list[str] = OOD_SEED_IDS,
) -> list[str]:
    api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    headers = {"x-api-key": api_key}

    response = httpx.post(
        S2_RECOMMEND_URL,
        headers=headers,
        json={"positivePaperIds": seed_ids, "negativePaperIds": []},
        params={"fields": "externalIds", "limit": 500},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    arxiv_ids = []
    for paper in data.get("recommendedPapers", []):
        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
        if arxiv_id:
            arxiv_ids.append(arxiv_id)

    logger.info(
        "fetch_arxiv_ids_from_s2_recommendations: got %d arXiv IDs", len(arxiv_ids)
    )
    return arxiv_ids


def fetch_papers_from_pwc(
    tasks: list[str] = TASKS,
    max_per_task: int = 200,
) -> list[str]:
    logger.info("fetch_papers_from_pwc: tasks=%s, max_per_task=%d", tasks, max_per_task)
    all_papers = set()  # arxiv_id -> paper dict, deduplicates across tasks

    for task in tasks:
        page = 1
        fetched = 0

        with tqdm(total=max_per_task, desc=f"PWC: {task}") as pbar:
            while fetched < max_per_task:
                response = httpx.get(
                    f"{PWC_BASE}/papers/",
                    params={"task": task, "page": page},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    logger.info(
                        "fetch_papers_from_pwc: no more results for task=%s at page=%d",
                        task,
                        page,
                    )
                    break

                for paper in results:
                    arxiv_id = paper.get("arxiv_id")
                    if not arxiv_id or arxiv_id in all_papers:
                        continue

                    all_papers.add(arxiv_id)
                    fetched += 1
                    pbar.update(1)

                    if fetched >= max_per_task:
                        break

                page += 1
                time.sleep(0.5)

    logger.info("fetch_papers_from_pwc: collected %d unique arxiv IDs", len(all_papers))
    return list(all_papers)


def fetch_arxiv_ids_from_s2(
    queries: list[str],
    target_per_query: int = 200,  # target arxiv IDs, not S2 results
) -> list[str]:
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


def fetch_arxiv_ids(
    keywords: list[str],
    categories: list[str],
    start_date: date,
    end_date: date,
    max_results: int = 300,
) -> list[RawPaper]:
    logger.info(
        "fetch_arxiv_ids: keywords=%s, categories=%s, date_range=%s to %s, max_results=%d",
        keywords,
        categories,
        start_date,
        end_date,
        max_results,
    )
    keyword_filter = " OR ".join(f'ti:"{kw}"' for kw in keywords)
    category_filter = " OR ".join(f"cat:{c}" for c in categories)
    # abstract_filter = " OR ".join(f'abs:"{kw}"' for kw in keywords)
    date_filter = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000 TO {end_date.strftime('%Y%m%d')}235959]"
    query = f"({category_filter}) AND ({keyword_filter}) AND {date_filter}"

    client = arxiv.Client(page_size=100, delay_seconds=10.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers = []
    skipped_future = 0
    for result in tqdm(client.results(search), desc="Fetching OOD papers"):
        if result.published.date() < start_date:
            logger.debug(
                "fetch_arxiv_ids: stopping — paper %s is before start_date",
                result.get_short_id(),
            )
            break
        if result.published.date() > end_date:
            skipped_future += 1
            continue
        papers.append(
            RawPaper(
                paper_id=result.get_short_id(),
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                categories=result.categories,
                published=result.published.date(),
                pdf_url=result.pdf_url,
            )
        )
    if skipped_future:
        logger.warning(
            "fetch_arxiv_ids: skipped %d papers published after end_date",
            skipped_future,
        )
    logger.info("fetch_arxiv_ids: returning %d papers", len(papers))
    return papers


def run_arxiv_id_fetch_and_save():
    logger.info("run_arxiv_id_fetch_and_save: starting")
    papers = fetch_arxiv_ids(
        keywords=[
            "anomaly detection",
            "out-of-distribution",
        ],
        categories=["cs.CV"],
        start_date=date(2019, 1, 1),
        end_date=date(2025, 12, 31),
        max_results=300,
    )

    filename = "arxiv_demo_papers_cv.jsonl"
    output_path = project_root / "data" / "processed" / "demo" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info(
        "run_arxiv_id_fetch_and_save: saved %d papers to %s", len(papers), output_path
    )


def run_ssd_fetch_and_save():
    logger.info("run_ssd_fetch_and_save: starting")
    queries = ["out-of-distribution", "anomaly detection"]
    arxiv_ids = fetch_arxiv_ids_from_s2(queries=queries)
    logger.info(
        "run_ssd_fetch_and_save: fetching full paper metadata for %d IDs",
        len(arxiv_ids),
    )
    papers = fetch_papers_by_ids(arxiv_ids)

    output_path = project_root / "data" / "processed" / "demo" / "ssd_demo_papers.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info(
        "run_ssd_fetch_and_save: saved %d papers to %s", len(papers), output_path
    )


def run_pwc_fetch_and_save():
    logger.info("run_pwc_fetch_and_save: starting")
    arxiv_ids = fetch_papers_from_pwc()
    logger.info(
        "run_pwc_fetch_and_save: fetching full paper metadata for %d IDs",
        len(arxiv_ids),
    )
    papers = fetch_papers_by_ids(arxiv_ids)

    output_path = project_root / "data" / "processed" / "demo" / "pwc_demo_papers.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info(
        "run_pwc_fetch_and_save: saved %d papers to %s", len(papers), output_path
    )

def run_ss2_recommendation_fetch_and_save():
    logger.info("run_ss2_recommendation_fetch_and_save: starting")
    arxiv_ids = fetch_arxiv_ids_from_s2_recommendations()
    logger.info(
        "run_ss2_recommendation_fetch_and_save: fetching full paper metadata for %d IDs",
        len(arxiv_ids),
    )
    papers = fetch_papers_by_ids(arxiv_ids)

    output_path = project_root / "data" / "processed" / "demo" / "ss2_recommendation_demo_papers.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info(
        "run_ss2_recommendation_fetch_and_save: saved %d papers to %s", len(papers), output_path
    )
if __name__ == "__main__":
    # run_arxiv_id_fetch_and_save()
    run_ss2_recommendation_fetch_and_save()
    # run_ssd_fetch_and_save()
    # run_pwc_fetch_and_save()
