"""
Snowball corpus expansion for Phase 2.

Three passes:
  1. Seeds  — 50 landmark ML papers fetched directly from arXiv by ID
  2. Expand — OpenAlex referenced_works for each seed → discover cited arXiv papers
  3. Backfill — relevance-sorted arXiv query (2018–2024) to reach TARGET quota

Output: data/processed/papers_foundational.jsonl
This file is merged with papers.jsonl at index-build time (build_indexes.py).
Phase 1 artifacts are untouched.
"""

import json
import logging
import time
from datetime import date
from pathlib import Path

from tqdm import tqdm

from researchmind.ingestion.arxiv_client import fetch_papers, fetch_papers_by_ids
from researchmind.ingestion.openalex_client import get_referenced_arxiv_ids
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging

from dotenv import load_dotenv

load_dotenv()

project_root = find_project_root()
logger = logging.getLogger(__name__)




# 50 landmark ML papers (arXiv IDs, no version suffix)
SEEDS = [
    # Transformers and attention
    "1706.03762",  # Attention Is All You Need
    "1409.0473",   # Bahdanau Attention (Neural MT)
    "1607.06450",  # Layer Normalization
    # BERT family
    "1810.04805",  # BERT
    "1907.11692",  # RoBERTa
    "1906.08237",  # XLNet
    "1909.11942",  # ALBERT
    "1910.01108",  # DistilBERT
    "1911.02116",  # XLM-R
    # Seq2seq / T5 family
    "1910.10683",  # T5
    "1706.05098",  # ConvS2S
    # GPT family
    "2005.14165",  # GPT-3
    "2203.15556",  # Chinchilla (scaling laws)
    "2204.02311",  # PaLM
    "2302.13971",  # LLaMA
    "2307.09288",  # LLaMA 2
    "2310.06825",  # Mistral
    # Instruction tuning and alignment
    "2203.02155",  # InstructGPT / RLHF
    "2109.01652",  # FLAN
    "2212.10560",  # Constitutional AI
    "2305.18290",  # DPO
    # Efficient attention
    "2205.14135",  # FlashAttention
    "2104.09864",  # RoPE
    # PEFT
    "2106.09685",  # LoRA
    "2305.14314",  # QLoRA
    # Prompting and reasoning
    "2201.11903",  # Chain-of-Thought
    "2212.09561",  # Self-Consistency
    "2210.11610",  # ReAct
    "2305.10601",  # Tree of Thoughts
    # Retrieval and RAG
    "2005.11401",  # RAG (Lewis et al.)
    "2004.07213",  # DPR
    "2303.11366",  # HyDE
    "2112.09118",  # RETRO
    # Agents and tools
    "2302.07842",  # Toolformer
    "2112.00114",  # WebGPT
    # Multimodal
    "2103.00020",  # CLIP
    "2102.12092",  # DALL-E
    "2112.10752",  # Latent Diffusion Models
    # Generative models
    "1406.2661",   # GAN
    "1312.6114",   # VAE
    # Vision
    "1512.03385",  # ResNet
    "1409.1556",   # VGGNet
    "1506.02640",  # YOLO
    "1512.00567",  # Inception V3
    "2010.11929",  # Vision Transformer (ViT)
    # Training utilities
    "1502.03167",  # Batch Normalization
    "1412.6980",   # Adam optimizer
    # ELMo (pre-BERT contextual embeddings)
    "1802.05365",  # ELMo
    # Evaluation
    "2108.07258",  # BIG-Bench
    "2303.18223",  # Sparks of AGI (GPT-4 eval)
]


def _save(papers: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    logger.info("Saved %d papers to %s", len(papers), path)


def pass1_seeds(seen_ids: set[str]) -> list[dict]:
    """Fetch the 50 seed papers directly from arXiv by ID."""
    logger.info("Pass 1: fetching %d seed papers from arXiv", len(SEEDS))
    new_ids = [pid for pid in SEEDS if pid not in seen_ids]
    # Use arxiv api to get papers in batches
    raw = fetch_papers_by_ids(new_ids)
    papers = [p.model_dump(mode="json") for p in raw]
    seen_ids.update(p["paper_id"].split("v")[0] for p in papers)
    logger.info("Pass 1 complete: %d seeds fetched", len(papers))
    return papers


def pass2_expand(seen_ids: set[str]) -> list[dict]:
    """
    For each seed, call OpenAlex to get its reference list, resolve those
    to arXiv IDs, then batch-fetch from arXiv.

    This is snowball sampling depth=1 (outbound edges only).
    Deeper traversal would be exponential and is unnecessary for corpus quality.
    """
    logger.info("Pass 2: citation expansion via OpenAlex referenced_works")
    discovered: set[str] = set()

    for seed_id in tqdm(SEEDS, desc="OpenAlex expansion"):
        ref_ids = get_referenced_arxiv_ids(seed_id)
        for rid in ref_ids:
            normalised = rid.split("v")[0]
            if normalised not in seen_ids:
                discovered.add(normalised)
        time.sleep(0.2)  # polite inter-seed delay

    logger.info("Discovered %d unique referenced papers via OpenAlex", len(discovered))

    if not discovered:
        return []

    # get papers from arxiv in batches
    raw = fetch_papers_by_ids(list(discovered))
    papers = [p.model_dump(mode="json") for p in raw]
    seen_ids.update(p["paper_id"].split("v")[0] for p in papers)
    logger.info("Pass 2 complete: %d expanded papers fetched", len(papers))
    return papers


def pass3_backfill(seen_ids: set[str], quota: int, categories: list[str], start_date: date, end_date: date) -> list[dict]:
    """
    Fill remaining quota with relevance-sorted arXiv results (2018–2024).
    Uses submittedDate filter so we don't waste max_results scanning 2025-2026.
    """
    if quota <= 0:
        logger.info("Pass 3: quota already met, skipping backfill")
        return []

    logger.info("Pass 3: backfill %d papers from arXiv relevance query", quota)
    raw = fetch_papers(
        categories=categories,
        start_date=start_date,
        end_date=end_date,
        max_results=quota + 200,  # overfetch to account for dedup loss
    )
    papers = []
    for p in raw:
        normalised = p.paper_id.split("v")[0]
        if normalised not in seen_ids:
            papers.append(p.model_dump(mode="json"))
            seen_ids.add(normalised)
            if len(papers) >= quota:
                break

    logger.info("Pass 3 complete: %d backfill papers added", len(papers))
    return papers

def pass_yearly_fill(seen_ids: set[str], quota: int,  categories: list[str], start_date: date, end_date: date) -> list[dict]:
    logger.info("Pass 3: backfill %d papers from arXiv relevance query, year-by-year", quota)
    papers_per_year = quota // (end_date.year - start_date.year + 1)
    papers = []
    for year in range(start_date.year, end_date.year + 1):
        yearly = fetch_papers(
            categories=categories,
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
            max_results=papers_per_year + 50,  # small overfetch for dedup loss
        )
        for p in yearly:
            normalised = p.paper_id.split("v")[0]
            if normalised not in seen_ids:
                papers.append(p.model_dump(mode="json"))
                seen_ids.add(normalised)
        logger.info("Year %d: %d papers added, total so far: %d", year, len(papers), len(papers))
        time.sleep(5)
        if len(papers) >= quota:
            break
    logger.info("Pass 3 complete: %d backfill papers added", len(papers))
    return papers

def fetch_foundational_papers(categories, start_date, end_date, max_results) -> None:
    configure_logging(project_root / "logs" / "fetch_foundational.log", logger)
    seen_ids: set[str] = set()
    all_papers: list[dict] = []

    seeds = pass1_seeds(seen_ids)
    all_papers.extend(seeds)

    # expanded = pass2_expand(seen_ids)
    # all_papers.extend(expanded)

    remaining = max_results - len(all_papers)
    time.sleep(30)
    backfill = pass_yearly_fill(seen_ids, remaining, categories, start_date, end_date)
    all_papers.extend(backfill)

    out_path = project_root / "data" / "processed" / "papers.jsonl"
    _save(all_papers, out_path)
    logger.info("Done. Total papers: %d / %d max_results", len(all_papers), max_results)


if __name__ == "__main__":
    fetch_foundational_papers()
