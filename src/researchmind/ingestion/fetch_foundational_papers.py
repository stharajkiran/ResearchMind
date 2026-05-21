"""
Corpus builder for the domain-specific OOD / anomaly-detection corpus.

Two passes:
  1. Seeds  — landmark OOD, anomaly detection, and DINO-based CV papers fetched
              directly from arXiv by ID.
  2. Expand — Semantic Scholar references for each seed → discover cited arXiv papers,
              then batch-fetch from arXiv.

Output: data/processed/papers.jsonl
"""

import json
import logging
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from researchmind.ingestion.discovery import ArxivSource, SemanticScholarCitationSource
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging

load_dotenv()

project_root = find_project_root()
logger = logging.getLogger(__name__)

# Landmark papers in OOD detection, visual anomaly detection,
# novel category discovery, DINO-based methods, and interpretable CV.
SEEDS = [
    # OOD detection — foundational
    "1610.02136",  # Baseline for OOD detection (Hendrycks & Gimpel)
    "1807.03888",  # Deep Anomaly Detection with Outlier Exposure
    "2002.11297",  # CSI: Novelty Detection via Contrastive Learning
    "2106.03004",  # ReAct: Out-of-distribution Detection with Rectified Activations
    "2108.11635",  # DICE: Leveraging Sparsification for OOD Detection
    "2209.15639",  # VIM: Out-of-distribution with Virtual-logit Matching
    "2207.07843",  # KNN-based OOD Detection
    "2110.11334",  # OpenOOD Benchmark
    "1706.02690",  # ODIN: Principled OOD Detection
    "2205.00693",  # GradNorm for OOD detection
    # Visual anomaly detection
    "2205.09510",  # PatchCore: Towards Total Recall in Industrial Anomaly Detection
    "2004.14435",  # Uninformed Students for Anomaly Detection
    "2106.08265",  # PADIM: Patch Distribution Modeling
    "2111.07677",  # SimpleNet: A Simple Network for Image Anomaly Detection
    "2208.03943",  # RD4AD: Reverse Distillation for Anomaly Detection
    "2203.08736",  # FastFlow: Unsupervised Anomaly Detection via NF
    # DINO / DINOv2 based methods
    "2104.14294",  # DINO: Self-distillation with no labels
    "2304.07193",  # DINOv2: Learning robust visual features
    "2209.07399",  # Masked Image Modeling with DINO features
    # Novel category discovery / open-world recognition
    "2004.12186",  # Automatically Discovering and Learning Novel Visual Categories
    "2106.10272",  # Novel Class Discovery with Unified Contrastive Learning
    "2110.03174",  # OpenLDN: Open-world Label Discovery
    "2301.01413",  # Parametric Classification for Novel Class Discovery
    # Prototype-based and energy-based OOD
    "2010.03759",  # Energy-based OOD Detection
    "2202.05575",  # Semantic Pyramid for OOD Detection (SSD)
    "2107.02672",  # Exploring the Limits of OOD Detection
    # Benchmarks
    "2110.11334",  # OpenOOD
    "2110.06207",  # CIFAR10/100 OOD benchmarks review
]

# Deduplicate seeds in case of accidental duplicates
SEEDS = list(dict.fromkeys(SEEDS))

_arxiv = ArxivSource()
_ss_citations = SemanticScholarCitationSource()


def _save(papers: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    logger.info("Saved %d papers to %s", len(papers), path)


def pass1_seeds(seen_ids: set[str]) -> list[dict]:
    """Fetch landmark domain papers directly from arXiv by ID."""
    logger.info("Pass 1: fetching %d seed papers from arXiv", len(SEEDS))
    new_ids = [pid for pid in SEEDS if pid not in seen_ids]
    raw = _arxiv.fetch_by_ids(new_ids)
    papers = [p.model_dump(mode="json") for p in raw]
    seen_ids.update(p["paper_id"].split("v")[0] for p in papers)
    logger.info("Pass 1 complete: %d seeds fetched", len(papers))
    return papers


def pass2_expand(seen_ids: set[str]) -> list[dict]:
    """Fetch references for each seed via Semantic Scholar, then pull from arXiv.

    Depth-1 outbound expansion only — going deeper is exponential and
    hurts corpus precision for a domain-specific build.
    """
    logger.info("Pass 2: citation expansion via Semantic Scholar references")
    discovered: set[str] = set()

    for seed_id in tqdm(SEEDS, desc="SS expansion"):
        ref_ids = _ss_citations.get_references(seed_id)
        for rid in ref_ids:
            normalised = rid.split("v")[0]
            if normalised not in seen_ids:
                discovered.add(normalised)
        time.sleep(0.5)

    logger.info("Discovered %d unique referenced papers via SS", len(discovered))
    if not discovered:
        return []

    raw = _arxiv.fetch_by_ids(list(discovered))
    papers = [p.model_dump(mode="json") for p in raw]
    seen_ids.update(p["paper_id"].split("v")[0] for p in papers)
    logger.info("Pass 2 complete: %d expanded papers fetched", len(papers))
    return papers


def fetch_foundational_papers(
    categories: list[str],
    start_date: date,
    end_date: date,
    max_results: int,
) -> None:
    configure_logging(project_root / "logs" / "fetch_foundational.log", logger)
    seen_ids: set[str] = set()
    all_papers: list[dict] = []

    seeds = pass1_seeds(seen_ids)
    all_papers.extend(seeds)

    expanded = pass2_expand(seen_ids)
    all_papers.extend(expanded)

    # Backfill remaining quota with category + date range search if needed
    remaining = max_results - len(all_papers)
    if remaining > 0:
        logger.info("Pass 3: backfill %d papers from arXiv category search", remaining)
        time.sleep(10)
        for year in range(start_date.year, end_date.year + 1):
            yearly = _arxiv.fetch_by_query(
                categories=categories,
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                max_results=remaining + 50,
            )
            added_this_year = 0
            for p in yearly:
                norm = p.paper_id.split("v")[0]
                if norm not in seen_ids:
                    all_papers.append(p.model_dump(mode="json"))
                    seen_ids.add(norm)
                    added_this_year += 1
            logger.info("Year %d: added %d papers, total so far: %d", year, added_this_year, len(all_papers))
            time.sleep(5)
            if len(all_papers) >= max_results:
                break

    out_path = project_root / "data" / "processed" / "papers.jsonl"
    _save(all_papers, out_path)
    logger.info("Done. Total: %d / %d max_results", len(all_papers), max_results)


if __name__ == "__main__":
    from researchmind.utils.config import load_phase_config

    cfg = load_phase_config(project_root)
    fetch_foundational_papers(
        categories=cfg.corpus.categories,
        start_date=date.fromisoformat(cfg.corpus.date_from),
        end_date=date.fromisoformat(cfg.corpus.date_to),
        max_results=cfg.corpus.max_results,
    )
