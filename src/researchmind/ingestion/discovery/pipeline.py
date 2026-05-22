"""Discovery pipeline step: collect paper IDs from configured sources, fetch full
metadata from arXiv, enrich with SS citation metadata, write to papers.jsonl.

Flow (sources configured per phase via ingestion.discovery.sources in YAML):
  arxiv           → ArxivSource.fetch_by_query (category + keyword search)
  ss_search       → SemanticScholarSearch.collect_ids + ArxivSource.fetch_by_ids
  ss_recommendations → SemanticScholarRecommendationSource.collect_ids + ArxivSource.fetch_by_ids

All sources are pooled and deduplicated before the final arXiv fetch.
"""

import json
import logging
from pathlib import Path

from researchmind.ingestion.models import RawPaper

logger = logging.getLogger(__name__)


def run(papers: list[RawPaper], output_path: Path, enrich: bool = True) -> None:
    """Enrich papers with SS metadata and write to JSONL."""
    if enrich:
        from researchmind.ingestion.discovery.semantic_scholar_enricher import SemanticScholarEnricher
        logger.info("Enriching %d papers with Semantic Scholar metadata...", len(papers))
        papers = SemanticScholarEnricher().enrich(papers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for paper in papers:
            out.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info("Wrote %d papers to %s", len(papers), output_path)


def main() -> None:
    from datetime import date
    from researchmind.utils.config import load_phase_config
    from researchmind.utils.find_root import find_project_root
    from researchmind.ingestion.discovery.arxiv_source import ArxivSource
    from researchmind.ingestion.discovery.semantic_scholar_search import SemanticScholarSearch
    from researchmind.ingestion.discovery.semantic_scholar_recommendations import SemanticScholarRecommendationSource

    cfg = load_phase_config(find_project_root())
    corpus = cfg.corpus
    discovery = cfg.ingestion.discovery
    active_sources = discovery.sources

    logger.info("Discovery | phase=%s sources=%s", cfg.name, active_sources)

    arxiv_source = ArxivSource()

    # --- Collect arXiv IDs from SS sources ---
    extra_ids: set[str] = set()

    if "ss_search" in active_sources:
        if not corpus.keywords:
            logger.warning("ss_search configured but corpus.keywords is empty — skipping")
        else:
            ss_ids = SemanticScholarSearch().collect_ids(
                identifiers=corpus.keywords,
                max_per_query=discovery.ss_search_max_per_query,
            )
            extra_ids.update(ss_ids)
            logger.info("ss_search: collected %d IDs", len(ss_ids))

    if "ss_recommendations" in active_sources:
        if not discovery.ss_rec_seed_ids:
            logger.warning("ss_recommendations configured but no seed_ids found — skipping")
        else:
            rec_ids = SemanticScholarRecommendationSource(
                max_recommendations=discovery.ss_rec_max_recommendations
            ).collect_ids(discovery.ss_rec_seed_ids)
            extra_ids.update(rec_ids)
            logger.info("ss_recommendations: collected %d IDs", len(rec_ids))

    # --- Fetch full papers from arXiv ---
    papers: list[RawPaper] = []

    if "arxiv" in active_sources:
        arxiv_papers = arxiv_source.fetch_by_query(
            categories=corpus.categories,
            start_date=date.fromisoformat(corpus.date_from),
            end_date=date.fromisoformat(corpus.date_to),
            max_results=corpus.max_results,
            keywords=corpus.keywords or None,
        )
        papers.extend(arxiv_papers)
        logger.info("arxiv: fetched %d papers", len(arxiv_papers))

    if extra_ids:
        existing_ids = {p.paper_id.split("v")[0] for p in papers}
        new_ids = [pid for pid in extra_ids if pid.split("v")[0] not in existing_ids]
        if new_ids:
            extra_papers = arxiv_source.fetch_by_ids(new_ids)
            papers.extend(extra_papers)
            logger.info("Extra sources: fetched %d additional papers", len(extra_papers))

    logger.info("Total papers before enrichment: %d", len(papers))
    run(papers, cfg.ingestion.papers_path, enrich=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
