import json
import logging
from datetime import date
from pathlib import Path

from researchmind.ingestion.discovery.interfaces import PaperSource
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)
project_root = find_project_root()


def run(
    source: PaperSource,
    categories: list[str],
    start_date: date,
    end_date: date,
    max_results: int = 5000,
    output_path: Path = project_root / "data" / "processed" / "papers.jsonl",
) -> None:
    logger.info(
        "Fetching papers via %s (phase categories=%s, max=%d)...",
        type(source).__name__, categories, max_results,
    )
    papers = source.fetch_papers(categories, start_date, end_date, max_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")
    logger.info("Wrote %d papers to %s", len(papers), output_path)


if __name__ == "__main__":
    from researchmind.ingestion.discovery import ArxivSource
    from researchmind.utils.config import load_phase_config

    cfg = load_phase_config(find_project_root())
    logger.info("Running ingestion pipeline for phase=%s...", cfg.phase)
    run(
        source=ArxivSource(),
        categories=cfg.corpus.categories,
        start_date=date.fromisoformat(cfg.corpus.date_from),
        end_date=date.fromisoformat(cfg.corpus.date_to),
        max_results=cfg.corpus.max_results,
    )
