import json
from datetime import date
from pathlib import Path

from researchmind.ingestion.arxiv_client import fetch_papers
from researchmind.ingestion.fetch_foundational_papers import fetch_foundational_papers
from researchmind.utils.find_root import find_project_root

project_root = find_project_root()
 
def run(
    categories: list[str],
    start_date: date,
    end_date: date,
    max_results: int = 5000,
    output_path: Path = project_root / "data" / "processed" / "papers.jsonl",
    foundational: bool = True,
) -> None:

    if foundational:
        print("Fetching foundational papers...")
        # Foundational papers are fetched separately with a more complex relevance-based strategy.
        # See fetch_foundational_papers.py for details.

        fetch_foundational_papers(categories, start_date, end_date, max_results)
    else:
        print("Fetching papers with simple query...")
        papers = fetch_papers(categories, start_date, end_date, max_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper.model_dump(mode="json")) + "\n")


if __name__ == "__main__":
    from researchmind.utils.config import load_phase_config
    cfg = load_phase_config()
    print(f"Running ingestion pipeline for phase={cfg.phase}...")
    run(
        categories=cfg.corpus_categories,
        start_date=date.fromisoformat(cfg.corpus_date_from),
        end_date=date.fromisoformat(cfg.corpus_date_to),
        max_results=cfg.corpus_max_results,
    )
