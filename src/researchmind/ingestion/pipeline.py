import json
from datetime import date
from pathlib import Path

from researchmind.ingestion.arxiv_client import fetch_papers


def run(
    categories: list[str],
    start_date: date,
    end_date: date,
    max_results: int = 5000,
    output_path: Path = Path("data/processed/papers.jsonl"),
) -> None:

    papers = fetch_papers(categories, start_date, end_date, max_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.model_dump(mode="json")) + "\n")

if __name__ == "__main__":
    print("Running ingestion pipeline...")
    run(
        categories=["cs.LG", "cs.AI", "cs.CL"],
        start_date=date(2024, 1, 1),
        end_date=date(2026, 12, 31),
        max_results=5000,
    )