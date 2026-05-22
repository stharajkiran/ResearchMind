import asyncio
import logging
from pathlib import Path

from researchmind.ingestion.download.downloader import HttpPDFDownloader
from researchmind.ingestion.models import RawPaper
from researchmind.utils.config import load_phase_config
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)


def load_papers(papers_path: Path) -> list[RawPaper]:
    with papers_path.open("r", encoding="utf-8") as f:
        return [RawPaper.model_validate_json(line) for line in f if line.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_phase_config(find_project_root())
    papers = load_papers(cfg.ingestion.papers_path)
    logger.info("Downloading %d PDFs to %s | phase=%s", len(papers), cfg.ingestion.pdf_dir, cfg.name)

    results = asyncio.run(
        HttpPDFDownloader().download_batch(
            papers=papers,
            dest_dir=cfg.ingestion.pdf_dir,
            concurrency=cfg.ingestion.download_concurrency,
        )
    )
    failed = results.count(None)
    logger.info("Done. %d succeeded, %d failed", len(results) - failed, failed)


if __name__ == "__main__":
    main()
