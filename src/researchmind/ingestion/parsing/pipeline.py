import logging

from researchmind.ingestion.parsing.pdf_parser import parse_pdfs
from researchmind.utils.config import load_phase_config
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_phase_config(find_project_root())
    ing = cfg.ingestion
    logger.info("Parsing PDFs | phase=%s pdf_dir=%s", cfg.name, ing.pdf_dir)

    parsed, skipped, failed = parse_pdfs(
        pdf_dir=ing.pdf_dir,
        papers_path=ing.papers_path,
        output_path=ing.parsed_papers_path,
    )
    logger.info("Done. parsed=%d skipped=%d failed=%d", parsed, skipped, failed)


if __name__ == "__main__":
    main()
