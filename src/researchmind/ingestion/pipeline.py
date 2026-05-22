"""Top-level ingestion orchestrator.

Chains all four ingestion steps in sequence:
  1. discovery  — fetch papers from arXiv + enrich with SS metadata
  2. download   — download PDFs
  3. parse      — extract text and sections from PDFs
  4. chunk      — split into chunks + canonicalise sections

Each step can also be run independently via its own pipeline module.
The Taskfile (task ingest) is the primary way to invoke the full pipeline.
"""

import logging

logger = logging.getLogger(__name__)


def run_all() -> None:
    """Run all four ingestion steps end-to-end using the active phase config."""
    import researchmind.ingestion.discovery.pipeline as discovery_pipeline
    import researchmind.ingestion.download.pipeline as download_pipeline
    import researchmind.ingestion.parsing.pipeline as parsing_pipeline
    import researchmind.ingestion.chunking.pipeline as chunking_pipeline

    logger.info("=== Ingestion pipeline start ===")

    logger.info("--- Step 1/4: Discovery ---")
    discovery_pipeline.main()

    logger.info("--- Step 2/4: Download ---")
    download_pipeline.main()

    logger.info("--- Step 3/4: Parse ---")
    parsing_pipeline.main()

    logger.info("--- Step 4/4: Chunk ---")
    chunking_pipeline.main()

    logger.info("=== Ingestion pipeline complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_all()
