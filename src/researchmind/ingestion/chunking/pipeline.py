"""Chunking pipeline step: reads parsed_papers.jsonl → writes raw_chunks.jsonl."""

import json
import logging
from pathlib import Path

from researchmind.ingestion.chunking.interfaces import Chunker
from researchmind.ingestion.models import ParsedPaper

logger = logging.getLogger(__name__)


def chunk_papers(
    chunker: Chunker,
    parsed_papers_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Read ParsedPapers from JSONL, chunk them, write Chunks to JSONL.

    Returns (papers_succeeded, papers_failed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parsed_papers_path, "r", encoding="utf-8") as f:
        parsed_papers = [json.loads(line) for line in f if line.strip()]

    failed = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for n, raw in enumerate(parsed_papers):
            try:
                paper = ParsedPaper.model_validate(raw)
            except Exception:
                logger.exception("Failed to validate paper at index %d", n)
                failed += 1
                continue

            for chunk in chunker.chunk(paper):
                out.write(json.dumps(chunk.model_dump(mode="json")) + "\n")

            if n % 100 == 0 and n > 0:
                logger.info("Chunked %d papers so far", n - failed)

    succeeded = len(parsed_papers) - failed
    logger.info("Chunking complete: %d succeeded, %d failed", succeeded, failed)
    return succeeded, failed


def _build_chunker(strategy: str, max_words: int, overlap: int) -> Chunker:
    if strategy == "section":
        from researchmind.ingestion.chunking.section_chunker import SectionChunker
        return SectionChunker(max_words=max_words, overlap_words=overlap)
    elif strategy == "fixed":
        from researchmind.ingestion.chunking.fixed_chunker import FixedSizeChunker
        return FixedSizeChunker(chunk_words=max_words, overlap_words=overlap)
    elif strategy == "semantic":
        from researchmind.ingestion.chunking.semantic_chunker import SemanticChunker
        return SemanticChunker(chunk_size=max_words, overlap=overlap)
    else:
        raise ValueError(
            f"Unknown chunk_strategy '{strategy}'. Choose: section | fixed | semantic"
        )


def main() -> None:
    from researchmind.utils.config import load_phase_config
    from researchmind.utils.find_root import find_project_root
    from researchmind.ingestion.chunking.clean_chunk_section import clean_section, save_cleaned_chunks

    cfg = load_phase_config(find_project_root())
    ing = cfg.ingestion

    chunker = _build_chunker(ing.chunk_strategy, ing.chunk_max_words, ing.chunk_overlap_words)
    logger.info(
        "Chunking | strategy=%s max_words=%d overlap=%d",
        ing.chunk_strategy, ing.chunk_max_words, ing.chunk_overlap_words,
    )

    # Step 1: chunk parsed papers → raw_chunks.jsonl
    chunk_papers(chunker, ing.parsed_papers_path, ing.raw_chunks_path)

    # Step 2: canonicalise section headers, drop full_text/reference → chunks.jsonl
    logger.info("Cleaning section headers in %s", ing.raw_chunks_path)
    cleaned = clean_section(ing.raw_chunks_path)
    save_cleaned_chunks(cleaned, cfg.index.chunks_path)
    logger.info("Final chunks written to %s", cfg.index.chunks_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
