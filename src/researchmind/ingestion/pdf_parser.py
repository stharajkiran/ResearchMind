import argparse
import json
import logging
import re
from pathlib import Path

import pymupdf

from researchmind.ingestion.models import ParsedPaper, RawPaper
from researchmind.utils.find_root import find_project_root


logger = logging.getLogger(__name__)

# Resolve project root once so default CLI paths stay stable regardless of cwd.
project_root = find_project_root()


HEADING_PATTERN = re.compile(
    r"^(abstract|introduction|background|related work|method(?:s)?|approach|experiment(?:s)?|results|Ablation Study|Discussion|Future Work|Limitation(?:s)?|Conclusion(?:s)?|References(?:s)?)$",
    re.IGNORECASE,
)


def configure_logging() -> None:
    """Configure console logging for parser runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_text(text: str) -> str:
    """Normalize raw text by cleaning control characters and excess whitespace."""
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_valid_line(line: str) -> bool:
    """Filter out noise and artifacts before classification."""
    line = line.strip()

    # Drop empty lines or lines that are just numbers/symbols
    if not line or len(line) < 3:
        return False
    if not re.search(r"[a-zA-Z]", line):
        return False  # Needs at least one letter

    # Deny-list for common artifacts
    artifacts = [r"←", r"→", r"↓", r"↑", r"•", r"\|"]
    if any(re.search(a, line) for a in artifacts):
        return False

    return True


def is_heading(line: str) -> bool:
    """Heuristically detect section headings in extracted PDF text."""
    # Remove common numeric prefixes like "2" or "3.1" before matching.
    candidate = re.sub(r"^\d+(?:\.\d+)*\s+", "", line.strip())
    candidate = candidate.rstrip(":")

    if HEADING_PATTERN.match(candidate) and 1 <= len(candidate.split()) <= 4:
        return True

    if not is_valid_line(candidate):
        return False

    return False


def extract_sections_from_text(text: str) -> dict[str, str]:
    """Split full paper text into rough sections using heading heuristics."""
    lines = [line.strip() for line in text.splitlines()]
    sections: dict[str, list[str]] = {}
    current_section = "full_text"
    sections[current_section] = []

    for line in lines:
        if not line:
            continue
        if is_heading(line):
            current_section = re.sub(r"\s+", " ", line.title())
            sections.setdefault(current_section, [])
            continue
        sections.setdefault(current_section, []).append(line)

    materialized = {
        section: normalize_text("\n".join(content))
        for section, content in sections.items()
        if normalize_text("\n".join(content))
    }
    if not materialized:
        return {"full_text": ""}
    return materialized


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from every page in a PDF file."""
    with pymupdf.open(pdf_path) as document:
        page_text = [page.get_text("text") for page in document]
    return normalize_text("\n".join(page_text))


def load_papers_metadata(papers_path: Path) -> dict[str, RawPaper]:
    """Load metadata JSONL into a map keyed by paper_id for fast lookup."""
    with papers_path.open("r", encoding="utf-8") as handle:
        papers = [RawPaper.model_validate_json(line) for line in handle if line.strip()]
    return {paper.paper_id: paper for paper in papers}


def parse_pdf(pdf_path: Path, paper: RawPaper) -> ParsedPaper:
    """Parse one PDF and attach extracted sections to its metadata record."""
    text = extract_pdf_text(pdf_path)
    sections = extract_sections_from_text(text)
    return ParsedPaper(paper=paper, sections=sections)


def parse_pdfs(
    pdf_dir: Path,
    papers_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Parse all PDFs in a directory and write ParsedPaper records to JSONL."""
    metadata = load_papers_metadata(papers_path)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_count = 0
    skipped_count = 0
    failed_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for pdf_path in pdf_paths:
            paper_id = pdf_path.stem
            paper = metadata.get(paper_id)
            if paper is None:
                skipped_count += 1
                logger.warning("Skipping %s: no matching metadata record", paper_id)
                continue

            try:
                parsed_paper = parse_pdf(pdf_path, paper)
            except Exception as exc:
                failed_count += 1
                logger.exception("Failed parsing %s: %s", pdf_path.name, exc)
                continue

            if not any(parsed_paper.sections.values()):
                failed_count += 1
                logger.warning("Skipping %s: extracted text was empty", pdf_path.name)
                continue

            handle.write(json.dumps(parsed_paper.model_dump(mode="json")) + "\n")
            parsed_count += 1

            if parsed_count % 100 == 0:
                logger.info("Parsed %s PDFs", parsed_count)

    return parsed_count, skipped_count, failed_count


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for local runs and batch parsing jobs."""
    parser = argparse.ArgumentParser(
        description="Parse arXiv PDFs into structured JSONL records."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=project_root / "data" / "raw" / "arxiv_pdfs",
        help="Directory containing downloaded PDF files.",
    )
    parser.add_argument(
        "--papers-path",
        type=Path,
        default=project_root / "data" / "processed" / "papers.jsonl",
        help="Path to the raw papers metadata JSONL.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "data" / "processed" / "parsed_papers.jsonl",
        help="Destination JSONL path for parsed papers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of PDFs to parse.",
    )
    return parser


def main() -> None:
    """CLI entrypoint: parse PDFs and log summary counters."""
    configure_logging()
    args = build_arg_parser().parse_args()
    logger.info("Parsing PDFs from %s", args.pdf_dir)
    parsed_count, skipped_count, failed_count = parse_pdfs(
        pdf_dir=args.pdf_dir,
        papers_path=args.papers_path,
        output_path=args.output_path,
        limit=args.limit,
    )
    logger.info(
        "Finished parsing PDFs. parsed=%s skipped=%s failed=%s output=%s",
        parsed_count,
        skipped_count,
        failed_count,
        args.output_path,
    )


if __name__ == "__main__":
    main()
