import argparse
import json
import logging
import re
from pathlib import Path

import pymupdf

from researchmind.ingestion.models import ParsedPaper, RawPaper
from researchmind.ingestion.parsing.interfaces import PaperParser
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)

project_root = find_project_root()

HEADING_PATTERN = re.compile(
    r"^(abstract|introduction|background|related work|method(?:s)?|approach|experiment(?:s)?|results|Ablation Study|Discussion|Future Work|Limitation(?:s)?|Conclusion(?:s)?|References(?:s)?)$",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_valid_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 3:
        return False
    if not re.search(r"[a-zA-Z]", line):
        return False
    artifacts = [r"←", r"→", r"↓", r"↑", r"•", r"\|"]
    if any(re.search(a, line) for a in artifacts):
        return False
    return True


def _is_heading(line: str) -> bool:
    candidate = re.sub(r"^\d+(?:\.\d+)*\s+", "", line.strip())
    candidate = candidate.rstrip(":")
    if HEADING_PATTERN.match(candidate) and 1 <= len(candidate.split()) <= 4:
        return True
    if not _is_valid_line(candidate):
        return False
    return False


def extract_sections(text: str) -> dict[str, str]:
    """Split full paper text into sections using heading heuristics."""
    lines = [line.strip() for line in text.splitlines()]
    sections: dict[str, list[str]] = {}
    current = "full_text"
    sections[current] = []

    for line in lines:
        if not line:
            continue
        if _is_heading(line):
            current = re.sub(r"\s+", " ", line.title())
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    materialized = {
        sec: _normalize_text("\n".join(content))
        for sec, content in sections.items()
        if _normalize_text("\n".join(content))
    }
    return materialized if materialized else {"full_text": ""}


def extract_text(pdf_path: Path) -> str:
    """Extract plain text from every page of a PDF."""
    with pymupdf.open(pdf_path) as doc:
        page_texts = [page.get_text("text") for page in doc]
    return _normalize_text("\n".join(page_texts))


class PyMuPDFParser(PaperParser):
    """PaperParser backed by PyMuPDF with heuristic section splitting."""

    def parse(self, pdf_path: Path, paper: RawPaper) -> ParsedPaper:
        text = extract_text(pdf_path)
        sections = extract_sections(text)
        return ParsedPaper(paper=paper, sections=sections)


def parse_pdfs(
    pdf_dir: Path,
    papers_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Parse all PDFs in a directory and write ParsedPaper records to JSONL."""
    with papers_path.open("r", encoding="utf-8") as f:
        metadata = {p.paper_id: p for line in f if line.strip()
                    for p in [RawPaper.model_validate_json(line)]}

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser = PyMuPDFParser()
    parsed, skipped, failed = 0, 0, 0
    seen: set[str] = set()

    with output_path.open("w", encoding="utf-8") as out:
        for pdf_path in pdf_paths:
            paper_id = pdf_path.stem
            if paper_id in seen:
                skipped += 1
                continue
            paper = metadata.get(paper_id)
            if paper is None:
                skipped += 1
                logger.warning("No metadata for %s — skipping", paper_id)
                continue
            try:
                result = parser.parse(pdf_path, paper)
                seen.add(paper_id)
            except Exception:
                failed += 1
                logger.exception("Failed to parse %s", pdf_path.name)
                continue
            if not any(result.sections.values()):
                failed += 1
                logger.warning("Empty text extracted from %s — skipping", pdf_path.name)
                continue
            out.write(json.dumps(result.model_dump(mode="json")) + "\n")
            parsed += 1
            if parsed % 100 == 0:
                logger.info("Parsed %d PDFs so far", parsed)

    return parsed, skipped, failed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse arXiv PDFs into structured JSONL. "
                    "Paths default to the active phase config (CONFIG_NAME env var)."
    )
    parser.add_argument("--pdf-dir", type=Path, default=None,
                        help="Directory of downloaded PDFs (overrides config)")
    parser.add_argument("--papers-path", type=Path, default=None,
                        help="Raw papers JSONL with metadata (overrides config)")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Destination parsed-papers JSONL (overrides config)")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_arg_parser().parse_args()

    from researchmind.utils.config import load_phase_config
    cfg = load_phase_config(project_root)

    pdf_dir = args.pdf_dir or cfg.ingestion.pdf_dir
    papers_path = args.papers_path or cfg.ingestion.papers_path
    output_path = args.output_path or cfg.ingestion.parsed_papers_path

    logger.info("Parsing PDFs | phase=%s pdf_dir=%s", cfg.name, pdf_dir)
    parsed, skipped, failed = parse_pdfs(
        pdf_dir=pdf_dir,
        papers_path=papers_path,
        output_path=output_path,
        limit=args.limit,
    )
    logger.info("Done. parsed=%d skipped=%d failed=%d", parsed, skipped, failed)


if __name__ == "__main__":
    main()
