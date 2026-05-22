import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

from researchmind.ingestion.models import ParsedPaper, RawPaper
from researchmind.ingestion.parsing.interfaces import PaperParser
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)

project_root = find_project_root()

HEADING_PATTERN = re.compile(
    r"^("
    r"abstract|introduction|background|motivation|overview|contributions?"
    r"|related work|literature review|prior work"
    r"|method(?:s)?|methodology|approach|framework|architecture|model|system"
    r"|problem (?:formulation|statement)|notation|preliminaries|setup"
    r"|experiment(?:s)?|experimental (?:setup|results?|evaluation)"
    r"|implementation(?: details?)?|training(?: details?)?|inference"
    r"|dataset(?:s)?|data(?: collection)?|evaluation|empirical evaluation"
    r"|results?|findings?|analysis|error analysis|ablation(?: study)?"
    r"|baseline(?:s)?|comparison(?:s)?"
    r"|discussion|future work|limitation(?:s)?|conclusion(?:s)?"
    r"|acknowledgm?ents?|reference(?:s)?|appendix|supplementary(?: material)?"
    r")$",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    """Sanitize raw PDF text: remove null bytes, normalize line endings,
    collapse whitespace runs, and trim leading/trailing space."""
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()



def _is_heading(line: str) -> bool:
    """Return True if the line looks like a section heading.

    Strategy:
    1. Strip leading section numbers (e.g. "3.1 Methods" → "Methods").
    2. Strip trailing colon.
    3. Match against HEADING_PATTERN — a vocabulary of known ML paper section
       names — and require at most 4 words (headings are short).
    """
    # Remove leading numbering like "3", "3.1.", "3." or Roman "IV." followed by whitespace
    candidate = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.?)\s+", "", line.strip())
    candidate = candidate.rstrip(":")
    if HEADING_PATTERN.match(candidate) and 1 <= len(candidate.split()) <= 4:
        return True
    return False


def extract_sections(text: str) -> dict[str, str]:
    """Split full paper text into named sections using heading heuristics.

    Walks every line of the extracted text. When a heading is detected the
    current section name is updated and subsequent lines accumulate under it.
    Everything before the first recognised heading lands in "full_text" (title,
    authors, affiliations, etc.).

    Returns a dict mapping section name → cleaned section text. If no content
    survives normalisation, returns {"full_text": ""}.
    """
    lines = [line.strip() for line in text.splitlines()]
    sections: dict[str, list[str]] = {}
    # Default bucket for content before the first detected heading
    current = "full_text"
    sections[current] = []

    for line in lines:
        if not line:
            continue
        if _is_heading(line):
            # Strip numbering and colon (same logic as _is_heading) so keys
            # like "3.1 Background" and "Background" normalise to the same key
            clean = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.?)\s+", "", line.strip()).rstrip(":")
            current = re.sub(r"\s+", " ", clean.title())
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    # Drop sections whose content is empty after normalisation
    materialized = {
        sec: _normalize_text("\n".join(content))
        for sec, content in sections.items()
        if _normalize_text("\n".join(content))
    }
    if not materialized:
        return {"full_text": ""}

    # Content before the first detected heading (stored as "full_text") is
    # almost always the abstract in a structured ML paper. Relabel it when
    # no explicit "Abstract" heading was found and at least one real section
    # exists — this avoids relabeling a full paper dump as "Abstract".
    real_sections = [k for k in materialized if k != "full_text"]
    if "full_text" in materialized and "Abstract" not in materialized and real_sections:
        materialized["Abstract"] = materialized.pop("full_text")

    logger.debug("Detected %d section(s): %s", len(materialized), list(materialized))
    return materialized


def extract_text(pdf_path: Path) -> str:
    """Extract clean plain text from a PDF, preserving visual reading order.

    Two-pass approach:
    1. Read each page using PyMuPDF's block-level API. Blocks are sorted by
       (y, x) position so two-column layouts are read top-to-bottom,
       left-to-right rather than in PDF internal order.
    2. Count how often each unique line appears across pages. Lines that
       appear on 20%+ of pages (minimum 2) are running headers/footers
       (e.g. "Proceedings of NeurIPS 2024", page numbers) and are filtered out.
    """
    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc)
        page_line_lists: list[list[str]] = []
        for page in doc:
            # "blocks" mode returns (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            # Sort top-to-bottom then left-to-right for correct column order
            blocks.sort(key=lambda b: (b[1], b[0]))
            lines: list[str] = []
            for b in blocks:
                if b[6] == 0:  # block_type 0 = text, 1 = image
                    lines.extend(b[4].splitlines())
            page_line_lists.append(lines)

    # Count per-page occurrences (set prevents double-counting within one page)
    line_freq: Counter[str] = Counter()
    for lines in page_line_lists:
        line_freq.update(set(lines))

    # Lines appearing on 20%+ of pages are treated as headers/footers
    threshold = max(2, int(n_pages * 0.2))
    noise = {line for line, count in line_freq.items() if count >= threshold}

    page_texts = []
    for lines in page_line_lists:
        filtered = [l for l in lines if l.strip() and l not in noise]
        page_texts.append("\n".join(filtered))

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
            if set(result.sections.keys()) == {"full_text"}:
                logger.warning("No sections detected in %s — fell back to full_text", pdf_path.name)
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
