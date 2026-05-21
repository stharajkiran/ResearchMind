import re
import json
from pathlib import Path
from collections import Counter
from researchmind.utils.find_root import find_project_root
import logging

logger = logging.getLogger(__name__)

HEADING_CANONICAL_SET = [
    "abstract",
    "introduction",
    "result",
    "relatedwork",
    "conclusion",
    "reference",
    "background",
    "experiment",
    "method",
    "discussion",
    "futurework",
    "approach",
    "limitation",
    "ablationstudy"
]
SECTION_MAP = {
    "abstracts": "abstract",
    "introductions": "introduction",
    "backgrounds": "background",
    "related works": "related work",
    "methods": "method",
    "methodology": "method",
    "methodologies": "method",
    "approaches": "approach",
    "experiments": "experiment",
    "results": "result",
    "discussions": "discussion",
    "conclusions": "conclusion",
    "references": "reference",    
}

def clean_header(section: str) -> str:
    # remove numbers and punctuation, keep only words using re
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s+", "", section.strip())  # remove leading numbers and dots
    cleaned = cleaned.rstrip(":.")  # remove trailing colons and dots
    # remove _
    cleaned = re.sub(r"[_\-]+", "", cleaned)
    cleaned = cleaned.lower()  # lowercase for mapping
    # check any spaces and remove them for mapping
    cleaned = cleaned.replace(" ", "")
    return cleaned

def map_to_canonical(section: str) -> str:
    cleaned = clean_header(section)
    if cleaned in HEADING_CANONICAL_SET:
        return cleaned
    # check if cleaned is in part of HEADING_CANONICAL_SET
    for canonical in HEADING_CANONICAL_SET:
        if canonical in cleaned:
            return canonical

    for key, value in SECTION_MAP.items():
        if cleaned == key.replace(" ", ""):
            return value.replace(" ", "")
    return cleaned  # if no mapping found, return cleaned version

def clean_section(chunks_path: Path = None) -> list[dict]:
    # load chunks
    with open(chunks_path, "r") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    chunks_sections = [chunk["section"] for chunk in chunks]
    logger.info("Total chunks: %d", len(chunks))
    section_freq = Counter(chunks_sections)
    logger.info("Unique sections before cleaning: %d", len(section_freq))

    for chunk in chunks:
        chunk["section"] = map_to_canonical(chunk["section"])

    chunks = [chunk for chunk in chunks if chunk["section"] not in ["fulltext", "reference"]]

    chunks_sections = [chunk["section"] for chunk in chunks]
    logger.info("Total chunks after cleaning: %d", len(chunks))
    section_freq = Counter(chunks_sections)
    logger.info("Unique sections after cleaning: %d", len(section_freq))
    for section, freq in section_freq.items():
        logger.info("  %s: %d", section, freq)
    return chunks # updated chunks with cleaned sections, can be saved back to file if needed

def save_cleaned_chunks(chunks: list[dict], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

if __name__ == "__main__":
    from researchmind.utils.config import load_phase_config
    cfg = load_phase_config(find_project_root())
    chunks = clean_section(cfg.ingestion.raw_chunks_path)
    save_cleaned_chunks(chunks, cfg.index.chunks_path)