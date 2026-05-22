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
    "limitation",
    "ablationstudy",
    "dataset",
]
SECTION_MAP = {
    # abstract / intro / background
    "abstracts": "abstract",
    "introductions": "introduction",
    "overview": "introduction",
    "motivation": "introduction",
    "contributions": "introduction",
    "contribution": "introduction",
    "backgrounds": "background",
    "preliminaries": "background",
    "notation": "background",
    # related work
    "related works": "related work",
    "literature review": "related work",
    "prior work": "related work",
    # method — approach, model, and architecture all map here
    "methods": "method",
    "methodology": "method",
    "methodologies": "method",
    "approach": "method",
    "approaches": "method",
    "model": "method",
    "models": "method",
    "implementation details": "method",
    "implementation detail": "method",
    "implementation": "method",
    "architecture": "method",
    "framework": "method",
    "system": "method",
    "inference": "method",
    "problem formulation": "method",
    "problem statement": "introduction",
    # experiment
    "experiments": "experiment",
    "experimental setup": "experiment",
    "setup": "experiment",
    "training": "experiment",
    "training details": "experiment",
    "evaluation": "experiment",
    "baseline": "experiment",
    "baselines": "experiment",
    "comparison": "experiment",
    "comparisons": "experiment",
    # result
    "results": "result",
    "analysis": "result",
    "findings": "result",
    "finding": "result",
    "error analysis": "result",
    # conclusion — future work folds in here
    "discussions": "discussion",
    "conclusions": "conclusion",
    "future work": "conclusion",
    "futurework": "conclusion",
    # dataset
    "datasets": "dataset",
    "data": "dataset",
    "data collection": "dataset",
    # misc
    "references": "reference",
    "ablation": "ablationstudy",
}

_FILTERED_SECTIONS = {
    "fulltext", "reference",
    "acknowledgments", "acknowledgements", "acknowledgment",
    "appendix", "supplementarymaterial", "supplementary",
}
_MIN_WORDS = 20

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
    # Substring match — longest canonical wins to avoid "result" beating
    # "experiment" in "experimentalresults"
    matches = [c for c in HEADING_CANONICAL_SET if c in cleaned]
    if matches:
        return max(matches, key=len)
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

    chunks = [
        chunk for chunk in chunks
        if chunk["section"] not in _FILTERED_SECTIONS
        and len(chunk["text"].split()) >= _MIN_WORDS
    ]

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