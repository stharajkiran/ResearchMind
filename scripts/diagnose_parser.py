"""Parser diagnostic script.

Reads a parsed-papers JSONL and reports section detection quality:
  - Fallback rate (papers with only full_text)
  - Section count distribution
  - Most common section names
  - Words-per-section stats

Usage:
    uv run python scripts/diagnose_parser.py
    uv run python scripts/diagnose_parser.py --parsed-path data/processed/full/arxiv_ss2_parsed_papers.jsonl
    uv run python scripts/diagnose_parser.py --limit 100
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from researchmind.utils.config import load_phase_config
from researchmind.utils.find_root import find_project_root


def diagnose(parsed_path: Path, limit: int | None = None) -> None:
    if not parsed_path.exists():
        print(f"File not found: {parsed_path}")
        return

    papers: list[dict] = []
    with parsed_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                papers.append(json.loads(line))

    if not papers:
        print("No papers found.")
        return

    total = len(papers)
    fallback_count = 0
    real_section_counts: list[int] = []
    section_name_freq: Counter[str] = Counter()
    words_per_section: list[int] = []

    for p in papers:
        sections: dict[str, str] = p.get("sections", {})
        names = list(sections.keys())

        is_fallback = set(names) <= {"full_text"}
        if is_fallback:
            fallback_count += 1

        real = [n for n in names if n.lower() != "full_text"]
        real_section_counts.append(len(real))
        section_name_freq.update(real)

        for text in sections.values():
            w = len(text.split())
            if w > 0:
                words_per_section.append(w)

    avg_sections = sum(real_section_counts) / total
    fallback_pct = fallback_count / total * 100

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Parser Diagnostic Report")
    print(f"{'─'*50}")
    print(f"  Papers analysed       : {total}")
    print(f"  Fallback (full_text)  : {fallback_count}  ({fallback_pct:.1f}%)")
    print(f"  Avg real sections     : {avg_sections:.1f}")
    print(f"  Min / Max sections    : {min(real_section_counts)} / {max(real_section_counts)}")

    # ── Section count distribution ────────────────────────────────────────────
    dist: Counter[int] = Counter(real_section_counts)
    print(f"\n  Section count distribution:")
    for count in sorted(dist):
        label = "0 (full_text only)" if count == 0 else f"{count} section(s)"
        bar = "█" * max(1, dist[count] * 30 // total)
        print(f"    {label:22s} {bar:30s} {dist[count]:4d} papers")

    # ── Top section names ─────────────────────────────────────────────────────
    print(f"\n  Top 20 detected section names:")
    for name, freq in section_name_freq.most_common(20):
        pct = freq / total * 100
        print(f"    {freq:4d} ({pct:4.1f}%)  {name}")

    # ── Words-per-section stats ───────────────────────────────────────────────
    if words_per_section:
        mean_w = sum(words_per_section) / len(words_per_section)
        print(f"\n  Words per section — mean: {mean_w:.0f}  "
              f"min: {min(words_per_section)}  max: {max(words_per_section)}")

    print(f"{'─'*50}\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parsed-path", type=Path, default=None,
                        help="Parsed papers JSONL (overrides config)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only analyse the first N papers")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    root = find_project_root()
    cfg = load_phase_config(root)

    parsed_path = args.parsed_path or cfg.ingestion.parsed_papers_path
    print(f"Reading: {parsed_path}")
    if args.limit:
        print(f"Limit: {args.limit} papers")

    diagnose(parsed_path, limit=args.limit)


if __name__ == "__main__":
    main()
