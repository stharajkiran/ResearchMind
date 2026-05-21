import hashlib

from researchmind.ingestion.chunking.interfaces import Chunker
from researchmind.ingestion.models import Chunk, ParsedPaper


def _chunk_id(paper_id: str, index: int) -> str:
    raw = f"{paper_id}::fixed::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


class FixedSizeChunker(Chunker):
    """Splits full paper text into fixed-size word windows, ignoring section boundaries.

    Used for the chunking A/B experiment (fixed 400w vs 512w vs section-aware).
    """

    def __init__(self, chunk_words: int = 400, overlap_words: int = 50) -> None:
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words

    def chunk(self, paper: ParsedPaper) -> list[Chunk]:
        raw = paper.paper
        full_text = " ".join(paper.sections.values())
        words = full_text.split()
        step = self.chunk_words - self.overlap_words
        chunks: list[Chunk] = []

        for i, start in enumerate(range(0, len(words), step)):
            window = words[start : start + self.chunk_words]
            if not window:
                break
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(raw.paper_id, i),
                    paper_id=raw.paper_id,
                    section="full_text",
                    text=" ".join(window),
                    authors=raw.authors,
                    year=raw.published.year,
                    title=raw.title,
                )
            )

        return chunks
