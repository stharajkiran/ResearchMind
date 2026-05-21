import hashlib

from researchmind.ingestion.chunking.interfaces import Chunker
from researchmind.ingestion.models import Chunk, ParsedPaper


def _chunk_id(paper_id: str, section: str, index: int) -> str:
    raw = f"{paper_id}::{section}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


class SectionChunker(Chunker):
    """Splits a ParsedPaper into one Chunk per section.

    For sections that exceed max_words, the text is split into overlapping
    windows so no chunk is too long for the embedding model.
    """

    def __init__(self, max_words: int = 400, overlap_words: int = 50) -> None:
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, paper: ParsedPaper) -> list[Chunk]:
        raw = paper.paper
        year = raw.published.year
        chunks: list[Chunk] = []

        for section, text in paper.sections.items():
            if not text.strip():
                continue
            words = text.split()
            if len(words) <= self.max_words:
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(raw.paper_id, section, 0),
                        paper_id=raw.paper_id,
                        section=section,
                        text=text,
                        authors=raw.authors,
                        year=year,
                        title=raw.title,
                    )
                )
            else:
                # Sliding window over long sections
                step = self.max_words - self.overlap_words
                for i, start in enumerate(range(0, len(words), step)):
                    window = words[start : start + self.max_words]
                    if not window:
                        break
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(raw.paper_id, section, i),
                            paper_id=raw.paper_id,
                            section=section,
                            text=" ".join(window),
                            authors=raw.authors,
                            year=year,
                            title=raw.title,
                        )
                    )

        return chunks
