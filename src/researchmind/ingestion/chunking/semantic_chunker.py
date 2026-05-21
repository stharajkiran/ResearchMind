import hashlib
import re

from researchmind.ingestion.chunking.interfaces import Chunker
from researchmind.ingestion.models import Chunk, ParsedPaper


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _chunk_id(paper_id: str, section: str, index: int) -> str:
    raw = f"{paper_id}::semantic::{section}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


class SemanticChunker(Chunker):
    """Splits sections on sentence boundaries, accumulating until chunk_size words.

    Produces more coherent chunks than fixed-window splitting since chunks
    never cut mid-sentence. Useful for methodology sections where a single
    idea spans several sentences.
    """

    def __init__(self, chunk_size: int = 150, overlap: int = 20) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, paper: ParsedPaper) -> list[Chunk]:
        raw = paper.paper
        chunks: list[Chunk] = []

        for section, text in paper.sections.items():
            sentences = _split_sentences(text)
            word_count = 0
            current: list[str] = []

            for sentence in sentences:
                sw = len(sentence.split())
                if word_count + sw > self.chunk_size and current:
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(raw.paper_id, section, len(chunks)),
                            paper_id=raw.paper_id,
                            section=section,
                            text=" ".join(current),
                            authors=raw.authors,
                            year=raw.published.year,
                            title=raw.title,
                        )
                    )
                    current = [sentence]
                    word_count = sw
                else:
                    current.append(sentence)
                    word_count += sw

            if current:
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(raw.paper_id, section, len(chunks)),
                        paper_id=raw.paper_id,
                        section=section,
                        text=" ".join(current),
                        authors=raw.authors,
                        year=raw.published.year,
                        title=raw.title,
                    )
                )

        return chunks
