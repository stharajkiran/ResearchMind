from abc import ABC, abstractmethod
from pathlib import Path

import json
from researchmind.ingestion.models import Chunk, ParsedPaper
import re
import logging

logger = logging.getLogger(__name__)

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, paper: ParsedPaper) -> list[Chunk]:
        ...

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 200, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, paper: ParsedPaper) -> list[Chunk]:
        # get sections as dict of {section_name: text}
        sections = paper.sections
        chunks = []
        for section_name, text in sections.items():
            start = 0
            text = text.replace("\n", " ")
            words = text.split(" ")
            while start < len(words):
                end = start + self.chunk_size
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                chunk_id = f"{paper.paper.paper_id}_{section_name}_{start}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        paper_id=paper.paper.paper_id,
                        section=section_name,
                        text=chunk_text,
                        page=None,
                        authors=paper.paper.authors,
                        year=paper.paper.published.year,
                        title=paper.paper.title,
                    )
                )
                start += self.chunk_size - self.overlap
        return chunks
       
def _split_sentences(text: str) -> list[str]:
    # split on ". ", "? ", "! " — simple, no nltk needed
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

class SemanticChunker(BaseChunker):
    def __init__(self, chunk_size: int = 150, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, paper: ParsedPaper) -> list[Chunk]:
        sections = paper.sections
        chunks = []
        for section_name, text in sections.items():
            sentences = _split_sentences(text)
            chunk_word_count = 0
            chunk_text = ""
            for sentence in sentences:
                sentence_word_count = len(sentence.split(" "))
                if chunk_word_count + sentence_word_count > self.chunk_size:
                    # emit chunk
                    chunk_id = f"{paper.paper.paper_id}_{section_name}_{len(chunks)}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            paper_id=paper.paper.paper_id,
                            section=section_name,
                            text=chunk_text,
                            page=None,
                            authors=paper.paper.authors,
                            year=paper.paper.published.year,
                            title=paper.paper.title,
                        )
                    )
                    # remove emitted words and apply overlap
                    chunk_word_count = sentence_word_count
                    chunk_text = sentence
                    continue
                # track of words in current chunk 
                chunk_word_count += sentence_word_count
                # add sentence to current chunk text
                chunk_text += " " + sentence

            # emit any remaining text as final chunk
            if chunk_text.strip():
                chunk_id = f"{paper.paper.paper_id}_{section_name}_{len(chunks)}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        paper_id=paper.paper.paper_id,
                        section=section_name,
                        text=chunk_text,
                        page=None,
                        authors=paper.paper.authors,
                        year=paper.paper.published.year,
                        title=paper.paper.title,
                    )
                )

        return chunks
    
def chunk_papers(chunker: BaseChunker, parsed_papers_path: Path, output_path: Path) -> tuple[int, int]:
    """Read ParsedPapers from JSONL, chunk them, and write Chunks to new JSONL.
    
    Returns (chunks written, papers failed) for logging."""
    
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Read parsed papers from JSONL file
    with open(parsed_papers_path, "r") as f:
        parsed_papers = [json.loads(line) for line in f]   

    # paper is dictionary with keys "paper" (RawPaper fields) and "sections"
    failed_papers = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for n, paper in enumerate(parsed_papers):
            try:
                # validate and convert dict to ParsedPaper model
                parsed_paper = ParsedPaper.model_validate(paper)
            except Exception as e:
                logger.error(f"Failed to validate {paper['paper']['paper_id']}: {e}")
                failed_papers += 1
                continue
            # list of Chunk objects
            chunks = chunker.chunk(parsed_paper)
            # saves chunks to output_path as JSONL, one chunk per line
            
            for chunk in chunks:
                handle.write(json.dumps(chunk.model_dump(mode="json")) + "\n")
            
            if n % 100 == 0:
                logger.info(f"success: {n - failed_papers} chunks written, {failed_papers} papers failed so far")

    return len(parsed_papers) - failed_papers, failed_papers

def main():
    parsed_papers_path = Path("data/processed/parsed_papers.jsonl")
    output_path = Path("data/processed/chunks.jsonl")
    chunker = FixedSizeChunker(chunk_size=200, overlap=20)
    chunks_written, papers_failed = chunk_papers(chunker, parsed_papers_path, output_path)
    logger.info(f"Chunking complete: {chunks_written} chunks written, {papers_failed} papers failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()