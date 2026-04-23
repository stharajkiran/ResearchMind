from datetime import date
from pathlib import Path
from pydantic import BaseModel


class S2Author(BaseModel):
    author_id: str
    name: str
    
class RawPaper(BaseModel):
    paper_id: str                              # arXiv ID e.g. "2301.07041"
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]                      # ["cs.LG", "cs.AI"]
    published: date
    pdf_url: str
    latex_path: Path | None = None             # data/raw/arxiv/{paper_id}/
    s2_paper_id: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    s2_authors: list[S2Author] | None = None       # includes S2 author IDs

    class Config:
        frozen = True                          # immutable after creation

class ParsedPaper(BaseModel):
    paper:RawPaper
    sections: dict[str, str]                     # {"Introduction": "text", "Methods": "text", ...}

class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: str
    text: str
    page: int | None = None
    authors: list[str]
    year: int
    title: str
