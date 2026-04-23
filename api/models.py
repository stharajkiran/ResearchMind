
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    k: int = 10

class SearchResult(BaseModel):
    paper_id: str
    title: str
    abstract: str

class RAGRequest(BaseModel):
    query: str
    session_id: str | None = None  # for Redis multi-turn context lookup

class RAGResponse(BaseModel):
    response: str
    sources: list[str]
    confidence: float
    citations: list[str]


