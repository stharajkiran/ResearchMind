from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    retrieval_mode: str  = "standard"
    recency_decay: float | None = None


class SearchResult(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str
    text: str
    year: int


class RAGRequest(BaseModel):
    query: str
    session_id: str | None = None  # for Redis multi-turn context lookup
    retrieval_mode: str  = "standard"
    recency_decay: float | None = None


