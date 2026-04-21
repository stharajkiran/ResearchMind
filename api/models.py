
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    k: int = 10

class SearchResult(BaseModel):
    paper_id: str
    title: str
    abstract: str


