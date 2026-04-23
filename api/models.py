
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    k: int = 10

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

# Define your strict schema
class RAGResponse(BaseModel):
    response: str = Field(
        ..., 
        description="The detailed, technical answer synthesized from the provided research chunks. Adhere strictly to academic tone."
    )
    sources: list[str] = Field(
        ..., 
        description="A list of paper IDs that directly contributed to the answer. Only include papers that provided actual information."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, le=1.0, 
        description="A confidence score between 0.0 and 1.0 representing how well the context supports the answer."
    )
    citations: list[str] = Field(
        ..., 
        description="List of direct quotes from the provided chunks that support the response."
    )