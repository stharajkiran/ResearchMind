from typing_extensions import TypedDict
from researchmind.ingestion.models import Chunk, RAGResponse, ComparisonRAGResponse, ResearchGapResponse


class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: list[Chunk]
    compared_chunks: dict[str, list[Chunk]] | None
    tool_call_history: list[str]
    session_id: str
    final_answer: RAGResponse  | ComparisonRAGResponse | ResearchGapResponse | None
