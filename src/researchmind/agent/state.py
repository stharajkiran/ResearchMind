"""Typed state passed between nodes in the ResearchMind LangGraph workflow."""

from typing_extensions import TypedDict
from researchmind.ingestion.models import (
    Chunk,
    RAGResponse,
    ComparisonRAGResponse,
    ResearchGapResponse,
)
from researchmind.guardrails.validators import PipeLineResult


class AgentState(TypedDict):
    """Represent request data and intermediate results for one agent invocation."""

    query: str
    intent: str
    retrieval_mode: str
    recency_decay: float | None
    retrieved_chunks: list[Chunk]
    compared_chunks: dict[str, list[Chunk]] | None
    citation_seed_id: str | None
    citation_neighbor_ids: list[str]
    citation_resolution_error: str | None
    tool_call_history: list[str]
    session_id: str
    final_answer: RAGResponse | ComparisonRAGResponse | ResearchGapResponse | None
    validation_result: PipeLineResult | None
    feedback_id: int | None
