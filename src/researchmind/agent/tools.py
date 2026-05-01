from langsmith import traceable

from researchmind.agent.state import AgentState
from researchmind.ingestion.models import (
    ComparisonRAGResponse,
    RAGResponse,
    ResearchGapResponse,
)
from researchmind.retrieval.retriever import RetrieverService
import networkx as nx
from itertools import chain
from researchmind.ingestion.models import Chunk

from researchmind.utils.llm_client import ResearchMindLLM
from researchmind.graph.citation_graph import get_neighbors
from researchmind.utils.build_prompt import (
    build_comparison_prompt,
    build_gap_prompt,
    build_prompt,
)
from pydantic import BaseModel, Field
from .agent_utils import (
    CITATION_DIRECTION_PROMPT,
    COMPARE_METHODOLOGIES_PROMPT,
    classify_citation_direction,
)


class SubjectList(BaseModel):
    subjects: list[str] = Field(
        ..., description="List of methods, models, or approaches being compared."
    )


@traceable
def search_corpus(state: AgentState, retriever: RetrieverService) -> dict:
    query = state["query"]
    retrieved_chunks = retriever.search(query, k=10)
    return {
        "retrieved_chunks": retrieved_chunks,
        "tool_call_history": state["tool_call_history"] + ["search_corpus"],
    }


@traceable
def search_recent(
    state: AgentState, retriever: RetrieverService, recency_decay_rate: float
) -> dict:
    query = state["query"]
    retrieved_chunks = retriever.search(
        query, k=20, recency_decay_rate=recency_decay_rate
    )[:10]
    return {
        "retrieved_chunks": retrieved_chunks,
        "tool_call_history": state["tool_call_history"] + ["search_recent"],
    }


@traceable
def trace_citation_graph(
    state: AgentState,
    retriever: RetrieverService,
    llm: ResearchMindLLM,
    graph: nx.DiGraph,
) -> dict:
    query = state["query"]
    seed_candidates = retriever.search(query, k=10)
    seed_id = next(
        (c.paper_id for c in seed_candidates if graph.has_node(c.paper_id)), None
    )
    if seed_id is None:
        return {
            "retrieved_chunks": [],
            "tool_call_history": state["tool_call_history"] + ["trace_citation_graph"],
        }

    # direction of citation (inbound vs outbound)
    direction = classify_citation_direction(query, CITATION_DIRECTION_PROMPT, llm)
    # Get all neighbors (both in and out) of the retrieved papers
    if direction == "both":
        neighbors = get_neighbors(graph, seed_id, "inbound", depth=2) + get_neighbors(
            graph, seed_id, "outbound", depth=2
        )
    else:
        neighbors = get_neighbors(graph, seed_id, direction, depth=2)

    neighbor_chunks = retriever.get_chunks_for_papers(list(neighbors))

    return {
        "retrieved_chunks": neighbor_chunks + seed_candidates,
        "tool_call_history": state["tool_call_history"] + ["trace_citation_graph"],
    }


@traceable
def synthesise_answer(state: AgentState, llm: ResearchMindLLM) -> dict:
    # Placeholder for answer synthesis logic
    query = state["query"]
    if state.get("compared_chunks"):
        compared_chunks = state["compared_chunks"]
        SYSTEM_PROMPT, content = build_comparison_prompt(query, compared_chunks)
        response_model = ComparisonRAGResponse
        max_tokens = 4096
    else:
        SYSTEM_PROMPT, content = build_prompt(query, state["retrieved_chunks"])
        response_model = RAGResponse
        max_tokens = 2048

    response = llm.complete_structured(
        user_prompt=content,
        response_model=response_model,
        system_prompt=SYSTEM_PROMPT,
        tier="best",
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return {
        "final_answer": response,
        "tool_call_history": state["tool_call_history"] + ["synthesise_answer"],
    }


@traceable
def compare_methodologies(
    state: AgentState, retriever: RetrieverService, llm: ResearchMindLLM
) -> dict:
    user_prompt = state["query"]
    subjects = llm.complete_structured(
        user_prompt=user_prompt,
        response_model=SubjectList,
        system_prompt=COMPARE_METHODOLOGIES_PROMPT,
        tier="fast",
        max_tokens=512,
        temperature=0.0,
    )
    subjects = subjects.subjects
    try:
        k = max(5, 20 // len(subjects))
    except ZeroDivisionError:
        k = 5
    # For each subject, retrieve relevant chunks and group them by subject for easier comparison in the final answer synthesis step
    compared_chunks: dict[str, list[Chunk]] = {}
    for subject in subjects:
        compared_chunks[subject] = retriever.search(subject, k=k)

    return {
        "compared_chunks": compared_chunks,
        "retrieved_chunks": list(chain.from_iterable(compared_chunks.values())),
        "tool_call_history": state["tool_call_history"] + ["compare_methodologies"],
    }


@traceable
def detect_research_gaps(
    state: AgentState, retriever: RetrieverService, llm: ResearchMindLLM
) -> dict:
    # Placeholder for research gap detection logic
    query = state["query"]
    retrieved_chunks = retriever.search(query, k=20)
    SYSTEM_PROMPT, user_prompt = build_gap_prompt(query, retrieved_chunks)

    response = llm.complete_structured(
        user_prompt=user_prompt,
        response_model=ResearchGapResponse,
        system_prompt=SYSTEM_PROMPT,
        tier="best",
        max_tokens=2048,
        temperature=0.0,
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "final_answer": response,
        "tool_call_history": state["tool_call_history"] + ["detect_research_gaps"],
    }


@traceable
def read_session_memory(state: AgentState) -> dict:
    # Redis wired in Phase 6 (Celery + Redis deferred)
    return {}
