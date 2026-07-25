"""LangGraph tool functions for the ResearchMind literature assistant."""

import re

from researchmind.agent.tracing import trace

from researchmind.agent.state import AgentState
from researchmind.feedback.interfaces import FeedbackStore
from researchmind.ingestion.models import (
    ComparisonRAGResponse,
    RAGResponse,
    ResearchGapResponse,
)
from researchmind.retrieval.vector_store import VectorStore
from researchmind.guardrails.pipeline import ValidatorPipeline
import networkx as nx
from itertools import chain
from researchmind.ingestion.models import Chunk

from researchmind.session.cache import QueryCache
from researchmind.session.memory import SessionMemory
from researchmind.utils.llm_client import ResearchMindLLM
from researchmind.graph.citation_graph import get_neighbors
from researchmind.metrics import tool_calls_total
from researchmind.utils.build_prompt import (
    build_comparison_prompt,
    build_citation_prompt,
    build_gap_prompt,
    build_prompt,
)
from pydantic import BaseModel, Field
from .agent_utils import (
    CITATION_DIRECTION_PROMPT,
    COMPARE_METHODOLOGIES_PROMPT,
    classify_citation_direction,
)
import logging

logger = logging.getLogger("agent_tools")

_ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_QUOTED_TITLE_PATTERN = re.compile(r'["“](.+?)["”]')
_AUTHOR_HINT_PATTERN = re.compile(r"\bby\s+([^?.!]+)", re.IGNORECASE)


class SubjectList(BaseModel):
    subjects: list[str] = Field(
        ..., description="List of methods, models, or approaches being compared."
    )


def _retrieval_options(state: AgentState) -> dict:
    """Return retrieval settings supplied with the current agent request."""
    return {
        "mode": state.get("retrieval_mode", "standard"),
        "recency_decay_rate": state.get("recency_decay"),
    }


def _extract_arxiv_id(query: str) -> str | None:
    """Extract and normalize the first modern-format arXiv ID in a query.

    Args:
        query: Researcher-supplied citation question.

    Returns:
        The unversioned arXiv ID, or ``None`` when the query contains none.
    """
    match = _ARXIV_ID_PATTERN.search(query)
    return match.group(1) if match else None


def _extract_quoted_title(query: str) -> str | None:
    """Extract the first quoted paper title from a citation query."""
    match = _QUOTED_TITLE_PATTERN.search(query)
    return match.group(1).strip() if match else None


def _extract_author_hint(query: str) -> str | None:
    """Extract an optional author phrase following ``by`` in a citation query."""
    match = _AUTHOR_HINT_PATTERN.search(query)
    return match.group(1).strip() if match else None


def _citation_resolution_message(seed_id: str | None, title: str | None = None) -> str:
    """Return the safe user-facing response for an unresolved citation seed."""
    if title:
        return (
            f"I can't uniquely identify the titled paper {title!r} in the "
            "configured corpus graph. I will not substitute a text-search "
            "result for a graph relationship."
        )
    if seed_id is None:
        return (
            "Citation exploration currently requires an arXiv ID or an exact "
            "quoted title from the configured corpus graph. Provide an ID such "
            "as 2210.10773."
        )
    return (
        f"I can't identify arXiv ID {seed_id} in the configured citation graph. "
        "I will not substitute a text-search result for a graph relationship."
    )


@trace
def search_corpus(state: AgentState, retriever: VectorStore) -> dict:
    """Retrieve corpus evidence for a general research query.

    Args:
        state: Current agent state containing query and retrieval settings.
        retriever: Vector-store abstraction used to search the corpus.

    Returns:
        Retrieved chunks and updated tool-call history.
    """
    tool_calls_total.labels(tool_name="search_corpus").inc()
    query = state["query"]
    retrieved_chunks = retriever.search(query, k=10, **_retrieval_options(state))
    return {
        "retrieved_chunks": retrieved_chunks,
        "tool_call_history": state["tool_call_history"] + ["search_corpus"],
    }


@trace
def search_recent(
    state: AgentState, retriever: VectorStore, default_recency_decay_rate: float
) -> dict:
    """Retrieve recent evidence, applying a default decay only when needed.

    Args:
        state: Current agent state containing query and optional decay override.
        retriever: Vector-store abstraction used to search the corpus.
        default_recency_decay_rate: Fallback decay for recent-work requests.

    Returns:
        The top recent chunks and updated tool-call history.
    """
    tool_calls_total.labels(tool_name="search_recent").inc()
    query = state["query"]
    options = _retrieval_options(state)
    if options["recency_decay_rate"] is None:
        options["recency_decay_rate"] = default_recency_decay_rate
    retrieved_chunks = retriever.search(
        query, k=20, **options
    )[:10]
    return {
        "retrieved_chunks": retrieved_chunks,
        "tool_call_history": state["tool_call_history"] + ["search_recent"],
    }


@trace
def trace_citation_graph(
    state: AgentState,
    retriever: VectorStore,
    llm: ResearchMindLLM,
    graph: nx.DiGraph,
) -> dict:
    """Retrieve graph-bounded evidence for an explicit arXiv citation query.

    Args:
        state: Current agent state containing the citation query and retrieval settings.
        retriever: Corpus service used to load chunks for graph-verified papers.
        llm: Client used to classify citation direction.
        graph: Directed citation graph for neighbor traversal.

    Returns:
        Citation seed, direct-neighbor chunks, and updated tool-call history.
    """
    tool_calls_total.labels(tool_name="trace_citation_graph").inc()
    query = state["query"]
    title = _extract_quoted_title(query)
    seed_id = _extract_arxiv_id(query)
    if seed_id is None and title:
        seed_id = retriever.resolve_paper_id_by_title(
            title, author_hint=_extract_author_hint(query)
        )
    logger.debug(
        "trace_citation_graph: resolved_seed_id=%s, title=%r, graph_nodes=%d",
        seed_id,
        title,
        graph.number_of_nodes(),
    )
    if seed_id is None or not graph.has_node(seed_id):
        return {
            "retrieved_chunks": [],
            "citation_seed_id": None,
            "citation_neighbor_ids": [],
            "citation_resolution_error": _citation_resolution_message(seed_id, title),
            "tool_call_history": state["tool_call_history"] + ["trace_citation_graph"],
        }

    # direction of citation (inbound vs outbound)
    direction = classify_citation_direction(query, CITATION_DIRECTION_PROMPT, llm)
    # Direct citation questions require one-hop relationships only.
    if direction == "both":
        neighbors = get_neighbors(
            graph, seed_id, "inbound", depth=1, max_neighbors=25
        ) + get_neighbors(graph, seed_id, "outbound", depth=1, max_neighbors=25)
    else:
        neighbors = get_neighbors(graph, seed_id, direction, depth=1, max_neighbors=25)

    neighbors = list(dict.fromkeys(neighbors))[:25]
    seed_chunks = retriever.get_chunks_for_papers([seed_id], max_per_paper=2)
    neighbor_chunks = retriever.get_relevant_chunks_for_papers(
        neighbors, query, max_per_paper=2
    )
    logger.debug(
        "trace_citation_graph: %d neighbors → %d chunks",
        len(neighbors),
        len(neighbor_chunks),
    )

    return {
        "retrieved_chunks": seed_chunks + neighbor_chunks,
        "citation_seed_id": seed_id,
        "citation_neighbor_ids": neighbors,
        "citation_resolution_error": None,
        "tool_call_history": state["tool_call_history"] + ["trace_citation_graph"],
    }


@trace
def synthesise_answer(
    state: AgentState,
    llm: ResearchMindLLM,
    pipeline: ValidatorPipeline,
    store: FeedbackStore,
    session_memory: SessionMemory,
    query_cache: QueryCache,
) -> dict:
    tool_calls_total.labels(tool_name="synthesise_answer").inc()
    query = state["query"]
    citation_seed_id = state.get("citation_seed_id")
    citation_resolution_error = state.get("citation_resolution_error")
    if citation_resolution_error:
        return {
            "final_answer": RAGResponse(
                response=citation_resolution_error,
                sources=[],
                confidence=0.0,
                citations=[],
                contexts=[],
            ),
            "tool_call_history": state["tool_call_history"] + ["citation_refusal"],
            "validation_result": None,
            "feedback_id": None,
        }

    # check the redis cache
    if state.get("intent") != "citation" and (cached_answer := query_cache.get(query)):
        return {
            "final_answer": cached_answer,
            "tool_call_history": state["tool_call_history"] + ["synthesise_answer"],
            "validation_result": None,
            "feedback_id": None,
        }
    # Comparison RAG
    if state.get("compared_chunks"):
        compared_chunks = state["compared_chunks"]
        SYSTEM_PROMPT, content = build_comparison_prompt(query, compared_chunks)
        response_model = ComparisonRAGResponse
        max_tokens = 4096
    elif citation_seed_id:
        SYSTEM_PROMPT, content = build_citation_prompt(
            query=query,
            retrieved_chunks=state["retrieved_chunks"],
            seed_id=citation_seed_id,
            neighbor_ids=state.get("citation_neighbor_ids", []),
        )
        response_model = RAGResponse
        max_tokens = 2048
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
    if citation_seed_id and isinstance(response, RAGResponse):
        response = response.model_copy(
            update={"sources": state.get("citation_neighbor_ids", [])}
        )
    pipeline_result = pipeline.run(
        response=response, chunks=state.get("retrieved_chunks", [])
    )
    logger.info("Cited sources: %s", response.sources)
    if pipeline_result.blocked:
        logger.warning(
            "Response failed validation checks: "
            + "; ".join(
                f"{v.validator}: {'PASSED' if v.passed else 'FAILED'}"
                for v in pipeline_result.results
            )
        )
    if pipeline_result.redacted_text:
        if isinstance(response, RAGResponse):
            response = response.model_copy(
                update={"response": pipeline_result.redacted_text}
            )
        elif isinstance(response, ComparisonRAGResponse):
            response = response.model_copy(
                update={"comparison": pipeline_result.redacted_text}
            )

    hallucination_score = next(
        (
            r.score
            for r in pipeline_result.results
            if r.validator == "HallucinationScoreValidator"
        ),
        None,
    )
    citation_score = next(
        (
            r.score
            for r in pipeline_result.results
            if r.validator == "CitationGroundingValidator"
        ),
        None,
    )
    session_id = state.get("session_id", "unknown_session")
    feedback_id = store.save_feedback(
        session_id=session_id,
        query=query,
        intent=state.get("intent", ""),
        answer_json=response.model_dump(),
        hallucination_score=hallucination_score,
        citation_grounding_score=citation_score,
        validation_passed=pipeline_result.overall_passed,
        validator_results=[v.model_dump() for v in pipeline_result.results],
        retrieved_paper_ids=[c.paper_id for c in state.get("retrieved_chunks", [])],  # type: ignore
        retrieved_chunk_ids=[c.chunk_id for c in state.get("retrieved_chunks", [])],  # type: ignore
        rating=None,
    )
    # Citation answers remain uncached so graph-grounded evidence is always rebuilt.
    if state.get("intent") != "citation":
        query_cache.set(query, response.model_dump())
    # save relevant info to session memory for potential future use
    session_memory.save(
        session_id,
        state.get("retrieved_chunks", []),
        response,
    )

    return {
        "final_answer": response,
        "tool_call_history": state["tool_call_history"] + ["synthesise_answer"],
        "validation_result": pipeline_result,
        "feedback_id": feedback_id,
    }


@trace
def compare_methodologies(
    state: AgentState, retriever: VectorStore, llm: ResearchMindLLM
) -> dict:
    """Retrieve separate evidence sets for methods named in a comparison query.

    Args:
        state: Current agent state containing query and retrieval settings.
        retriever: Corpus search service.
        llm: Client used to extract the methods being compared.

    Returns:
        Per-method chunks, flattened retrieved chunks, and tool-call history.
    """
    tool_calls_total.labels(tool_name="compare_methodologies").inc()

    user_prompt = state["query"]
    subjects = llm.complete_structured(
        user_prompt=user_prompt,
        response_model=SubjectList,
        system_prompt=COMPARE_METHODOLOGIES_PROMPT,
        tier="fast",
        max_tokens=512,
        temperature=0.0,
    )
    if not subjects.subjects:
        raise ValueError("Subject extraction returned empty list — cannot compare.")
    subjects = subjects.subjects
    try:
        k = max(5, 20 // len(subjects))
    except ZeroDivisionError:
        k = 5
    # For each subject, retrieve relevant chunks and group them by subject for easier comparison in the final answer synthesis step
    compared_chunks: dict[str, list[Chunk]] = {}
    for subject in subjects:
        compared_chunks[subject] = retriever.search(
            subject, k=k, **_retrieval_options(state)
        )

    if not any(compared_chunks.values()):
        raise ValueError("No chunks retrieved for any subject.")
    return {
        "compared_chunks": compared_chunks,
        "retrieved_chunks": list(chain.from_iterable(compared_chunks.values())),
        "tool_call_history": state["tool_call_history"] + ["compare_methodologies"],
    }


@trace
def detect_research_gaps(
    state: AgentState,
    retriever: VectorStore,
    llm: ResearchMindLLM,
    pipeline: ValidatorPipeline,
    store: FeedbackStore,
) -> dict:
    """Generate a corpus-bounded limitations response from retrieved evidence.

    Args:
        state: Current agent state containing query and retrieval settings.
        retriever: Corpus search service.
        llm: Client used to generate the structured limitations response.
        pipeline: Validators applied to the generated response.
        store: Feedback persistence service.

    Returns:
        Retrieved evidence, structured response, validation result, and feedback ID.
    """
    tool_calls_total.labels(tool_name="detect_research_gaps").inc()
    # Placeholder for research gap detection logic
    query = state["query"]
    retrieved_chunks = retriever.search(query, k=20, **_retrieval_options(state))
    SYSTEM_PROMPT, user_prompt = build_gap_prompt(query, retrieved_chunks)

    response = llm.complete_structured(
        user_prompt=user_prompt,
        response_model=ResearchGapResponse,
        system_prompt=SYSTEM_PROMPT,
        tier="best",
        max_tokens=2048,
        temperature=0.0,
    )

    pipeline_result = pipeline.run(response=response, chunks=retrieved_chunks)
    if pipeline_result.blocked:
        logger.warning(
            "Response failed validation checks: "
            + "; ".join(
                f"{v.validator}: {'PASSED' if v.passed else 'FAILED'}"
                for v in pipeline_result.results
            )
        )
    hallucination_score = next(
        (
            r.score
            for r in pipeline_result.results
            if r.validator == "HallucinationScoreValidator"
        ),
        None,
    )
    citation_score = next(
        (
            r.score
            for r in pipeline_result.results
            if r.validator == "CitationGroundingValidator"
        ),
        None,
    )
    session_id = state.get("session_id", "unknown_session")
    feedback_id = store.save_feedback(
        session_id=session_id,
        query=query,
        intent=state.get("intent", ""),
        answer_json=response.model_dump(),
        hallucination_score=hallucination_score,
        citation_grounding_score=citation_score,
        validation_passed=pipeline_result.overall_passed,
        validator_results=[v.model_dump() for v in pipeline_result.results],
        retrieved_paper_ids=[c.paper_id for c in retrieved_chunks],
        retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
        rating=None,
    )
    return {
        "retrieved_chunks": retrieved_chunks,
        "final_answer": response,
        "tool_call_history": state["tool_call_history"] + ["detect_research_gaps"],
        "validation_result": pipeline_result,
        "feedback_id": feedback_id,
    }


@trace
def read_session_memory(state: AgentState, session_memory: SessionMemory) -> dict:
    tool_calls_total.labels(tool_name="read_session_memory").inc()

    # Redis wired in Phase 6 (Celery + Redis deferred)
    # get the session ID from state
    session_id = state.get("session_id", "unknown_session")
    # read any relevant information from session memory using the session ID as key
    data = session_memory.load(session_id)
    if data is None:
        return {}
    # return the data as chunks in the expected format to be added to state for use in subsequent steps
    return {"retrieved_chunks": [Chunk(**c) for c in data["chunks"]]}
