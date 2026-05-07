from langsmith import traceable

from researchmind.agent.state import AgentState
from researchmind.feedback.store import FeedbackStore
from researchmind.ingestion.models import (
    ComparisonRAGResponse,
    RAGResponse,
    ResearchGapResponse,
)
from researchmind.retrieval.retriever import RetrieverService
from researchmind.guardrails.pipeline import ValidatorPipeline
import networkx as nx
from itertools import chain
from researchmind.ingestion.models import Chunk

from researchmind.session.cache import QueryCache
from researchmind.session.memory import SessionMemory
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
import logging

logger = logging.getLogger("agent_tools")


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
def synthesise_answer(
    state: AgentState,
    llm: ResearchMindLLM,
    pipeline: ValidatorPipeline,
    store: FeedbackStore,
    session_memory: SessionMemory,
    query_cache: QueryCache,
) -> dict:
    # Placeholder for answer synthesis logic
    query = state["query"]
    # check the redis cache
    if cached_answer := query_cache.get(query):
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
        retrieved_paper_ids=[
            c.paper_id for c in state.get("retrieved_chunks", [])
        ],  # type: ignore
        retrieved_chunk_ids=[c.chunk_id for c in state.get("retrieved_chunks", [])],  # type: ignore
        rating=None,
    )
    # save to redis cache
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
        compared_chunks[subject] = retriever.search(subject, k=k)

    if not any(compared_chunks.values()):
        raise ValueError("No chunks retrieved for any subject.")
    return {
        "compared_chunks": compared_chunks,
        "retrieved_chunks": list(chain.from_iterable(compared_chunks.values())),
        "tool_call_history": state["tool_call_history"] + ["compare_methodologies"],
    }


@traceable
def detect_research_gaps(
    state: AgentState,
    retriever: RetrieverService,
    llm: ResearchMindLLM,
    pipeline: ValidatorPipeline,
    store: FeedbackStore,
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


@traceable
def read_session_memory(state: AgentState, session_memory: SessionMemory) -> dict:
    # Redis wired in Phase 6 (Celery + Redis deferred)
    # get the session ID from state
    session_id = state.get("session_id", "unknown_session")
    # read any relevant information from session memory using the session ID as key
    data = session_memory.load(session_id)
    if data is None:
        return {}
    # return the data as chunks in the expected format to be added to state for use in subsequent steps
    return {"retrieved_chunks": [Chunk(**c) for c in data["chunks"]]}
