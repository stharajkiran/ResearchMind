"""Tests for forwarding agent request options into routed retrieval tools."""

from types import SimpleNamespace

from api.app import agent
from api.models import RAGRequest
from researchmind.agent.tools import search_corpus, search_recent


class RecordingAgent:
    """Record the state passed to an agent invocation for assertions."""

    def __init__(self) -> None:
        self.states: list[dict] = []

    def invoke(self, state: dict) -> dict:
        """Record state and return the minimum API-compatible agent result."""
        self.states.append(state)
        return {"final_answer": {"response": "ok"}, "feedback_id": None}


class RecordingRetriever:
    """Capture retrieval calls without loading an index or external provider."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "standard",
        recency_decay_rate: float | None = None,
    ) -> list:
        """Record a retrieval request and return an empty fixture result."""
        self.calls.append(
            {
                "query": query,
                "k": k,
                "mode": mode,
                "recency_decay_rate": recency_decay_rate,
            }
        )
        return []


def _state(**overrides: object) -> dict:
    """Build the minimum agent state needed by retrieval tools in these tests."""
    state = {
        "query": "test query",
        "intent": "search",
        "retrieval_mode": "standard",
        "recency_decay": None,
        "retrieved_chunks": [],
        "compared_chunks": None,
        "citation_seed_id": None,
        "citation_neighbor_ids": [],
        "citation_resolution_error": None,
        "tool_call_history": [],
        "session_id": "session-a",
        "final_answer": None,
        "validation_result": None,
        "feedback_id": None,
    }
    state.update(overrides)
    return state


def test_agent_forwards_distinct_sessions_and_retrieval_options() -> None:
    """The API must preserve each caller's session and retrieval options."""
    recording_agent = RecordingAgent()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent=recording_agent)))

    agent(
        RAGRequest(
            query="first query",
            session_id="session-one",
            retrieval_mode="rewrite",
            recency_decay=0.25,
        ),
        request,
    )
    agent(
        RAGRequest(
            query="second query",
            session_id="session-two",
            retrieval_mode="hyde",
        ),
        request,
    )

    first_state, second_state = recording_agent.states
    assert first_state["session_id"] == "session-one"
    assert first_state["retrieval_mode"] == "rewrite"
    assert first_state["recency_decay"] == 0.25
    assert second_state["session_id"] == "session-two"
    assert second_state["retrieval_mode"] == "hyde"
    assert second_state["recency_decay"] is None


def test_search_corpus_uses_request_retrieval_options() -> None:
    """The standard search route must honor the selected retrieval settings."""
    retriever = RecordingRetriever()

    search_corpus(
        _state(retrieval_mode="rewrite", recency_decay=0.2),  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
    )

    assert retriever.calls == [
        {
            "query": "test query",
            "k": 10,
            "mode": "rewrite",
            "recency_decay_rate": 0.2,
        }
    ]


def test_search_recent_uses_default_decay_only_when_request_omits_it() -> None:
    """Recent-search keeps its default decay while allowing an explicit override."""
    retriever = RecordingRetriever()

    search_recent(
        _state(retrieval_mode="standard"),  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
        default_recency_decay_rate=0.1,
    )
    search_recent(
        _state(retrieval_mode="rewrite", recency_decay=0.3),  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
        default_recency_decay_rate=0.1,
    )

    assert retriever.calls[0]["mode"] == "standard"
    assert retriever.calls[0]["recency_decay_rate"] == 0.1
    assert retriever.calls[1]["mode"] == "rewrite"
    assert retriever.calls[1]["recency_decay_rate"] == 0.3
