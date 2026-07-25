"""Offline correctness tests for explicit-ID citation traversal."""

import networkx as nx

from researchmind.agent.tools import synthesise_answer, trace_citation_graph
from researchmind.guardrails.validators import PipeLineResult
from researchmind.ingestion.models import Chunk, RAGResponse
from researchmind.retrieval.retriever import RetrieverService


class RecordingCitationRetriever:
    """Provide exact paper chunks while rejecting general-search seeding."""

    def __init__(self) -> None:
        self.search_called = False
        self.seed_requests: list[list[str]] = []
        self.neighbor_requests: list[list[str]] = []
        self.title_requests: list[tuple[str, str | None]] = []

    def search(self, *args: object, **kwargs: object) -> list[Chunk]:
        """Fail when citation traversal attempts general text-search seeding."""
        self.search_called = True
        raise AssertionError("Citation traversal must not use general text search.")

    def get_chunks_for_papers(
        self, paper_ids: list[str], max_per_paper: int | None = None
    ) -> list[Chunk]:
        """Return one deterministic chunk for each requested seed paper."""
        self.seed_requests.append(paper_ids)
        return [self._chunk(paper_id) for paper_id in paper_ids]

    def get_relevant_chunks_for_papers(
        self,
        paper_ids: list[str],
        query: str,
        max_per_paper: int = 2,
    ) -> list[Chunk]:
        """Return one deterministic chunk for each graph-neighbor paper."""
        self.neighbor_requests.append(paper_ids)
        return [self._chunk(paper_id) for paper_id in paper_ids]

    def resolve_paper_id_by_title(
        self, title: str, author_hint: str | None = None
    ) -> str | None:
        """Resolve only the known title used by the deterministic test fixture."""
        self.title_requests.append((title, author_hint))
        if (
            title == "Anomaly Detection Requires Better Representations"
            and author_hint == "Tal Reiss"
        ):
            return "2210.10773"
        return None

    @staticmethod
    def _chunk(paper_id: str) -> Chunk:
        """Build a minimal valid chunk for an exact paper ID."""
        return Chunk(
            chunk_id=f"chunk-{paper_id}",
            paper_id=paper_id,
            section="abstract",
            text=f"Evidence for {paper_id}.",
            authors=[],
            year=2024,
            title=f"Paper {paper_id}",
        )


class InboundDirectionLLM:
    """Classify every test citation query as an inbound relationship request."""

    def complete(self, **_: object) -> str:
        """Return the direct inbound direction without a provider call."""
        return "inbound"


class FailingLLM:
    """Fail if an unresolved citation request attempts LLM synthesis."""

    def complete_structured(self, **_: object) -> object:
        """Raise because safe citation refusals must not call an LLM."""
        raise AssertionError("Unresolved citation queries must not call an LLM.")


class CitationResponderLLM:
    """Return an intentionally contaminated answer to test source normalization."""

    def complete_structured(self, **_: object) -> RAGResponse:
        """Return a response whose source list contains a non-neighbor paper."""
        return RAGResponse(
            response="The graph-bounded answer is summarized here.",
            sources=["2503.01184"],
            confidence=0.5,
            citations=[],
        )


class PassingPipeline:
    """Return a passing validation result without model or provider work."""

    def run(self, **_: object) -> PipeLineResult:
        """Return an empty passing result for synthesis-path testing."""
        return PipeLineResult(results=[], overall_passed=True, blocked=False)


class RecordingStore:
    """Record feedback persistence calls from synthesis tests."""

    def save_feedback(self, **_: object) -> None:
        """Accept a feedback record without external persistence."""
        return None


class RecordingMemory:
    """Accept synthesis session writes without external persistence."""

    def save(self, *_: object, **__: object) -> None:
        """Accept a session-memory write for this focused test."""
        return None


def _state(query: str) -> dict:
    """Build the minimum citation state required by tool and synthesis tests."""
    return {
        "query": query,
        "intent": "citation",
        "retrieval_mode": "standard",
        "recency_decay": None,
        "retrieved_chunks": [],
        "compared_chunks": None,
        "citation_seed_id": None,
        "citation_neighbor_ids": [],
        "citation_resolution_error": None,
        "tool_call_history": [],
        "session_id": "citation-test",
        "final_answer": None,
        "validation_result": None,
        "feedback_id": None,
    }


def _citation_graph() -> nx.DiGraph:
    """Build a graph with direct and second-hop inbound relationships."""
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("2301.11514", "2210.10773"),
            ("2307.11085", "2210.10773"),
            ("2311.14773", "2210.10773"),
            ("second-hop", "2301.11514"),
            ("2503.01184", "unrelated-paper"),
        ]
    )
    return graph


def test_explicit_id_uses_direct_graph_neighbors_only() -> None:
    """An explicit ID bypasses text search and excludes second-hop papers."""
    retriever = RecordingCitationRetriever()

    result = trace_citation_graph(
        _state("What papers cite 2210.10773v1?"),  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
        InboundDirectionLLM(),  # type: ignore[arg-type]
        _citation_graph(),
    )

    assert retriever.search_called is False
    assert result["citation_seed_id"] == "2210.10773"
    assert result["citation_neighbor_ids"] == [
        "2301.11514",
        "2307.11085",
        "2311.14773",
    ]
    assert retriever.seed_requests == [["2210.10773"]]
    assert retriever.neighbor_requests == [
        ["2301.11514", "2307.11085", "2311.14773"]
    ]
    assert [chunk.paper_id for chunk in result["retrieved_chunks"]] == [
        "2210.10773",
        "2301.11514",
        "2307.11085",
        "2311.14773",
    ]
    assert "2503.01184" not in [
        chunk.paper_id for chunk in result["retrieved_chunks"]
    ]
    assert "second-hop" not in [
        chunk.paper_id for chunk in result["retrieved_chunks"]
    ]


def test_unknown_id_returns_safe_refusal_without_retrieval() -> None:
    """An unknown graph ID must not degrade into a text-search citation answer."""
    retriever = RecordingCitationRetriever()

    result = trace_citation_graph(
        _state("What papers cite 9999.99999?"),  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
        InboundDirectionLLM(),  # type: ignore[arg-type]
        _citation_graph(),
    )

    assert retriever.search_called is False
    assert result["retrieved_chunks"] == []
    assert result["citation_seed_id"] is None
    assert result["citation_neighbor_ids"] == []
    assert "can't identify arXiv ID 9999.99999" in result[
        "citation_resolution_error"
    ]


def test_exact_title_uses_direct_graph_neighbors_only() -> None:
    """An exact quoted title resolves before one-hop graph traversal."""
    retriever = RecordingCitationRetriever()

    result = trace_citation_graph(
        _state(
            "Which papers in this corpus cite “Anomaly Detection Requires "
            "Better Representations” by Tal Reiss?"
        ),  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
        InboundDirectionLLM(),  # type: ignore[arg-type]
        _citation_graph(),
    )

    assert retriever.search_called is False
    assert retriever.title_requests == [
        ("Anomaly Detection Requires Better Representations", "Tal Reiss")
    ]
    assert result["citation_seed_id"] == "2210.10773"
    assert result["citation_neighbor_ids"] == [
        "2301.11514",
        "2307.11085",
        "2311.14773",
    ]


def test_unresolved_quoted_title_refuses_without_general_search() -> None:
    """An unresolvable title must not fall back to semantic citation seeding."""
    retriever = RecordingCitationRetriever()
    result = trace_citation_graph(
        _state('What papers cite "Unknown Paper Title"?'),  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
        InboundDirectionLLM(),  # type: ignore[arg-type]
        _citation_graph(),
    )

    assert retriever.search_called is False
    assert "can't uniquely identify the titled paper" in result[
        "citation_resolution_error"
    ]


def test_title_lookup_normalizes_punctuation_and_validates_author() -> None:
    """Exact lookup tolerates formatting but rejects a conflicting author hint."""
    retriever = RetrieverService.__new__(RetrieverService)
    retriever._chunk_dict = {
        "chunk-1": {
            "paper_id": "2210.10773",
            "title": "Anomaly Detection Requires Better Representations",
            "authors": ["Tal Reiss", "Niv Cohen"],
            "year": 2022,
        }
    }

    assert retriever.resolve_paper_id_by_title(
        "Anomaly-Detection Requires Better Representations", "Tal Reiss"
    ) == "2210.10773"
    assert (
        retriever.resolve_paper_id_by_title(
            "Anomaly Detection Requires Better Representations", "Other Author"
        )
        is None
    )


def test_unresolved_citation_stops_before_llm_synthesis() -> None:
    """The agent returns an honest citation refusal without a provider request."""
    result = synthesise_answer(
        _state("What papers cite 9999.99999?")
        | {
            "citation_resolution_error": "I can't identify arXiv ID 9999.99999 "
            "in the configured citation graph."
        },
        FailingLLM(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )

    assert result["final_answer"].response.startswith("I can't identify")
    assert result["final_answer"].sources == []
    assert result["feedback_id"] is None


def test_citation_synthesis_normalizes_sources_to_graph_neighbors() -> None:
    """Synthesis cannot expose a source outside the direct graph-neighbor set."""
    result = synthesise_answer(
        _state("What papers cite 2210.10773?")
        | {
            "citation_seed_id": "2210.10773",
            "citation_neighbor_ids": ["2301.11514", "2307.11085", "2311.14773"],
            "retrieved_chunks": [
                RecordingCitationRetriever._chunk("2210.10773"),
                RecordingCitationRetriever._chunk("2301.11514"),
                RecordingCitationRetriever._chunk("2307.11085"),
                RecordingCitationRetriever._chunk("2311.14773"),
            ],
        },
        CitationResponderLLM(),  # type: ignore[arg-type]
        PassingPipeline(),  # type: ignore[arg-type]
        RecordingStore(),  # type: ignore[arg-type]
        RecordingMemory(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )

    assert result["final_answer"].sources == [
        "2301.11514",
        "2307.11085",
        "2311.14773",
    ]
