"""Offline API route tests for the supported ResearchMind release path."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import app
from researchmind.ingestion.models import Chunk


class StubRetriever:
    """Return a fixed scored chunk and record API-to-retriever requests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search_scored(
        self,
        query: str,
        k: int = 10,
        mode: str = "standard",
        recency_decay_rate: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Record a request and return one deterministic scored corpus chunk."""
        self.calls.append(
            {
                "query": query,
                "k": k,
                "mode": mode,
                "recency_decay_rate": recency_decay_rate,
            }
        )
        return [
            (
                Chunk(
                    chunk_id="chunk-1",
                    paper_id="paper-1",
                    section="abstract",
                    text="Transformer attention is used in this fixture.",
                    authors=["Test Author"],
                    year=2024,
                    title="Fixture Paper",
                ),
                0.875,
            )
        ]


@asynccontextmanager
async def _offline_lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Provide test-only application state without production startup work."""
    application.state.retriever = StubRetriever()
    yield


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Create an API client whose lifespan has no service dependencies."""
    monkeypatch.setattr(app.router, "lifespan_context", _offline_lifespan)
    with TestClient(app) as client:
        yield client


def test_search_returns_retriever_results(api_client: TestClient) -> None:
    """The search route serializes chunks and forwards request options."""
    response = api_client.post(
        "/search",
        json={"query": "transformer attention", "k": 5},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "chunk_id": "chunk-1",
            "paper_id": "paper-1",
            "section": "abstract",
            "text": "Transformer attention is used in this fixture.",
            "page": None,
            "authors": ["Test Author"],
            "year": 2024,
            "title": "Fixture Paper",
            "score": 0.875,
        }
    ]
    assert app.state.retriever.calls == [
        {
            "query": "transformer attention",
            "k": 5,
            "mode": "standard",
            "recency_decay_rate": None,
        }
    ]


def test_health_returns_ok(api_client: TestClient) -> None:
    """The health endpoint is available without application services."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
