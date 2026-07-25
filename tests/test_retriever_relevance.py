"""Deterministic unit tests for hybrid retrieval result handling."""

import json
from pathlib import Path

import numpy as np

from researchmind.retrieval.retriever import RetrieverService


class FakeDenseIndex:
    """Return configured dense candidate IDs for retrieval tests."""

    def __init__(self, candidate_ids: list[str]) -> None:
        self._candidate_ids = candidate_ids

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[str]:
        """Return the requested number of deterministic dense candidates."""
        return self._candidate_ids[:k]


class FakeSparseIndex:
    """Return configured sparse candidate IDs for retrieval tests."""

    def __init__(self, candidate_ids: list[str]) -> None:
        self._candidate_ids = candidate_ids

    def search(self, query: str, k: int = 10) -> list[str]:
        """Return the requested number of deterministic sparse candidates."""
        return self._candidate_ids[:k]


class FakeFilteredStore:
    """Placeholder metadata store; filtering is outside this focused suite."""

    def query(self, query: str, k: int = 10, filters: dict | None = None) -> list:
        """Return no metadata-filtered results for these fixtures."""
        return []


class FakeEncoder:
    """Provide a fixed query embedding for dense retrieval fixtures."""

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        """Return one fixed vector per input text."""
        return np.ones((len(texts), 2), dtype=float)


def _write_chunks(path: Path) -> None:
    """Write the minimal indexed chunk corpus used by these tests."""
    chunk = {
        "chunk_id": "chunk-1",
        "paper_id": "paper-1",
        "section": "abstract",
        "text": "OOD detection evidence",
        "authors": [],
        "year": 2024,
        "title": "OOD paper",
    }
    path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")


def _build_retriever(
    chunks_path: Path,
    dense_ids: list[str],
    sparse_ids: list[str],
) -> RetrieverService:
    """Build a retriever with deterministic backends and no relevance gate."""
    return RetrieverService(
        dense=FakeDenseIndex(dense_ids),
        sparse=FakeSparseIndex(sparse_ids),
        filtered=FakeFilteredStore(),
        encoder=FakeEncoder(),
        chunks_path=chunks_path,
    )


def test_search_returns_hybrid_result_for_available_candidate(tmp_path: Path) -> None:
    """A candidate returned by both backends becomes a corpus chunk."""
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    results = _build_retriever(
        chunks_path,
        dense_ids=["chunk-1"],
        sparse_ids=["chunk-1"],
    ).search("OOD detection", k=1)

    assert [chunk.chunk_id for chunk in results] == ["chunk-1"]
    assert results[0].paper_id == "paper-1"


def test_search_returns_empty_list_when_backends_return_no_candidates(
    tmp_path: Path,
) -> None:
    """An empty retrieval backend response remains a valid empty API result."""
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    results = _build_retriever(
        chunks_path,
        dense_ids=[],
        sparse_ids=[],
    ).search("unavailable topic", k=5)

    assert results == []
