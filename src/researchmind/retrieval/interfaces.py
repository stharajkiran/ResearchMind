from abc import ABC, abstractmethod

import numpy as np

from researchmind.ingestion.models import Chunk


class DenseIndex(ABC):
    """Abstract interface for dense vector similarity search."""

    @abstractmethod
    def build(self, embeddings: np.ndarray, ids: list[str]) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int = 10) -> list[str]: ...


class SparseIndex(ABC):
    """Abstract interface for sparse keyword search."""

    @abstractmethod
    def build(self, texts: list[str], ids: list[str]) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[str]: ...


class FilteredStore(ABC):
    """Abstract interface for metadata-filtered vector search."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def query(self, query: str, k: int = 10, filters: dict | None = None) -> list[Chunk]: ...
