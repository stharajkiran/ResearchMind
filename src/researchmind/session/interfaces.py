from abc import ABC, abstractmethod


class CacheBackend(ABC):
    """Abstract interface for key-value cache storage with optional TTL."""

    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: int | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...
