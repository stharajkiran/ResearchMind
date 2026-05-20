import os

import redis

from researchmind.session.interfaces import CacheBackend


class RedisCache(CacheBackend):
    """Redis-backed cache. Gracefully degrades to no-op if REDIS_URL is unset."""

    def __init__(self, url: str | None = None):
        resolved = url or os.environ.get("REDIS_URL")
        self._client = redis.Redis.from_url(resolved, decode_responses=True) if resolved else None

    def get(self, key: str) -> str | None:
        if self._client is None:
            return None
        return self._client.get(key)

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        if self._client is None:
            return
        self._client.set(key, value, ex=ttl)

    def delete(self, key: str) -> None:
        if self._client is None:
            return
        self._client.delete(key)
