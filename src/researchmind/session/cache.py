import hashlib
import json
import os

import redis
from researchmind.metrics import cache_hits_total


class QueryCache:
    def __init__(self):
        url = os.environ.get("REDIS_URL")
        self._client = redis.Redis.from_url(url, decode_responses=True) if url else None

    def _key(self, query: str) -> str:
        # normalize + hash
        normalized = query.lower().strip()
        return "cache:" + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> dict | None:
        if self._client is None:
            return None
        # look up key in Redis, return parsed JSON or None
        key = self._key(query)
        value = self._client.get(key)
        if value is None:
            return None
        cache_hits_total.inc()
        return json.loads(value)

    def set(self, query: str, answer: dict, ttl: int = 3600) -> None:
        if self._client is None:
            return
        # save answer as JSON with TTL
        key = self._key(query)
        self._client.set(key, json.dumps(answer), ex=ttl)

    def invalidate(self, query: str) -> None:
        if self._client is None:
            return
        self._client.delete(self._key(query))
