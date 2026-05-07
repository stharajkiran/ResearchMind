import hashlib
import json
import os

import redis


class QueryCache:
    def __init__(self):
        self._client = redis.Redis.from_url(
            os.environ["REDIS_URL"], decode_responses=True
        )

    def _key(self, query: str) -> str:
        # normalize + hash
        normalized = query.lower().strip()
        return "cache:" + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> dict | None:
        # look up key in Redis, return parsed JSON or None
        key = self._key(query)
        value = self._client.get(key)
        if value is None:
            return None
        return json.loads(value)

    def set(self, query: str, answer: dict, ttl: int = 3600) -> None:
        # save answer as JSON with TTL
        key = self._key(query)
        self._client.set(key, json.dumps(answer), ex=ttl)

    def invalidate(self, query: str) -> None:
        self._client.delete(self._key(query))