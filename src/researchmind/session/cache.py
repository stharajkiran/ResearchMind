import hashlib
import json

from researchmind.metrics import cache_hits_total
from researchmind.session.interfaces import CacheBackend


class QueryCache:
    def __init__(self, backend: CacheBackend):
        self._backend = backend

    def _key(self, query: str) -> str:
        normalized = query.lower().strip()
        return "cache:" + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> dict | None:
        value = self._backend.get(self._key(query))
        if value is None:
            return None
        cache_hits_total.inc()
        return json.loads(value)

    def set(self, query: str, answer: dict, ttl: int = 3600) -> None:
        self._backend.set(self._key(query), json.dumps(answer), ttl)

    def invalidate(self, query: str) -> None:
        self._backend.delete(self._key(query))
