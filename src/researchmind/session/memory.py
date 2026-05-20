import json

from researchmind.ingestion.models import Chunk
from researchmind.session.interfaces import CacheBackend


class SessionMemory:
    def __init__(self, backend: CacheBackend):
        self._backend = backend

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def save(self, session_id: str, chunks: list[Chunk], answer) -> None:
        data = {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "answer": answer.model_dump(),
        }
        self._backend.set(self._key(session_id), json.dumps(data), ttl=3600)

    def load(self, session_id: str) -> dict | None:
        value = self._backend.get(self._key(session_id))
        if value is None:
            return None
        return json.loads(value)
