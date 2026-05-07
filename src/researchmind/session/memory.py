import redis
import os
import json

from researchmind.ingestion.models import Chunk


class SessionMemory:
    """
    Manages the memory of a research session, including storing and retrieving information relevant to the session's context.
    """

    def __init__(self):
        self.redis_client = redis.Redis.from_url(
            os.environ["REDIS_URL"],
            decode_responses=True,
        )

    def save(self, session_id: str, chunks: list[Chunk], answer):
        """
        Save information to the session memory, including retrieved chunks and generated answer.

        Args:
            session_id (_type_): _description_
            chunks (_type_): _description_
            answer (_type_): _description_
        """
        key = f"session:{session_id}"
        data = {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "answer": answer.model_dump(),
        }
        self.redis_client.set(key, json.dumps(data), ex=3600)  # expire in 1 hour

    def load(self, session_id):
        """
        Load information from the session memory for a given session ID.

        Args:
            session_id (_type_): _description_
        """
        key = f"session:{session_id}"
        data = self.redis_client.get(key)
        if data is None:
            return None
        return json.loads(data)
