import json
import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv

load_dotenv()

class FeedbackStore:
    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or os.environ["POSTGRES_DSN"]

    def _conn(self):
        return psycopg2.connect(self._dsn)

    def create_tables(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS feedback (
            id                      SERIAL PRIMARY KEY,
            session_id              TEXT,
            query                   TEXT NOT NULL,
            intent                  TEXT,
            answer_json             JSONB,

            hallucination_score     FLOAT,
            citation_grounding_score FLOAT,
            validation_passed       BOOLEAN,
            validator_results       JSONB,

            retrieved_paper_ids     TEXT[],
            retrieved_chunk_ids     TEXT[],

            ragas_faithfulness      FLOAT,
            ragas_context_precision FLOAT,
            ragas_answer_relevancy  FLOAT,

            rating                  INT,
            created_at              TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback (rating);
        CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback (session_id);
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def save_feedback(
        self,
        session_id: str,
        query: str,
        intent: str,
        answer_json: dict[str, Any],
        hallucination_score: float | None,
        citation_grounding_score: float | None,
        validation_passed: bool,
        validator_results: list[dict[str, Any]],
        retrieved_paper_ids: list[str],
        retrieved_chunk_ids: list[str],
        rating: int | None = None,
    ) -> int:
        sql = """
        INSERT INTO feedback (
            session_id, query, intent, answer_json,
            hallucination_score, citation_grounding_score,
            validation_passed, validator_results,
            retrieved_paper_ids, retrieved_chunk_ids, rating
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        session_id,
                        query,
                        intent,
                        json.dumps(answer_json),
                        hallucination_score,
                        citation_grounding_score,
                        validation_passed,
                        json.dumps(validator_results),
                        retrieved_paper_ids,
                        retrieved_chunk_ids,
                        rating,
                    ),
                )
                return cur.fetchone()[0]

    def update_rating(self, feedback_id: int, rating: int) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE feedback SET rating = %s WHERE id = %s",
                    (rating, feedback_id),
                )

    def update_ragas(
        self,
        feedback_id: int,
        faithfulness: float,
        context_precision: float,
        answer_relevancy: float,
    ) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE feedback
                       SET ragas_faithfulness = %s,
                           ragas_context_precision = %s,
                           ragas_answer_relevancy = %s
                       WHERE id = %s""",
                    (faithfulness, context_precision, answer_relevancy, feedback_id),
                )

    def get_low_rated(self, threshold: int = 3) -> list[dict[str, Any]]:
        sql = """
        SELECT * FROM feedback
        WHERE rating IS NOT NULL AND rating <= %s
        ORDER BY created_at DESC;
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (threshold,))
                return [dict(r) for r in cur.fetchall()]

    def get_all_with_scores(self) -> list[dict[str, Any]]:
        sql = """
        SELECT * FROM feedback
        WHERE ragas_faithfulness IS NOT NULL
        ORDER BY created_at DESC;
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                return [dict(r) for r in cur.fetchall()]

    def get_all(self) -> list[dict]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM feedback ORDER BY created_at DESC;")
                return [dict(r) for r in cur.fetchall()]
