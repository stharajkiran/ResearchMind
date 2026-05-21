from abc import ABC, abstractmethod
from typing import Any


class FeedbackStore(ABC):
    """Abstract interface for persisting query feedback and evaluation scores."""

    @abstractmethod
    def create_tables(self) -> None: ...

    @abstractmethod
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
    ) -> int | None: ...

    @abstractmethod
    def update_rating(self, feedback_id: int, rating: int) -> None: ...

    @abstractmethod
    def update_ragas(
        self,
        feedback_id: int,
        faithfulness: float,
        context_precision: float,
        answer_relevancy: float,
    ) -> None: ...

    @abstractmethod
    def get_low_rated(self, threshold: int = 3) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_all_with_scores(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_all(self) -> list[dict[str, Any]]: ...
