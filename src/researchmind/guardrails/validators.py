from pydantic import BaseModel
import re
from researchmind.embedding.models import BaseResearchEncoder

from researchmind.ingestion.models import Chunk, ResearchGapResponse


class ValidationResult(BaseModel):
    score: float
    validator: str
    passed: bool


class PipeLineResult(BaseModel):
    results: list[ValidationResult]
    overall_passed: bool
    blocked: bool
    redacted_text: str | None = None


class CitationGroundingValidator:
    """Validates whether a citation grounding is valid based on the presence of a cited paper in the retrieved chunks."""

    def __init__(self, corpus_paper_ids: set[str]):
        self.corpus_paper_ids = corpus_paper_ids

    def validate(self, cited_papers: list[str]) -> ValidationResult:
        if not cited_papers:
            score = 1.0
            passed = True
        else:
            valid_citations = sum(
                1 for paper_id in cited_papers if paper_id in self.corpus_paper_ids
            )
            score = valid_citations / len(cited_papers)
            passed = (
                score >= 0.7
            )  # pass if more than 50% of cited papers are in the corpus
        return ValidationResult(
            score=score,
            validator="CitationGroundingValidator",
            passed=passed,
        )


class PIIRedactionValidator:
    """Validates that no personally identifiable information (PII) is present in the text."""

    EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

    def validate(self, text: str) -> tuple[str, ValidationResult]:
        matches = self.EMAIL_PATTERN.findall(text)
        redacted_text = self.EMAIL_PATTERN.sub("[REDACTED]", text)

        match_count = len(matches)
        score = 1.0 if match_count == 0 else max(0.0, 1.0 - 0.1 * match_count)

        return redacted_text, ValidationResult(
            score=score,
            validator="PIIRedactionValidator",
            passed=True,
        )


class HallucinationScoreValidator:
    """Validates the hallucination score of generated content based on a threshold."""

    def __init__(self, encoder: BaseResearchEncoder):
        self.encoder = encoder

    def validate(self, answer: str, chunks: list[Chunk]) -> ValidationResult:
        if not chunks:
            max_similarity = 1.0
            passed = True
        else:
            answer_embedding = self.encoder.encode([answer])[0]
            chunk_embeddings = self.encoder.encode([c.text for c in chunks])

            # Calculate similarities for every chunk individually
            similarities = chunk_embeddings @ answer_embedding

            # Take the highest similarity found
            max_similarity = similarities.max()
            passed = max_similarity >= 0.7

            # mean_embedding = chunk_embeddings.mean(axis=0)
            # cosine_similarity = (
            #     answer_embedding @ mean_embedding
            # )  # encode has embeddings normalized to unit length, so dot product is cosine similarity
            # passed = cosine_similarity >= 0.7

        return ValidationResult(
            score=max_similarity,
            validator="HallucinationScoreValidator",
            passed=passed,
        )


class ResearchGapSchemaValidator:

    def validate(
        self, response: ResearchGapResponse, retrieved_paper_ids: set[str]
    ) -> ValidationResult:
        # set of all supporting paper IDS
        supporting_paper_ids = set(
            pid for gap in response.gaps for pid in gap.supporting_paper_ids
        )
        # check if any supporting papers id is not in retrieved paper ids, which indicates potential hallucination
        has_hallucination = any(
            pid for pid in supporting_paper_ids if pid not in retrieved_paper_ids
        )

        score = 0.0 if has_hallucination else 1.0
        passed = not has_hallucination
        return ValidationResult(
            score=score,
            validator="ResearchGapSchemaValidator",
            passed=passed,
        )
    
