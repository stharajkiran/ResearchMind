from researchmind.guardrails.validators import (
    CitationGroundingValidator,
    PIIRedactionValidator,
    PipeLineResult,
    HallucinationScoreValidator,
    ResearchGapSchemaValidator,
)
from researchmind.embedding.models import BaseResearchEncoder

from researchmind.ingestion.models import (
    Chunk,
    ComparisonRAGResponse,
    RAGResponse,
    ResearchGapResponse,
)


class ValidatorPipeline:

    def __init__(
        self,
        corpus_paper_ids: set[str],
        encoder: BaseResearchEncoder,
    ):
        self._citation_validator = CitationGroundingValidator(corpus_paper_ids)
        self._pii_redaction_validator = PIIRedactionValidator()
        self._hallucination_validator = HallucinationScoreValidator(encoder)
        self._research_gap_validator = ResearchGapSchemaValidator()

    def run(
        self,
        response: ResearchGapResponse | RAGResponse | ComparisonRAGResponse,
        chunks: list[Chunk],
    ) -> PipeLineResult:

        validation_results = []
        redacted_text = None
        if isinstance(response, ResearchGapResponse):
            retrieved_paper_ids = set(c.paper_id for c in chunks)
            validation_results.append(
                self._research_gap_validator.validate(response, retrieved_paper_ids)
            )

        else:
            citation_validation = self._citation_validator.validate(response.sources)
            if isinstance(response, RAGResponse):
                text = response.response
            elif isinstance(response, ComparisonRAGResponse):
                text = response.comparison
            else:
                raise ValueError("Unsupported response type for validation")
            redacted_text, pii_validation = self._pii_redaction_validator.validate(text)
            hallucination_validation = self._hallucination_validator.validate(
                redacted_text, chunks
            )
            validation_results.extend(
                [citation_validation, pii_validation, hallucination_validation]
            )
        overall_passed = all(v.passed for v in validation_results)
        blocked = not overall_passed
        return PipeLineResult(
            results=validation_results,
            overall_passed=overall_passed,
            blocked=blocked,
            redacted_text=redacted_text,
        )
