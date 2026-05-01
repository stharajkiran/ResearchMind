from datetime import date
from pathlib import Path
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ingestion data models and Parsing logic
# ---------------------------------------------------------------------------


class S2Author(BaseModel):
    author_id: str
    name: str


class RawPaper(BaseModel):
    paper_id: str  # arXiv ID e.g. "2301.07041"
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]  # ["cs.LG", "cs.AI"]
    published: date
    pdf_url: str
    latex_path: Path | None = None  # data/raw/arxiv/{paper_id}/
    s2_paper_id: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    s2_authors: list[S2Author] | None = None  # includes S2 author IDs

    class Config:
        frozen = True  # immutable after creation


class ParsedPaper(BaseModel):
    paper: RawPaper
    sections: dict[str, str]  # {"Introduction": "text", "Methods": "text", ...}


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: str
    text: str
    page: int | None = None
    authors: list[str]
    year: int
    title: str


# ---------------------------------------------------------------------------
# RAG response schemas
# ---------------------------------------------------------------------------


# Define your strict schema
class RAGResponse(BaseModel):
    response: str = Field(
        ...,
        description="The detailed, technical answer synthesized from the provided research chunks. Adhere strictly to academic tone.",
    )
    sources: list[str] = Field(
        ...,
        description="A list of paper IDs that directly contributed to the answer. Only include papers that provided actual information.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A confidence score between 0.0 and 1.0 representing how well the context supports the answer.",
    )
    citations: list[str] = Field(
        ...,
        description="List of direct quotes from the provided chunks that support the response.",
    )

    # the chunk texts that were actually retrieved by your system at query time
    # these texts are
    contexts: list[str] = Field(
        default=[], description="Retrieved chunk texts used for synthesis"
    )


class MethodSummary(BaseModel):
    method: str = Field(
        ..., description="The name of the method or approach being summarised."
    )
    summary: str = Field(
        ...,
        description="Technical summary of the method, its approach, and key findings from the retrieved chunks.",
    )
    sources: list[str] = Field(
        ..., description="Paper IDs that directly support this method summary."
    )


class ComparisonRAGResponse(BaseModel):
    summaries: list[MethodSummary] = Field(
        ...,
        description="One summary per method being compared, grounded in retrieved chunks for that method.",
    )
    comparison: str = Field(
        ...,
        description="Structured comparison across all methods covering key differences, tradeoffs, and which performs better and why.",
    )
    sources: list[str] = Field(
        ...,
        description="All paper IDs that contributed to the comparison across all methods.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score representing how well the retrieved chunks support the comparison.",
    )
    citations: list[str] = Field(
        ...,
        description="Direct quotes from retrieved chunks that support the comparison.",
    )


# ---------------------------------------------------------------------------
# Research gap response schema
# ---------------------------------------------------------------------------


class ResearchGap(BaseModel):
    description: str = Field(
        ...,
        description="A specific unsolved problem, missing benchmark, unexplored method combination, or contradictory finding identified across the retrieved papers.",
    )
    supporting_paper_ids: list[str] = Field(
        ...,
        description="Paper IDs from the retrieved set that evidence this gap. Every ID must exist in the provided chunks — do not invent paper IDs.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that this is a genuine open problem based on evidence in the retrieved chunks. 1.0 means explicitly stated by multiple papers, 0.5 means implied.",
    )


class ResearchGapResponse(BaseModel):
    gaps: list[ResearchGap] = Field(
        ...,
        description="List of research gaps identified across the retrieved papers. Each gap must be grounded in the provided chunks.",
    )
    topic: str = Field(
        ...,
        description="The research topic these gaps belong to, as inferred from the query.",
    )
