from researchmind.ingestion.discovery.interfaces import (
    CitationSource,
    PaperEnricher,
    PaperIDSource,
    PaperSource,
)
from researchmind.ingestion.discovery.arxiv_source import ArxivSource
from researchmind.ingestion.discovery.semantic_scholar_search import SemanticScholarSearch
from researchmind.ingestion.discovery.semantic_scholar_enricher import SemanticScholarEnricher
from researchmind.ingestion.discovery.semantic_scholar_citations import SemanticScholarCitationSource
from researchmind.ingestion.discovery.semantic_scholar_recommendations import SemanticScholarRecommendationSource

__all__ = [
    # Interfaces
    "PaperSource",
    "PaperIDSource",
    "CitationSource",
    "PaperEnricher",
    # arXiv — full paper fetch
    "ArxivSource",
    # Semantic Scholar — ID discovery
    "SemanticScholarSearch",
    "SemanticScholarRecommendationSource",
    # Semantic Scholar — enrichment + citation graph
    "SemanticScholarEnricher",
    "SemanticScholarCitationSource",
]
