from researchmind.ingestion.discovery import (
    ArxivSource,
    CitationSource,
    PaperEnricher,
    PaperIDSource,
    PaperSource,
    SemanticScholarCitationSource,
    SemanticScholarEnricher,
    SemanticScholarRecommendationSource,
    SemanticScholarSearch,
)
from researchmind.ingestion.download import HttpPDFDownloader, PDFDownloader
from researchmind.ingestion.parsing import PaperParser, PyMuPDFParser
from researchmind.ingestion.chunking import Chunker, FixedSizeChunker, SectionChunker

__all__ = [
    # Interfaces
    "PaperSource",
    "PaperIDSource",
    "CitationSource",
    "PaperEnricher",
    # Discovery — full papers
    "ArxivSource",
    # Discovery — ID collectors
    "SemanticScholarSearch",
    "SemanticScholarRecommendationSource",
    # Discovery — enrichment + citation graph
    "SemanticScholarEnricher",
    "SemanticScholarCitationSource",
    # Download
    "PDFDownloader",
    "HttpPDFDownloader",
    # Parsing
    "PaperParser",
    "PyMuPDFParser",
    # Chunking
    "Chunker",
    "SectionChunker",
    "FixedSizeChunker",
]
