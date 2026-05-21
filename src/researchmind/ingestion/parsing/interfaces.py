from abc import ABC, abstractmethod
from pathlib import Path

from researchmind.ingestion.models import ParsedPaper, RawPaper


class PaperParser(ABC):
    """Parse a downloaded PDF into structured sections."""

    @abstractmethod
    def parse(self, pdf_path: Path, paper: RawPaper) -> ParsedPaper:
        """Extract and section-split text from a PDF file."""
        ...
