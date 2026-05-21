from abc import ABC, abstractmethod
from pathlib import Path

from researchmind.ingestion.models import RawPaper


class PDFDownloader(ABC):
    """Download PDFs for a list of papers to a local directory."""

    @abstractmethod
    async def download(self, paper_id: str, url: str, dest_dir: Path) -> Path | None:
        """Download a single PDF. Returns local path on success, None on failure."""
        ...

    @abstractmethod
    async def download_batch(
        self,
        papers: list[RawPaper],
        dest_dir: Path,
        concurrency: int = 10,
    ) -> list[Path | None]:
        """Download PDFs for a batch of papers concurrently."""
        ...
