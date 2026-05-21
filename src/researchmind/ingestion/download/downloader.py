import asyncio
import logging
from pathlib import Path

import httpx
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from researchmind.ingestion.download.interfaces import PDFDownloader
from researchmind.ingestion.models import RawPaper

logger = logging.getLogger(__name__)


class HttpPDFDownloader(PDFDownloader):
    """Downloads PDFs over HTTP with retries and concurrency control.

    A single AsyncClient is shared across all downloads in a batch —
    this keeps the connection pool alive and avoids per-file overhead.
    """

    async def download(
        self,
        paper_id: str,
        url: str,
        dest_dir: Path,
        client: httpx.AsyncClient | None = None,
    ) -> Path | None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{paper_id}.pdf"
        if dest.exists():
            return dest

        own_client = client is None
        if own_client:
            client = httpx.AsyncClient()

        try:
            for attempt in range(3):
                try:
                    response = await client.get(url, timeout=30)
                    response.raise_for_status()
                    dest.write_bytes(response.content)
                    return dest
                except Exception as exc:
                    if attempt == 2:
                        logger.error("Failed %s after 3 attempts: %s", paper_id, exc)
                        return None
                    await asyncio.sleep(2 ** attempt)
        finally:
            if own_client:
                await client.aclose()

        return None

    async def download_batch(
        self,
        papers: list[RawPaper],
        dest_dir: Path,
        concurrency: int = 10,
    ) -> list[Path | None]:
        semaphore = asyncio.Semaphore(concurrency)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(
                "Downloading %d papers..." % len(papers), total=len(papers)
            )

            async with httpx.AsyncClient() as client:

                async def _bounded(paper: RawPaper) -> Path | None:
                    async with semaphore:
                        result = await self.download(paper.paper_id, paper.pdf_url, dest_dir, client)
                        progress.update(task_id, advance=1)
                        return result

                results = await asyncio.gather(*[_bounded(p) for p in papers])

        failed = results.count(None)
        logger.info(
            "download_batch: %d succeeded, %d failed out of %d",
            len(results) - failed,
            failed,
            len(results),
        )
        return list(results)
