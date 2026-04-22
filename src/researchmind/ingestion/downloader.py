import httpx
import asyncio
from pathlib import Path
import json
import random
import logging
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


logger = logging.getLogger(__name__)


# async function to download PDFs with retries and concurrency control
async def download_pdf(paper_id: str, pdf_url: str, output_dir: Path) -> str | None:

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{paper_id}.pdf"

    # If file already exists, skip download
    if output_path.exists():
        return str(output_path)
    
    async with httpx.AsyncClient() as client:
        for attempt in range(3):  # 3 attempts with exponential backoff
            try:
                response = await client.get(pdf_url, timeout=30)
                response.raise_for_status()
                output_path.write_bytes(response.content)
                return str(output_path)
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed {paper_id}: {e}")
                    return None
                await asyncio.sleep(2**attempt)


async def download_batch(
    papers: list[dict[str, str]], output_dir: Path, concurrency: int = 10
) -> list[str | None]:
    """Download a batch of PDFs concurrently with real-time progress tracking."""
    semaphore = asyncio.Semaphore(concurrency)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(f"[cyan]Downloading {len(papers)} papers...", total=len(papers))
        
        async def bounded_download(paper):
            async with semaphore:
                result = await download_pdf(paper["paper_id"], paper["pdf_url"], output_dir)
                progress.update(task_id, advance=1)
                return result
        
        results = await asyncio.gather(*[bounded_download(p) for p in papers])
    return list(results)


if __name__ == "__main__":
    output_dir = Path("data/raw/arxiv_pdfs")
    papers_path = Path("data/processed/papers.jsonl")
    with papers_path.open("r", encoding="utf-8") as f:
        papers = [json.loads(line) for line in f]

    # get randomly 2000 papers with seed
    random.seed(42)
    papers = random.sample(papers, min(2000, len(papers)))
    
    logger.info(f"Starting download of {len(papers)} PDFs to {output_dir}")
    results = asyncio.run(download_batch(papers, output_dir))
    failed = results.count(None)
    logger.info(f"Done. {len(results) - failed} succeeded, {failed} failed.")
