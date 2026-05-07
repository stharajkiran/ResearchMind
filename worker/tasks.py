import asyncio
from pathlib import Path
from researchmind.ingestion.downloader import download_pdf
from researchmind.ingestion.pdf_parser import parse_pdf
from researchmind.ingestion.models import RawPaper
from researchmind.chunker.chunker import FixedSizeChunker
from researchmind.utils.find_root import find_project_root
import arxiv

from dotenv import load_dotenv

load_dotenv()


from celery import Celery
import os

celery_app = Celery(
    "researchmind",
    broker=os.environ["REDIS_URL"],
    backend=os.environ["REDIS_URL"],
)


@celery_app.task
def ingest_paper_task(paper_url: str) -> dict:

    # extract paper ID from URL e.g. https://arxiv.org/abs/2104.14294
    paper_id = paper_url.rstrip("/").split("/")[-1]

    # fetch metadata
    client = arxiv.Client()
    results = list(client.results(arxiv.Search(id_list=[paper_id])))
    if not results:
        return {"status": "error", "message": f"Paper {paper_id} not found on arXiv"}
    result = results[0]

    raw_paper = RawPaper(
        paper_id=paper_id,
        title=result.title,
        abstract=result.summary,
        categories=result.categories,
        authors=[a.name for a in result.authors],
        published=result.published.date(),
        pdf_url=result.pdf_url,
    )

    # download PDF
    root = find_project_root()
    pdf_dir = root / "data" / "raw" / "arxiv_pdfs"
    pdf_path = asyncio.run(download_pdf(paper_id, result.pdf_url, pdf_dir))
    if pdf_path is None:
        return {"status": "error", "message": f"Failed to download PDF for {paper_id}"}

    # parse + chunk
    parsed = parse_pdf(Path(pdf_path), raw_paper)
    chunks = FixedSizeChunker().chunk(parsed)

    return {
        "status": "success",
        "paper_id": paper_id,
        "title": result.title,
        "chunks_created": len(chunks),
    }
