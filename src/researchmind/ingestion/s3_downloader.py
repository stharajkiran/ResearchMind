import tarfile
import boto3
from pathlib import Path

S3_BUCKET = "arxiv"
RAW_DIR = Path("data/raw/arxiv")

def download_latex(paper_id: str) -> Path | None:
    ...