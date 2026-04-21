import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

_BASE = "https://api.openalex.org"
_ARXIV_PREFIX = "https://arxiv.org/abs/"

_PARAMS = {
    "mailto": os.environ.get("OPENALEX_MAILTO", ""),
    "api_key": os.environ.get("OPENALEX_API_KEY", ""),
}

_BATCH_SIZE = 50  # max Work IDs per batch resolve call


def get_work(arxiv_id: str) -> dict | None:
    """
    Fetch a single OpenAlex Work object by arXiv ID.
    """
    api_key = _PARAMS.get("api_key", "")
    url = (
        f"{_BASE}/works"
        f"?filter=doi:10.48550/arXiv.{arxiv_id}"
        f"&per-page=1"
        f"&api_key={api_key}"
    )
    try:
        r = httpx.get(url, timeout=15.0)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0] if results else None
    except Exception as exc:
        logger.warning("OpenAlex get_work failed %s: %s", arxiv_id, exc)
        return None

def _resolve_openalex_ids_to_arxiv(openalex_ids: list[str]) -> list[str]:
    """
    Batch-resolve OpenAlex Work IDs to arXiv IDs.

    Naive approach: 1 API call per ID = O(n) calls.
    This approach: ceil(n / batch_size) calls — the N+1 fix.

    openalex_ids: list of full URLs like "https://openalex.org/W2741809807"
                  OR bare IDs like "W2741809807"
    """
    # Normalise to bare IDs (strip URL prefix if present)
    work_ids = [oid.split("/")[-1] for oid in openalex_ids]
    arxiv_ids: list[str] = []

    for i in range(0, len(work_ids), _BATCH_SIZE):
        chunk = work_ids[i : i + _BATCH_SIZE]
        pipe_filter = "|".join(chunk)
        url = f"{_BASE}/works?filter=openalex:{pipe_filter}&select=id,ids&per-page={_BATCH_SIZE}&api_key={_PARAMS.get('api_key', '')}"

        try:
            # openalex api call
            r = httpx.get(url, timeout=15.0)
            data = r.json() if r.is_success else None
            if not data:
                continue
            for work in data.get("results", []):
                arxiv_url = (work.get("ids") or {}).get("arxiv", "")
                if arxiv_url.startswith(_ARXIV_PREFIX):
                    raw_id = arxiv_url.replace(_ARXIV_PREFIX, "").strip()
                    # Normalise: "1706.03762v7" → "1706.03762"
                    arxiv_ids.append(raw_id.split("v")[0])
        except Exception as exc:
            logger.warning("OpenAlex batch resolve failed for IDs %s: %s", chunk, exc)
        time.sleep(0.15)  # stay within polite pool

    return arxiv_ids


def get_referenced_arxiv_ids(arxiv_id: str) -> list[str]:
    """
    Return arXiv IDs of papers this paper's reference list (outbound edges).
    Uses batch resolution — one OpenAlex call per 50 references.
    """
    work = get_work(arxiv_id)
    if not work:
        return []
    # referenced_works is a list of OpenAlex Work IDs (e.g. "https://openalex.org/W2741809807")
    referenced = work.get("referenced_works", [])
    if not referenced:
        return []
    return _resolve_openalex_ids_to_arxiv(referenced)


def enrich_paper(arxiv_id: str) -> Optional[dict]:
    """
    Return enrichment metadata for Block 6 (citation graph + ranking signal).

    citation_velocity = cited_by_count / max(2026 - publication_year, 1)
    Floor of 1 year prevents division-by-zero on papers published this year.
    """
    work = get_work(arxiv_id)
    if not work:
        return None

    year = work.get("publication_year") or 0
    cited_by = work.get("cited_by_count") or 0
    velocity = round(cited_by / max(2026 - year, 1), 2) if year else 0.0

    primary_loc = work.get("primary_location") or {}
    venue = (primary_loc.get("source") or {}).get("display_name")

    return {
        "openalex_id": work.get("id"),
        "citation_count": cited_by,
        "publication_year": year,
        "citation_velocity": velocity,
        "venue": venue,
    }
