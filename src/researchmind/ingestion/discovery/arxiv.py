import logging
from datetime import date

import arxiv
from tqdm import tqdm

from researchmind.ingestion.discovery.interfaces import PaperSource
from researchmind.ingestion.models import RawPaper

logger = logging.getLogger(__name__)

_BATCH_SIZE = 200
_CLIENT_DEFAULTS = dict(page_size=100, delay_seconds=10.0, num_retries=3)


def _to_raw_paper(result: arxiv.Result) -> RawPaper:
    return RawPaper(
        paper_id=result.get_short_id(),
        title=result.title,
        authors=[a.name for a in result.authors],
        abstract=result.summary,
        categories=result.categories,
        published=result.published.date(),
        pdf_url=result.pdf_url,
    )


class ArxivSource(PaperSource):
    """arXiv implementation of PaperSource.

    fetch_by_query: category-code + date-range search (cs.CV, cs.LG, ...).
    fetch_by_ids:   targeted fetch for known arXiv IDs — used for citation expansion.
    """

    def fetch_by_query(
        self,
        categories: list[str],
        start_date: date,
        end_date: date,
        max_results: int = 1000,
        keywords: list[str] | None = None,
    ) -> list[RawPaper]:
        date_filter = (
            f"submittedDate:[{start_date.strftime('%Y%m%d')}000000"
            f" TO {end_date.strftime('%Y%m%d')}235959]"
        )
        cat_clause = f"({' OR '.join(f'cat:{c}' for c in categories)})"
        if keywords:
            kw_clause = f"({' OR '.join(f'ti:{kw!r}' for kw in keywords)})"
            query = f"{cat_clause} AND {kw_clause} AND {date_filter}"
        else:
            query = f"{cat_clause} AND {date_filter}"
        client = arxiv.Client(**_CLIENT_DEFAULTS)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        papers: list[RawPaper] = []
        skipped = 0
        for result in tqdm(client.results(search), desc="arXiv query"):
            pub = result.published.date()
            if pub < start_date:
                break
            if pub > end_date:
                skipped += 1
                continue
            papers.append(_to_raw_paper(result))
        if skipped:
            logger.warning("Skipped %d papers published after end_date", skipped)
        logger.info("fetch_by_query: returned %d papers", len(papers))
        return papers

    def fetch_by_ids(self, paper_ids: list[str]) -> list[RawPaper]:
        if not paper_ids:
            return []
        all_papers: list[RawPaper] = []
        for i in range(0, len(paper_ids), _BATCH_SIZE):
            batch = paper_ids[i : i + _BATCH_SIZE]
            client = arxiv.Client(**_CLIENT_DEFAULTS)
            search = arxiv.Search(id_list=batch)
            for result in tqdm(client.results(search), desc="arXiv ID fetch", total=len(batch)):
                all_papers.append(_to_raw_paper(result))
        logger.info("fetch_by_ids: returned %d papers for %d requested IDs", len(all_papers), len(paper_ids))
        return all_papers
