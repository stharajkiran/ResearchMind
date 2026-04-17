import arxiv
from datetime import date
from .models import RawPaper
from tqdm import tqdm


def fetch_papers(
    categories: list[str],
    start_date: date,
    end_date: date,
    max_results: int = 1000,
) -> list[RawPaper]:

    # " OR ".join(...) → "cat:cs.LG OR cat:cs.AI OR cat:cs.CL"
    query = " OR ".join(f"cat:{c}" for c in categories)

    # arxiv objects
    client = arxiv.Client(
        page_size=100,
        delay_seconds=5.0,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = client.results(search)
    papers = []
    for result in tqdm(results, desc="Fetching papers"):
        # Results are sorted by submission date descending.
        if result.published.date() < start_date:
            break
        if result.published.date() > end_date:
            continue
        papers.append(
            RawPaper(
                paper_id=result.get_short_id(),
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                categories=result.categories,
                published=result.published.date(),
                pdf_url=result.pdf_url,
            )
        )
    return papers
