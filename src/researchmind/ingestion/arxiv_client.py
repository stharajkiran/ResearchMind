import arxiv
from datetime import date
from .models import RawPaper
from tqdm import tqdm


def fetch_papers_by_ids(arxiv_ids: list[str]) -> list[RawPaper]:
    """Fetch a known list of arXiv papers by ID. Used for seed + citation-expanded papers."""
    BATCH_SIZE = 200
    all_papers = []
    for i in range(0, len(arxiv_ids), BATCH_SIZE):
        batch_ids = arxiv_ids[i : i + BATCH_SIZE]
        client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
        search = arxiv.Search(id_list=batch_ids)
        papers = []
        for result in tqdm(client.results(search), desc="Fetching by ID", total=len(batch_ids)):
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
        all_papers.extend(papers)
    return all_papers


def fetch_papers(
    categories: list[str],
    start_date: date,
    end_date: date,
    max_results: int = 1000,
) -> list[RawPaper]:

    # " OR ".join(...) → "cat:cs.LG OR cat:cs.AI OR cat:cs.CL"
    # query = " OR ".join(f"cat:{c}" for c in categories)

    date_filter = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000 TO {end_date.strftime('%Y%m%d')}235959]"
    query = f"({' OR '.join(f'cat:{c}' for c in categories)}) AND {date_filter}"
    
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
