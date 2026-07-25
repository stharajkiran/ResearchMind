from collections import defaultdict



def reciprocal_rank_fusion(
    faiss_results: list[str],
    bm25_results: list[str],
    k: int = 60,
    return_scores: bool = False,
):
    """Fuse two ranked ID lists via reciprocal rank fusion.

    Returns document IDs ordered by fused score (highest first). With
    ``return_scores=True``, returns ``(doc_id, score)`` pairs instead — same
    ordering — for callers that need the relevance weight (e.g. the search UI).
    """
    rrf_scores = defaultdict(float)

    # Process Vector Results
    for rank, doc_id in enumerate(faiss_results, 1):
        rrf_scores[doc_id] += 1 / (k + rank)

    # Process Keyword Results
    for rank, doc_id in enumerate(bm25_results, 1):
        rrf_scores[doc_id] += 1 / (k + rank)

    # Sort by combined score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if return_scores:
        return sorted_docs
    return [doc_id for doc_id, _ in sorted_docs]


