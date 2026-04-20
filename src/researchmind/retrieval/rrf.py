from collections import defaultdict



def reciprocal_rank_fusion(faiss_results: list[str], bm25_results: list[str], k: int = 60):
    rrf_scores = defaultdict(float)
    
    # Process Vector Results
    for rank, doc_id in enumerate(faiss_results, 1):
        rrf_scores[doc_id] += 1 / (k + rank)
        
    # Process Keyword Results
    for rank, doc_id in enumerate(bm25_results, 1):
        rrf_scores[doc_id] += 1 / (k + rank)

    # Sort by combined score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # return sorted document IDs based on RRF scores
    return [doc_id for doc_id, score in sorted_docs]


