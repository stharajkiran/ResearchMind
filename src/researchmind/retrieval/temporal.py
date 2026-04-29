from datetime import datetime

def apply_recency_decay(chunk_ids: list[str], chunk_dict: dict[str,dict], decay_factor: float)->list[str]:
    """Applies a recency decay to the original scores of chunks based on their rank in the list.

    Args:
        chunk_ids (list[str]): A list of chunk IDs sorted by rank (e.g., relevance score) after rrf fusion.
        chunk_dict (dict[str,dict]): chunk id to chunk metadata dict, where each chunk's metadata 
        decay_factor (float): The factor by which to decay scores for each subsequent rank (e.g., 0.9 means each rank gets 90% of the previous rank's score).

    Returns:
        list[str]: Returns resorted list of chunk IDs after applying recency decay to their scores.
    """
    current_year = datetime.now().year
    decayed_scores = {}
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        # get the year of the chunk, default to current year if not available
        chunk_year = chunk_dict[chunk_id].get("year", current_year)
        # apply decay based on rank and age of the chunk
        decayed_score = (1 / rank) * decay_factor ** (current_year - chunk_year)
        # store the decayed score in dict
        decayed_scores[chunk_id] = decayed_score
    # Sort chunk IDs by decayed score in descending order
    sorted_chunk_ids = sorted(decayed_scores, key=decayed_scores.get, reverse=True)
    return sorted_chunk_ids