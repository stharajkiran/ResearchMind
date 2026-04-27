from pathlib import Path
import pickle
import networkx as nx
from researchmind.ingestion.openalex_client import get_referenced_arxiv_ids
import logging
from tqdm import tqdm
import json
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)


def build_graph(corpus_ids: list[str]) -> nx.DiGraph:
    """Build  a directed citation graph from the given list of paper IDs.

    Args:
        corpus_ids (list[str]): arxiv IDs of papers in the corpus (without version suffix, e.g. "1706.03762")

    Returns:
        nx.DiGraph: A directed graph where nodes are paper IDs and edges represent citations (A → B means A cites B).
    """
    # Dict[str, list[str]] mapping paper_id to list of cited paper_ids (outbound edges)
    citation_graph = {}
    for arxiv_id in tqdm(
        corpus_ids,
        desc="Building citation graph",
        total=len(corpus_ids),
        unit="paper",
        unit_scale=True,
        smoothing=0.1,
    ):
        try:
            # papers cited by this paper (outbound edges)
            reference_arxiv_ids = get_referenced_arxiv_ids(arxiv_id)
            if reference_arxiv_ids is None:
                logger.warning("No citation data returned for %s. Skipping.", arxiv_id)
                continue

            # Filter out references that are not in the corpus (e.g. non-arXiv papers or arXiv papers outside our dataset)
            distinct_corpus = set(corpus_ids)
            clean_references = [
                ref for ref in reference_arxiv_ids if ref and ref in distinct_corpus and ref != arxiv_id
            ]
            citation_graph[arxiv_id] = clean_references
        except Exception as e:
            logger.error("Failed to fetch citations for %s: %s", arxiv_id, e)
            continue

    # Final safety check before graph construction
    if not citation_graph:
        logger.error("Citation graph is empty. Returning empty graph.")
        return nx.DiGraph()

    logger.info("Constructed a directed graph")
    return nx.from_dict_of_lists(citation_graph, create_using=nx.DiGraph)


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    # Save
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        G_loaded = pickle.load(f)
    return G_loaded


def get_neighbors(
    graph: nx.DiGraph, paper_id: str, direction: str, depth: int
) -> list[str]:
    """Get neighboring papers in the citation graph.

    Args:
        graph (nx.DiGraph): The citation graph.
        paper_id (str): The arXiv ID of the paper to query.
        direction (str): "inbound" for papers that cite this paper, "outbound" for papers this paper cites.
        depth (int): How many hops away to retrieve neighbors.

    Returns:
        list[str]: List of neighboring paper IDs up to the specified depth.
    """
    if direction == "outbound":
        neighbors = nx.single_source_shortest_path_length(graph, paper_id, cutoff=depth)
    elif direction == "inbound":
        neighbors = nx.single_source_shortest_path_length(
            graph.reverse(), paper_id, cutoff=depth
        )
    else:
        raise ValueError("Direction must be 'inbound' or 'outbound'.")

    # Exclude the original paper_id and return only neighbors
    return [nid for nid in neighbors if nid != paper_id]


def main():
    # get the arxiv ids
    project_root = find_project_root()
    # read the papers.jsonl
    papers_path = project_root / "data" / "processed" / "parsed_papers.jsonl"

    logger.info("Loading corpus paper IDs from %s", papers_path)
    with open(papers_path, "r") as f:
        papers = [json.loads(line) for line in f]
    arxiv_ids = [p["paper"]["paper_id"] for p in papers]
    arxiv_ids = [
        aid.split("v")[0] for aid in arxiv_ids
    ]  # normalise by removing version suffix

    graph_output_path = project_root / "artifacts" / "citation_graph.pkl"
    graph_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building citation graph for %d papers...", len(arxiv_ids))
    citation_graph = build_graph(arxiv_ids)

    logger.info("Saving citation graph to %s", graph_output_path)
    save_graph(citation_graph, graph_output_path)
    logger.info("Citation graph saved successfully.")


if __name__ == "__main__":
    main()
