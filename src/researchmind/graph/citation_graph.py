import json
import logging
import pickle
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from researchmind.ingestion.discovery import CitationSource
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)

project_root = find_project_root()


def build_graph(corpus_ids: list[str], source: CitationSource) -> nx.DiGraph:
    """Build a directed citation graph from the given list of paper IDs.

    Fetches outbound citations (papers cited by each paper) from the provided
    CitationSource and filters to only edges within the corpus. Papers outside
    the corpus are ignored — we only care about intra-corpus citation links.

    Args:
        corpus_ids: arXiv IDs of papers in the corpus (no version suffix, e.g. "1706.03762").
        source: CitationSource implementation that resolves IDs to cited references.

    Returns:
        Directed graph where edge A → B means paper A cites paper B.
    """
    # Precompute as a set once — used for O(1) membership checks inside the loop
    distinct_corpus = set(corpus_ids)
    citation_map: dict[str, list[str]] = {}

    for arxiv_id in tqdm(
        corpus_ids,
        desc="Building citation graph",
        total=len(corpus_ids),
        unit="paper",
        smoothing=0.1,
    ):
        try:
            reference_ids = source.get_referenced_ids(arxiv_id)
            if reference_ids is None:
                logger.warning("No citation data for %s — skipping.", arxiv_id)
                continue
            # Keep only references that exist in the corpus
            citation_map[arxiv_id] = [
                ref for ref in reference_ids
                if ref and ref in distinct_corpus and ref != arxiv_id
            ]
        except Exception as e:
            logger.error("Failed to fetch citations for %s: %s", arxiv_id, e)
            continue

    if not citation_map:
        logger.error("Citation map is empty — returning empty graph.")
        return nx.DiGraph()

    graph = nx.from_dict_of_lists(citation_map, create_using=nx.DiGraph)
    logger.info(
        "Citation graph built: %d nodes, %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_neighbors(
    graph: nx.DiGraph, paper_id: str, direction: str, depth: int
) -> list[str]:
    """Return neighbouring paper IDs up to `depth` hops away.

    Args:
        graph: The citation graph.
        paper_id: arXiv ID of the seed paper.
        direction: "outbound" — papers this paper cites; "inbound" — papers that cite this paper.
        depth: Maximum number of hops.

    Returns:
        List of neighbouring paper IDs (seed excluded).
    """
    if direction == "outbound":
        neighbors = nx.single_source_shortest_path_length(graph, paper_id, cutoff=depth)
    elif direction == "inbound":
        neighbors = nx.single_source_shortest_path_length(
            graph.reverse(), paper_id, cutoff=depth
        )
    else:
        raise ValueError(f"direction must be 'inbound' or 'outbound', got '{direction}'.")

    return [nid for nid in neighbors if nid != paper_id]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from researchmind.utils.config import load_phase_config
    from researchmind.ingestion.discovery import SemanticScholarCitationSource

    cfg = load_phase_config(project_root)
    papers_path = cfg.ingestion.parsed_papers_path

    logger.info("Loading corpus paper IDs from %s", papers_path)
    with open(papers_path, "r", encoding="utf-8") as f:
        papers = [json.loads(line) for line in f if line.strip()]

    # ParsedPaper structure: {"paper": {"paper_id": ..., ...}, "sections": {...}}
    # Strip version suffix so "2401.12345v2" → "2401.12345"
    arxiv_ids = [p["paper"]["paper_id"].split("v")[0] for p in papers]
    logger.info("Loaded %d paper IDs from phase=%s", len(arxiv_ids), cfg.name)

    graph_path = cfg.index.graph_path
    logger.info("Building citation graph → %s", graph_path)
    graph = build_graph(arxiv_ids, source=SemanticScholarCitationSource())

    save_graph(graph, graph_path)
    logger.info("Citation graph saved: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges())


if __name__ == "__main__":
    main()
