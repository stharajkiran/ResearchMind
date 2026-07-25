from mcp.server.fastmcp import FastMCP

from typing import Literal

from researchmind.mcp_server.config import Config
from researchmind.ingestion.models import ResearchGapResponse
from researchmind.utils.build_prompt import build_gap_prompt
from researchmind.utils.llm_client import ResearchMindLLM
from researchmind.retrieval.retriever import RetrieverService
from researchmind.retrieval.faiss_index import FaissIndexBuilder
from researchmind.retrieval.bm25_index import BM25IndexBuilder
from researchmind.retrieval.chroma_store import ChromaStore
from researchmind.embedding.models import MPNetEncoder
from researchmind.graph.citation_graph import get_neighbors, load_graph

mcp = FastMCP("ResearchMind", port=8001)

encoder = MPNetEncoder()
dense = FaissIndexBuilder(
    dimension=encoder.dim,
    artifact_dir=Config.artifact_dir,
    index_type=Config.index_type,
)
dense.load()
sparse = BM25IndexBuilder(artifact_dir=Config.artifact_dir)
sparse.load()
filtered = ChromaStore(collection_name=Config.collection_name, encoder=encoder)

retriever = RetrieverService(
    dense=dense,
    sparse=sparse,
    filtered=filtered,
    encoder=encoder,
    chunks_path=Config.chunks_path,
)

llm = ResearchMindLLM()
citation_graph = load_graph(Config.graph_path)


@mcp.tool()
def search_research(query: str, recency_days: int = 0) -> list[dict]:
    # convert recency_days to a decay rate for the retriever
    recency_decay_rate = None if recency_days == 0 else 1 / recency_days
    results = retriever.search(query, k=10, recency_decay_rate=recency_decay_rate)
    # convert results to dicts for JSON serialization
    return [r.model_dump() for r in results]


@mcp.tool()
def get_citations(
    paper_id: str, direction: Literal["inbound", "outbound", "both"] = "outbound"
) -> list[dict]:
    # Get neighboring papers in the citation graph
    if direction == "both":
        neighbors = get_neighbors(
            citation_graph, paper_id, "inbound", depth=2
        ) + get_neighbors(citation_graph, paper_id, "outbound", depth=2)
    else:
        neighbors = get_neighbors(citation_graph, paper_id, direction, depth=2)
    # Get chunks for the neighboring papers
    chunks = retriever.get_chunks_for_papers(neighbors)
    # return chunks as dicts for JSON serialization
    return [c.model_dump() for c in chunks]


@mcp.tool()
def detect_gaps(query: str) -> dict:
    retrieved_chunks = retriever.search(query, k=20)
    SYSTEM_PROMPT, user_prompt = build_gap_prompt(query, retrieved_chunks)

    response = llm.complete_structured(
        user_prompt=user_prompt,
        response_model=ResearchGapResponse,
        system_prompt=SYSTEM_PROMPT,
        tier="best",
        max_tokens=2048,
        temperature=0.0,
    )

    return response.model_dump()


@mcp.tool()
def validate_claim(claim: str, paper_ids: list[str]) -> dict:
    chunks = retriever.get_chunks_for_papers(paper_ids)
    # check if claim is in the text of the retrieved chunks
    literal_match = any(c for c in chunks if claim in c.text)
    # encode claim and chunk texts and calculate semantic similarity
    encoder = retriever.encoder
    # (dim, )
    encoded_claim = encoder.encode(claim, normalize_embeddings=True)
    # (num_chunks, dim)
    chunk_embeddings = encoder.encode(
        [c.text for c in chunks], normalize_embeddings=True
    )
    scores = chunk_embeddings @ encoded_claim
    max_score = float(scores.max()) if len(chunks) > 0 else 0.0
    # check if either claim passes
    supported = literal_match or max_score > 0.6
    return {
        "literal_match": literal_match,
        "semantic_score": max_score,
        "supported": supported,
    }

@mcp.tool()
def ingest_paper(paper_url: str) -> dict:
    """Ingest a new paper into the corpus by arXiv URL."""
    from worker.tasks import ingest_paper_task
    # Trigger the asynchronous ingestion task and return the task ID for tracking
    task = ingest_paper_task.delay(paper_url)
    return {"task_id": task.id, "status": "queued"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
