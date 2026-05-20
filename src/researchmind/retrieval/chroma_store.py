import json
import logging
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from tqdm import tqdm

from researchmind.embedding.models import BaseResearchEncoder
from researchmind.ingestion.models import Chunk
from researchmind.retrieval.interfaces import FilteredStore
from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)


class MPNetEmbeddingFunction(EmbeddingFunction):
    def __init__(self, encoder: BaseResearchEncoder):
        self.encoder = encoder

    def __call__(self, input: list[str]) -> Embeddings:
        return self.encoder.encode(input).tolist()


class ChromaStore(FilteredStore):
    def __init__(self, collection_name: str, encoder: BaseResearchEncoder | None = None):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(find_project_root() / "data" / "chroma_db")
        )
        self.corpus_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_corpus",
            embedding_function=MPNetEmbeddingFunction(encoder=encoder),
        )

    # ── FilteredStore interface ───────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> None:
        self.insert_chunks(chunks)

    def query(self, query: str, k: int = 10, filters: dict | None = None) -> list[Chunk]:
        return self.query_collection(query, n_results=k, where=filters)

    # ── Chroma-specific methods ───────────────────────────────────────────────

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        BATCH_SIZE = 4000
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(
            "Inserting %d chunks into Chroma in %d batches...",
            len(chunks),
            total_batches,
        )
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), total=total_batches, desc="Inserting chunks"):
            self._insert_batch(chunks[i : i + BATCH_SIZE])

    def _insert_batch(self, chunks: list[Chunk]) -> None:
        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "paper_id": chunk.paper_id,
                    "section": chunk.section,
                    "title": chunk.title,
                    "year": chunk.year,
                    "authors": ", ".join(chunk.authors),
                }
            )
        self.corpus_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query_collection(
        self, query: str, n_results: int = 10, where: dict | None = None
    ) -> list[Chunk]:
        kwargs = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self.corpus_collection.query(**kwargs)

        retrieved_chunks = []
        for i in range(len(results["documents"][0])):
            authors_text = results["metadatas"][0][i]["authors"]
            retrieved_chunks.append(
                Chunk(
                    chunk_id=results["ids"][0][i],
                    paper_id=results["metadatas"][0][i]["paper_id"],
                    section=results["metadatas"][0][i]["section"],
                    text=results["documents"][0][i],
                    year=results["metadatas"][0][i]["year"],
                    authors=authors_text.split(", ") if authors_text else [],
                    title=results["metadatas"][0][i]["title"],
                    page=None,
                )
            )
        return retrieved_chunks


if __name__ == "__main__":
    project_root = find_project_root()
    folder_dir = project_root / "data" / "processed"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with open(folder_dir / "cleaned_chunks.jsonl", "r", encoding="utf-8") as f:
        chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]

    with open(folder_dir / "200_test_queries_set.json", "r", encoding="utf-8") as f:
        queries_dict = json.load(f)

    logger.info("Initialized ChromaStore and connected to collection.")
    store = ChromaStore(collection_name="researchmind")

    if "researchmind_corpus" not in store.client.list_collections():
        store.insert_chunks(chunks)

    results = store.query_collection("transformer attention mechanism", n_results=5)
    output_path = Path("experiments/chromadb_queries")
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{store.collection_name}_query_results.json", "w") as f:
        json.dump([t.model_dump(mode="json") for t in results], f, indent=2)
