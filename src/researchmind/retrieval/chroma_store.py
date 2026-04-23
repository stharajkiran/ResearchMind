import chromadb
from pathlib import Path
import json
import logging

from tqdm import tqdm
from researchmind.ingestion.models import Chunk
from researchmind.embedding.models import MPNetEncoder
from researchmind.ingestion.models import Chunk
from chromadb import EmbeddingFunction, Embeddings

from researchmind.utils.find_root import find_project_root

logger = logging.getLogger(__name__)

class MPNetEmbeddingFunction(EmbeddingFunction):
    def __init__(self, encoder):
        self.encoder = encoder  # your existing MPNetEncoder from Phase 1

    def __call__(self, input: list[str]) -> Embeddings:
        return self.encoder.encode(input).tolist()

class ChromaStore:
    def __init__(self, collection_name: str, encoder: MPNetEncoder | None = None):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(find_project_root() / "data" / "chroma_db"))

        encoder = encoder
        self.corpus_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_corpus",
            embedding_function=MPNetEmbeddingFunction(encoder=encoder),
        )
        # self.user_collection = self.client.get_or_create_collection(
        #     name=f"{collection_name}_users",
        #     embedding_function=MPNetEmbeddingFunction(encoder=encoder),
        # )

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        # Inser in batches to avoid memory issues if the chunk list is very large
        BATCH_SIZE = 4000
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Inserting {len(chunks)} chunks into Chroma collection in {total_batches} batches...")
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), total=total_batches, desc="Inserting chunks"):
            batch_chunks = chunks[i : i + BATCH_SIZE]
            self._insert_batch(batch_chunks)
    
    def _insert_batch(self, chunks: list[Chunk]) -> None:
        # Convert list of Chunk objects to list of dicts for Chroma 
        documents = []
        ids = []
        metadatas = []
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
        self.corpus_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def query_collection(self, query: str, n_results: int = 10) -> list[Chunk]:
        # results is QueryResult type
        results = self.corpus_collection.query(query_texts=[query], n_results=n_results)

        # each query in the query_texts list will have a corresponding list of results in results["documents"]
        retrieved_chunks = []
        for i in range(len(results["documents"][0])):
            authors_text = results["metadatas"][0][i]["authors"]
            authors = authors_text.split(", ") if authors_text else []
            retrieved_chunks.append(
                Chunk(
                    chunk_id=results["ids"][0][i],
                    paper_id=results["metadatas"][0][i]["paper_id"],
                    section=results["metadatas"][0][i]["section"],
                    text=results["documents"][0][i],
                    year=results["metadatas"][0][i]["year"],
                    authors=authors,
                    title=results["metadatas"][0][i]["title"],
                    page=None,  # page info not stored in metadata, can be added if needed
                )
            )
        return retrieved_chunks
    
    

if __name__ == "__main__":
    project_root = find_project_root()
    folder_dir = project_root / "data" / "processed"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Loading chunks and queries from JSON files.")

    # load some chunks from a JSONL file and insert into Chroma
    with open(folder_dir / "cleaned_chunks.jsonl", "r", encoding="utf-8") as f:
        chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]

    # load queries
    with open(folder_dir / "200_test_queries_set.json", "r", encoding="utf-8") as f:
        queries_dict = json.load(f)

    query_chunk_ids = {q["chunk_id"] for q in queries_dict}


    print(f"Unique chunks matched: {len(chunks)}")

    logger.info("Initialized ChromaStore and connected to collection.")
    #    -------------------------------------------------------------------------------
    store = ChromaStore(collection_name="researchmind")

    # check if the collection exists before inserting
    if "researchmind_corpus" not in store.client.list_collections():
        logger.info("Collection does not exist. Creating new collection.")
        store.client.create_collection("researchmind")

        # add chunks to Chroma collection
        logger.info(f"Inserting {len(chunks)} chunks into Chroma collection.")
        store.insert_chunks(chunks)
        logger.info(f"Inserted {len(chunks)} chunks into Chroma collection.")

    logger.info("Testing query retrieval from Chroma collection.")
    logger.info(f"Query: 'transformer attention mechanism'")
    results = store.query_collection('transformer attention mechanism', n_results=5)

    output_path = Path(f"experiments/chromadb_queries")
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{store.collection_name}_query_results.json", "w") as f:
        json.dump([t.model_dump(mode="json") for t in results], f, indent=2)
    logger.info(f"Generated test set with {len(results)} queries.")