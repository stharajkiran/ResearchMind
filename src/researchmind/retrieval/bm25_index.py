import bm25s
import pickle
from pathlib import Path


class BM25IndexBuilder:
    def __init__(self, artifact_dir: str = "artifacts/indexes"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = None
        self.corpus_ids = None

    def build_index(self, corpus_texts: list[str], corpus_ids: list[str]) -> None:
        """Builds the BM25 index from the given corpus texts and their corresponding IDs."""
        self.corpus_ids = corpus_ids
        tokenized_corpus = bm25s.tokenize(corpus_texts)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokenized_corpus)
        # Save the ID map for later retrieval
        self._save_index()


    def search(self, query: str, k: int = 10) -> list[str]:
        """
        Returns ranked paper IDs for a given query.
        """
        tokenized_query = bm25s.tokenize([query])

        # Retrieve
        results, _ = self.retriever.retrieve(tokenized_query, k=k)
        
        # results contains indices, map them back to strings
        return [self.corpus_ids[i] for i in results[0]]

    def _save_index(self):
        """Save the index to disk. Save the corpus ids as pickle"""
        self.retriever.save(self.artifact_dir / "bm25_index.json") 
        with open(self.artifact_dir / "bm25_id_map.pkl", "wb") as f:
            pickle.dump(self.corpus_ids, f)

    def load_index(self, path: str):
        """Load the index from disk."""
        self.retriever = bm25s.BM25.load(path)
        with open(self.artifact_dir / "bm25_id_map.pkl", "rb") as f:
            self.id_map = pickle.load(f)