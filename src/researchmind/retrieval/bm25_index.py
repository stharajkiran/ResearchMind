import pickle
from pathlib import Path

import bm25s

from researchmind.retrieval.interfaces import SparseIndex


class BM25IndexBuilder(SparseIndex):
    def __init__(self, artifact_dir: str = "artifacts/indexes"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = None
        self.corpus_ids = None

    # ── SparseIndex interface ─────────────────────────────────────────────────

    def build(self, texts: list[str], ids: list[str]) -> None:
        self.build_index(texts, ids)

    def load(self) -> None:
        self.load_index()

    def search(self, query: str, k: int = 10) -> list[str]:
        tokenized_query = bm25s.tokenize([query])
        results, _ = self.retriever.retrieve(tokenized_query, k=k)
        return [self.corpus_ids[i] for i in results[0]]

    # ── BM25-specific methods (used by build scripts) ─────────────────────────

    def build_index(self, corpus_texts: list[str], corpus_ids: list[str]) -> None:
        self.corpus_ids = corpus_ids
        tokenized_corpus = bm25s.tokenize(corpus_texts)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokenized_corpus)
        self._save_index()

    def load_index(self) -> None:
        self.retriever = bm25s.BM25.load(self.artifact_dir / "bm25_index.json")
        with open(self.artifact_dir / "bm25_id_map.pkl", "rb") as f:
            self.corpus_ids = pickle.load(f)

    def _save_index(self) -> None:
        self.retriever.save(self.artifact_dir / "bm25_index.json")
        with open(self.artifact_dir / "bm25_id_map.pkl", "wb") as f:
            pickle.dump(self.corpus_ids, f)
