import pickle
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

from researchmind.retrieval.interfaces import DenseIndex

IndexType = Literal["Flat", "IVF100", "HNSW32"]


class FaissIndexBuilder(DenseIndex):
    def __init__(
        self,
        dimension: int,
        artifact_dir: str = "artifacts/indexes",
        index_type: IndexType = "HNSW32",
    ):
        self.dimension = dimension
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.index_type = index_type.lower()
        self.id_map = None
        self.index = None

    # ── DenseIndex interface ──────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        self.build_index(embeddings, ids, index_type=self.index_type)

    def load(self) -> None:
        self.load_index(self.index_type)

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[str]:
        distances, indices = self.index.search(query_vec.astype("float32"), k)
        return [self.id_map[i] for i in indices[0] if i != -1]

    # ── FAISS-specific methods (used by build scripts) ────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        paper_ids: list[str],
        index_type: IndexType = "Flat",
    ) -> None:
        self.id_map = paper_ids
        n_vectors = embeddings.shape[0]
        ids = np.arange(n_vectors, dtype=np.int64)
        index_type = index_type.lower()

        if index_type == "flat":
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
        elif index_type == "ivf100":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index = faiss.IndexIDMap(self.index)
            self.index.train(embeddings.astype("float32"))
        elif index_type == "hnsw32":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index = faiss.IndexIDMap(self.index)

        self.index.add_with_ids(embeddings.astype("float32"), ids)
        self._save_index(index_type)

    def load_index(self, index_type: IndexType) -> None:
        index_type = index_type.lower()
        try:
            self.index = faiss.read_index(
                str(self.artifact_dir / f"{index_type}.index")
            )
            with open(self.artifact_dir / f"{index_type}_ids.pkl", "rb") as f:
                self.id_map = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Index file for '{index_type}' not found in {self.artifact_dir}. "
                "Run 'make indexes' to build it first."
            )

    def _save_index(self, index_type: IndexType) -> None:
        faiss.write_index(self.index, str(self.artifact_dir / f"{index_type}.index"))
        with open(self.artifact_dir / f"{index_type}_ids.pkl", "wb") as f:
            pickle.dump(self.id_map, f)
