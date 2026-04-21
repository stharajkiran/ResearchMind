import faiss
import numpy as np
import pickle
from pathlib import Path

from typing import Literal

IndexType = Literal["Flat", "IVF100", "HNSW32"]

class FaissIndexBuilder:
    def __init__(self, dimension: int, artifact_dir: str = "artifacts/indexes"):
        self.dimension = dimension
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.id_map = None # Will store list of paper IDs
        self.index = None

    def build_index(self, embeddings: np.ndarray, paper_ids: list[str], index_type: IndexType = "Flat") -> None:
        # Store ID mapping
        self.id_map = paper_ids
        n_vectors = embeddings.shape[0]
        
        # Convert IDs to int64 for FAISS
        ids = np.arange(n_vectors, dtype=np.int64)
        index_type = index_type.lower()
        
        if index_type == "flat":
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
            
        elif index_type == "ivf100":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            # IVF requires training
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index = faiss.IndexIDMap(self.index)
            self.index.train(embeddings.astype('float32'))
            
        elif index_type == "hnsw32":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index = faiss.IndexIDMap(self.index)
            
        self.index.add_with_ids(embeddings.astype('float32'), ids)
        
        # Save both index and id_map
        self._save_index(index_type)

    def _save_index(self, index_type: IndexType):
        faiss.write_index(self.index, str(self.artifact_dir / f"{index_type}.index"))
        with open(self.artifact_dir / f"{index_type}_ids.pkl", "wb") as f:
            pickle.dump(self.id_map, f)

    def load_index(self, index_type: IndexType) -> None:
        """Load the index and ID map from disk.
        
        Args:
            index_type (IndexType): The type of index to load ("flat", "ivf100", "hnsw32").

        """
        index_type = index_type.lower()
        try:
            self.index = faiss.read_index(str(self.artifact_dir / f"{index_type}.index"))
            with open(self.artifact_dir / f"{index_type}_ids.pkl", "rb") as f:
                self.id_map = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Index file for {index_type} not found in {self.artifact_dir}. Please build the index first.")

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[str]:
        # Perform search
        distances, indices = self.index.search(query_vec.astype('float32'), k)
        
        # Map back to paper IDs
        return [self.id_map[i] for i in indices[0] if i != -1]