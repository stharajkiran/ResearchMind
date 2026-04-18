import numpy as np
from sentence_transformers import SentenceTransformer


class BaseResearchEncoder:
    """Base class for research document encoders. Handles model loading and provides a standardized encode method."""

    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        # Automatically detect the dimension from the loaded model
        self.dim = self.model.get_embedding_dimension()
        self.model_name = model_name

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        """Standardized encoding method for all child classes."""
        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=False,
            normalize_embeddings=True,  # Best practice for RAG/Cosine Similarity
        )


class SPECTER2Encoder(BaseResearchEncoder):
    """The domain-adapted expert for scientific citations, with query/document specialization."""

    def __init__(self):
        # Load the document model via the base class (sets self.model, self.dim, self.model_name).
        super().__init__("allenai/specter2_base")
        # Load a separate query model so both are ready without adapter-swapping at encode time.



class MPNetEncoder(BaseResearchEncoder):
    """A strong general-purpose encoder that performs well across various research domains."""

    def __init__(self):
        super().__init__("sentence-transformers/all-mpnet-base-v2")


class BGEEncoder(BaseResearchEncoder):
    """A lightweight and efficient encoder from BAAI, suitable for large-scale applications with limited resources."""

    def __init__(self):
        super().__init__("BAAI/bge-small-en-v1.5")
