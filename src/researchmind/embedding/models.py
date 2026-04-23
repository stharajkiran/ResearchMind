import numpy as np
from sentence_transformers import SentenceTransformer
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

import torch
import numpy as np

class BaseResearchEncoder:
    """Base class for research document encoders. Handles model loading and provides a standardized encode method."""

    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        # Automatically detect the dimension from the loaded model
        self.dim = self.model.get_embedding_dimension()
        self.model_name = model_name

    def encode(self, texts: list[str], convert_to_numpy: bool = True, batch_size: int = 32) -> np.ndarray:
        """Standardized encoding method for all child classes."""
        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=True,
            normalize_embeddings=True,  # Best practice for RAG/Cosine Similarity
            batch_size=batch_size,
        )


class MPNetEncoder(BaseResearchEncoder):
    """A strong general-purpose encoder that performs well across various research domains."""

    def __init__(self):
        super().__init__("sentence-transformers/all-mpnet-base-v2")


class BGEEncoder(BaseResearchEncoder):
    """A lightweight and efficient encoder from BAAI, suitable for large-scale applications with limited resources."""

    def __init__(self):
        super().__init__("BAAI/bge-small-en-v1.5")


class SPECTER2Encoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "allenai/specter2_base"
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        # Encoding corpus documents — use proximity
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="proximity", set_active=True
        )

        # Encoding user queries — use adhoc_query
        self.model.load_adapter(
            "allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query"
        )
        self.model.set_active_adapters("adhoc_query")
        self.model.to(self.device)
        self.model.eval()

    def encode_corpus(self, papers: list[dict], batch_size: int = 32) -> np.ndarray:
        self.model.set_active_adapters("proximity")
        # Documents use title + sep + abstract
        texts = [p["title"] + self.tokenizer.sep_token + p["abstract"] for p in papers]
        return self._encode(texts, batch_size)

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        self.model.set_active_adapters("adhoc_query")
        # Queries are passed as-is — no title prefix
        return self._encode(queries, batch_size)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            # Normalize to unit length — matches SentenceTransformer behavior
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)
