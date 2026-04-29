from openai import OpenAI
from  ollama import Client

# HyDE: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
# Original evaluation domain is arXiv/BEIR, which matches this corpus exactly.
_HYDE_PROMPT = """Write a hypothetical research paper abstract that would directly answer the following query.

The abstract must follow this structure:
1. Problem: what problem or gap does the paper address
2. Method: what approach or technique is proposed
3. Experiments: what datasets or benchmarks were used
4. Results: what quantitative improvements were achieved

Write in the style of an arXiv preprint in computer vision, anomaly detection, or out-of-distribution detection.
Use precise technical language. Include specific metric names (AUROC, FPR95, AUPRC) and dataset names (MVTec, CIFAR-100, ImageNet) where appropriate.

Return only the abstract text — no title, no author line, no preamble.

Query: {query}"""

_REWRITE_PROMPT = """You are a search query optimizer for academic literature retrieval.

Rewrite the query below following these rules:
1. Expand abbreviations and acronyms only (e.g. "RL" -> "reinforcement learning")
2. Make implicit concepts explicit using only terms present or directly implied in the original
3. Do not add new concepts, methods, or research directions not mentioned in the original query

Return only the rewritten query. If the query needs no changes, return it unchanged.

Query: {query}"""


class QueryTransformer:
    """LLM-based query transformations applied before dense embedding.

    Two strategies:
    - rewrite: expands shorthand and adds related terms (improves technical-term recall)
    - hyde: generates a hypothetical abstract and embeds that instead of the raw query
      (Gao et al. 2022, original arXiv evaluation domain)
    """

    def __init__(self, client:Client, model: str = None):
        self._client = client
        self._model = model

    def set_model(self, model: str):
        """Dynamically update the model used for transformations."""
        self._model = model

    def _call(self, prompt: str) -> str:
        # use Ollama client
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options={
                'num_predict': 512, 
                "num_ctx": 1024
                },
            think=False,
            keep_alive=-1,  # keep loaded for 30 minutes
        )
        return response.message.content.strip()

    def rewrite(self, query: str) -> str:
        return self._call(_REWRITE_PROMPT.format(query=query))

    def hyde(self, query: str) -> str:
        """Return a hypothetical abstract that would answer the query (Gao et al. 2022)."""
        return self._call(_HYDE_PROMPT.format(query=query))