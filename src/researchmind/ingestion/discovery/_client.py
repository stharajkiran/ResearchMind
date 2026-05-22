"""Shared Semantic Scholar HTTP utilities."""

import os

_BASE = "https://api.semanticscholar.org/graph/v1"
_RECOMMEND_BASE = "https://api.semanticscholar.org/recommendations/v1"
_BATCH_SIZE = 500  # SS /paper/batch hard limit


def headers() -> dict[str, str]:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return {"x-api-key": key} if key else {}
