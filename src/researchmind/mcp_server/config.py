from datetime import datetime
import os

from researchmind.utils.find_root import find_project_root

project_root = find_project_root()


class Config:
    "Configuration for API, including logging and any other global setup."

    project_root = project_root
    phase = os.environ.get("INDEX_PHASE", "phase2")
    artifact_dir = project_root / "artifacts" / "indexes" / phase
    if phase == "semantic":
        chunks_path = (
            project_root / "data" / "processed" / "cleaned_semantic_chunks.jsonl"
        )
    else:
        chunks_path = project_root / "data" / "processed" / "cleaned_chunks.jsonl"

    # model_name = "claude-sonnet-4-6"
    model_name = "qwen3.6:27b"
    collection_name = "researchmind"
