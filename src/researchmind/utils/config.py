import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from researchmind.utils.find_root import find_project_root


@dataclass
class LLMTierConfig:
    model: str
    provider: str


@dataclass
class PhaseConfig:
    phase: str
    chunks_path: Path
    artifact_dir: Path
    index_type: str
    vector_backend: str
    embedding_model: str
    llm_tiers: dict[str, LLMTierConfig]
    corpus_categories: list[str]
    corpus_date_from: str
    corpus_date_to: str
    corpus_max_results: int


def load_phase_config(project_root: Path | None = None) -> PhaseConfig:
    root = project_root or find_project_root()
    phase = os.environ.get("INDEX_PHASE", "phase2")
    config_path = root / "configs" / f"{phase}.yaml"

    if not config_path.exists():
        available = [p.stem for p in (root / "configs").glob("*.yaml")]
        raise FileNotFoundError(
            f"No config found for INDEX_PHASE='{phase}'. "
            f"Available: {available}. Add configs/{phase}.yaml to create a new domain."
        )

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    llm_tiers = {
        tier: LLMTierConfig(model=v["model"], provider=v["provider"])
        for tier, v in raw["llm"].items()
    }

    return PhaseConfig(
        phase=raw["phase"],
        chunks_path=root / raw["paths"]["chunks"],
        artifact_dir=root / raw["paths"]["artifact_dir"],
        index_type=raw["index"]["type"],
        vector_backend=raw["index"]["vector_backend"],
        embedding_model=raw["model"]["embedding"],
        llm_tiers=llm_tiers,
        corpus_categories=raw["corpus"]["categories"],
        corpus_date_from=raw["corpus"]["date_from"],
        corpus_date_to=raw["corpus"]["date_to"],
        corpus_max_results=raw["corpus"]["max_results"],
    )
