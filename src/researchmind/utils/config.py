import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from researchmind.utils.find_root import find_project_root
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMTierConfig:
    model: str
    provider: str


@dataclass
class CorpusConfig:
    categories: list[str]
    keywords: list[str]
    date_from: str
    date_to: str
    max_results: int


@dataclass
class DiscoveryConfig:
    sources: list[str]  # e.g. ["arxiv", "ss_search", "ss_recommendations"]
    ss_search_max_per_query: int = 200
    ss_rec_seed_ids: list[str] = None  # populated from seeds.yaml at load time
    ss_rec_max_recommendations: int = 500

    def __post_init__(self):
        if self.ss_rec_seed_ids is None:
            self.ss_rec_seed_ids = []


@dataclass
class IngestionConfig:
    # paths
    pdf_dir: Path
    papers_path: Path
    parsed_papers_path: Path
    raw_chunks_path: Path
    # discovery
    discovery: DiscoveryConfig = None
    # download
    download_concurrency: int = 10
    download_timeout_seconds: int = 30
    # chunking
    chunk_strategy: str = "section"  # "section" | "fixed" | "semantic"
    chunk_max_words: int = 400
    chunk_overlap_words: int = 50

    def __post_init__(self):
        if self.discovery is None:
            self.discovery = DiscoveryConfig(sources=["arxiv"])


@dataclass
class IndexConfig:
    chunks_path: Path
    artifact_dir: Path
    index_type: str
    vector_backend: str


@dataclass
class ModelConfig:
    embedding: str
    llm_tiers: dict[str, LLMTierConfig]


@dataclass
class EvaluationConfig:
    test_set_path: Path
    ragas_responses_path: Path
    log_dir: Path
    batch_size: int
    experiment_name: str


@dataclass
class PhaseConfig:
    name: str
    corpus: CorpusConfig
    ingestion: IngestionConfig
    index: IndexConfig
    model: ModelConfig
    evaluation: EvaluationConfig


def load_phase_config(project_root: Path | None = None) -> PhaseConfig:
    root = project_root or find_project_root()
    phase = os.environ.get("CONFIG_NAME", "phase2")
    config_path = root / "configs" / f"{phase}.yaml"

    if not config_path.exists():
        available = [p.stem for p in (root / "configs").glob("*.yaml")]
        raise FileNotFoundError(
            f"No config found for CONFIG_NAME='{phase}'. "
            f"Available: {available}. Add configs/{phase}.yaml to create a new domain."
        )

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    corpus_raw = raw["corpus"]
    corpus = CorpusConfig(
        categories=corpus_raw["categories"],
        keywords=corpus_raw.get("keywords", []),
        date_from=corpus_raw["date_from"],
        date_to=corpus_raw["date_to"],
        max_results=corpus_raw["max_results"],
    )

    # Load seeds.yaml once — shared across all phases
    seeds_path = root / "configs" / "seeds.yaml"
    all_seeds: dict[str, list[str]] = {}
    if seeds_path.exists():
        with seeds_path.open() as sf:
            all_seeds = yaml.safe_load(sf) or {}

    ing_raw = raw.get("ingestion", {})
    ing_paths = ing_raw.get("paths", {})
    dl = ing_raw.get("download", {})
    ck = ing_raw.get("chunking", {})
    disc_raw = ing_raw.get("discovery", {})
    rec_raw = disc_raw.get("ss_recommendations", {})
    seed_set = rec_raw.get("seed_set")
    seed_ids = all_seeds.get(seed_set, []) if seed_set else []

    discovery = DiscoveryConfig(
        sources=disc_raw.get("sources", ["arxiv"]),
        ss_search_max_per_query=disc_raw.get("ss_search", {}).get("max_per_query", 200),
        ss_rec_seed_ids=seed_ids,
        ss_rec_max_recommendations=rec_raw.get("max_recommendations", 500),
    )

    ingestion = IngestionConfig(
        pdf_dir=root / ing_paths["pdf_dir"],
        papers_path=root / ing_paths["papers"],
        parsed_papers_path=root / ing_paths["parsed_papers"],
        raw_chunks_path=root / ing_paths["raw_chunks"],
        discovery=discovery,
        download_concurrency=dl.get("concurrency", 10),
        download_timeout_seconds=dl.get("timeout_seconds", 30),
        chunk_strategy=ck.get("strategy", "section"),
        chunk_max_words=ck.get("max_words", 400),
        chunk_overlap_words=ck.get("overlap_words", 50),
    )

    idx_raw = raw["index"]
    index = IndexConfig(
        chunks_path=root / idx_raw["chunks"],
        artifact_dir=root / idx_raw["artifact_dir"],
        index_type=idx_raw["type"],
        vector_backend=idx_raw["vector_backend"],
    )

    model_raw = raw["model"]
    llm_tiers = {
        tier: LLMTierConfig(model=v["model"], provider=v["provider"])
        for tier, v in model_raw["llm"].items()
    }
    model = ModelConfig(
        embedding=model_raw["embedding"],
        llm_tiers=llm_tiers,
    )

    ev_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        test_set_path=root / ev_raw.get("test_set", "data/processed/test_queries.json"),
        ragas_responses_path=root
        / ev_raw.get("ragas_responses", "data/processed/ragas_responses.json"),
        log_dir=root / ev_raw.get("log_dir", "logs/ragas_evaluation"),
        batch_size=ev_raw.get("batch_size", 10),
        experiment_name=ev_raw.get("experiment_name", "RAGAS_Evaluation"),
    )

    return PhaseConfig(
        name=raw["name"],
        corpus=corpus,
        ingestion=ingestion,
        index=index,
        model=model,
        evaluation=evaluation,
    )
