from researchmind.utils.find_root import find_project_root
from researchmind.utils.config import load_phase_config

project_root = find_project_root()
_cfg = load_phase_config(project_root)


class Config:
    project_root = project_root
    artifact_dir = _cfg.index.artifact_dir
    chunks_path = _cfg.index.chunks_path
    graph_path = _cfg.index.graph_path
    index_type = _cfg.index.index_type
    collection_name = "researchmind"
