from pathlib import Path
from typing import Optional


def find_project_root(sentinel: str = "pyproject.toml") -> Path:
    """
    Search upwards from the current file until the sentinel file is found.
    This guarantees we hit the root regardless of directory nesting.
    """
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if (parent / sentinel).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root containing {sentinel}")
