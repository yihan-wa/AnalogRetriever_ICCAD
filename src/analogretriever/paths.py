"""Repository-relative locations of data splits, checkpoints and results."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_root():
    if not (REPOSITORY_ROOT / "pyproject.toml").is_file():
        raise RuntimeError("Run from a source checkout installed with `pip install -e .`.")
    return str(REPOSITORY_ROOT)
