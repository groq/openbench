"""Utilities for locating cached FActScore assets.

The FActScore benchmark expects a large Wikipedia SQLite database and prompt
entity lists that are distributed with the official project.
To run factscore these assets need to be staged under ``~/.openbench/factscore`` so
multiple runs can share the same assets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


FACTSCORE_CACHE_ENV = "OPENBENCH_FACTSCORE_CACHE"
DEFAULT_FACTSCORE_CACHE = Path("~/.openbench/factscore").expanduser()


class FactScoreResourceError(FileNotFoundError):
    """Raised when required FActScore resources are missing."""


def resolve_cache_root(cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the cache root for FActScore assets, creating it if required."""

    if cache_root is None:
        cache_root = os.getenv(FACTSCORE_CACHE_ENV)

    path = (
        Path(cache_root).expanduser().resolve()
        if cache_root
        else DEFAULT_FACTSCORE_CACHE
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory that should contain downloaded FActScore data."""

    root = resolve_cache_root(cache_root)
    path = (root / "data").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory that stores local FActScore model artefacts."""

    root = resolve_cache_root(cache_root)
    path = (root / "models").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory used for runtime caches (retrieval, API, etc.)."""

    root = resolve_cache_root(cache_root)
    path = (root / "cache").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_resources_exist(required_files: Iterable[Path]) -> None:
    """Validate that all required paths exist, raising a helpful error."""

    missing = [path for path in required_files if not path.exists()]
    if not missing:
        return

    missing_str = "\n  - ".join(str(path) for path in missing)
    raise FactScoreResourceError(
        "Missing FactScore resources. download the assets here: \n"
        " https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I\n"
        "Use bench cache upload to upload the assets to the respective cache directory.\n"
        "The following paths were not found:\n  - "
        f"{missing_str}"
    )


def knowledge_db_path(
    cache_root: str | os.PathLike[str] | None = None,
    filename: str = "enwiki-20230401.db",
) -> Path:
    """Return the path to the expected Wikipedia SQLite database."""

    path = data_dir(cache_root) / filename
    ensure_resources_exist([path])
    return path


def prompt_entities_path(
    split: str,
    cache_root: str | os.PathLike[str] | None = None,
    filename: str = "prompt_entities.txt",
) -> Path:
    """Return the path to the prompt entities list for a given split."""

    path = data_dir(cache_root) / split / filename
    ensure_resources_exist([path])
    return path


def _required_files(root: Path) -> List[Path]:
    base_data = data_dir(root)
    return [
        base_data / "enwiki-20230401.db",
        base_data / "labeled" / "prompt_entities.txt",
        base_data / "unlabeled" / "prompt_entities.txt",
    ]
