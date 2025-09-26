"""
Preparation utilities to prefetch and precompute Copilot caches.

"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional
import json as _json

from .server import _user_cache_dir, _ensure_parent_dir, _generate_embeddings_file
from .upstream_cache import (
    get_clean_config_cached,
    get_tools_json_cached,
    get_annotated_data_cached,
)
import shutil


def _default_embeddings_path() -> Path:
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    abstract_model = os.getenv("ABSTRACT_MODEL", "gpt-4.1-2025-04-14")
    return (
        _user_cache_dir()
        / "config"
        / f"mcp_arg_{embedding_model}_{abstract_model}.json"
    )


def prepare_copilot_cache(
    force_refresh: bool = False, embeddings_path: Optional[Path] = None
) -> Path:
    """Prefetch upstream JSONs and generate the embeddings file.

    Args:
        force_refresh: If True, refetch upstream JSONs (overrides cached copy)
        embeddings_path: Optional output path; defaults to user cache path.

    Returns:
        Path to the generated embeddings JSON.
    """
    # Ensure upstream JSONs are cached
    get_clean_config_cached(force_refresh)
    get_tools_json_cached(force_refresh)

    # Prepare embeddings path
    out = embeddings_path or _default_embeddings_path()
    _ensure_parent_dir(out)

    # Generate if missing or if the user requested refresh
    if force_refresh or not out.exists():
        # Require API key
        if not (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("ABSTRACT_API_KEY")
        ):
            raise RuntimeError(
                "OPENAI_API_KEY is required to generate embeddings (or provide EMBEDDING_API_KEY/ABSTRACT_API_KEY)."
            )
        asyncio.run(_generate_embeddings_file(out))

    # Validate generated file structure
    try:
        data = _json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("unexpected format (not a list)")
    except Exception as e:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid embeddings file at {out}: {e}")

    return out


def _root_sandbox_dir() -> Path:
    return Path(os.path.expanduser("~/.openbench/livemcpbench/root")).resolve()


def prepare_root_data(force_refresh: bool = False) -> Path:
    """Populate the sandbox root directory with annotated_data contents.

    If the root sandbox already appears populated and refresh is False,
    this function is a no-op.
    """
    annotated = get_annotated_data_cached(force_refresh)
    root_dir = _root_sandbox_dir()
    root_dir.mkdir(parents=True, exist_ok=True)

    # If already populated (has any files/dirs) and not forcing, skip copy
    if any(root_dir.iterdir()) and not force_refresh:
        return root_dir

    # Copy contents into root sandbox (flattened)
    for item in annotated.iterdir():
        target = root_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    return root_dir
