"""
Cached upstream fetch for LiveMCPBench Copilot artifacts.

This module downloads and caches the curated `clean_config.json` and the
`tools.json` from the LiveMCPBench repository, storing them under the user's
cache directory so subsequent runs can be offline and reproducible.

Defaults:
- Cache dir: ~/.openbench/livemcpbench/copilot/raw
- Refresh control: set env OPENBENCH_LIVEMCPBENCH_REFRESH=1 to force refetch
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Tuple

from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import tempfile
import zipfile
import shutil


# Raw content URLs
CLEAN_CONFIG_URL = (
    "https://raw.githubusercontent.com/icip-cas/LiveMCPBench/main/"
    "baseline/mcp_copilot/config/clean_config.json"
)
TOOLS_JSON_URL = (
    "https://raw.githubusercontent.com/icip-cas/LiveMCPBench/main/"
    "tools/LiveMCPTool/tools.json"
)

REPO_ZIP_URL = "https://github.com/icip-cas/LiveMCPBench/archive/refs/heads/main.zip"


def _base_cache_dir() -> Path:
    return Path(os.path.expanduser("~/.openbench/livemcpbench/copilot/raw")).resolve()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_json(url: str) -> Any:
    with urlopen(url, timeout=30) as resp:
        content = resp.read().decode("utf-8")
        return json.loads(content)


def _should_refresh(force: bool | None = None) -> bool:
    if force is not None:
        return force
    return os.getenv("OPENBENCH_LIVEMCPBENCH_REFRESH", "0") in {"1", "true", "True"}


def get_annotated_data_cached(force_refresh: bool | None = None) -> Path:
    """Fetch and cache annotated_data folder from upstream repo.

    Returns the local cache path containing the contents of annotated_data.
    """
    dest_dir = _base_cache_dir() / "annotated_data"
    refresh = _should_refresh(force_refresh)

    if dest_dir.exists() and any(dest_dir.iterdir()) and not refresh:
        return dest_dir

    if dest_dir.exists() and refresh:
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract repo zip
    with tempfile.TemporaryDirectory() as td:
        tmp_zip = Path(td) / "repo.zip"
        try:
            with urlopen(REPO_ZIP_URL, timeout=60) as resp, open(tmp_zip, "wb") as f:
                shutil.copyfileobj(resp, f)
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                zf.extractall(td)
        except Exception as e:
            # fallback to existing cache if available
            if any(dest_dir.iterdir()):
                return dest_dir
            raise RuntimeError(f"Unable to fetch repo zip for annotated_data: {e}")

        # Find extracted annotated_data path
        root = Path(td)
        candidate = None
        for child in root.iterdir():
            p = child / "annotated_data"
            if p.exists() and p.is_dir():
                candidate = p
                break
        if candidate is None:
            raise RuntimeError("annotated_data not found in repo zip")

        for item in candidate.iterdir():
            target = dest_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    return dest_dir


def get_clean_config_cached(
    force_refresh: bool | None = None,
) -> Tuple[dict[str, Any], Path]:
    """Fetch and cache clean_config.json. Returns (config_dict, cache_path)."""
    cache_path = _base_cache_dir() / "clean_config.json"
    refresh = _should_refresh(force_refresh)

    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data, cache_path

    try:
        data = _fetch_json(CLEAN_CONFIG_URL)
        _write_json(cache_path, data)
        return data, cache_path
    except (URLError, HTTPError, json.JSONDecodeError) as e:
        if cache_path.exists():
            # Fallback to cached
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data, cache_path
        raise RuntimeError(f"Unable to fetch clean_config.json: {e}")


def get_tools_json_cached(
    force_refresh: bool | None = None,
) -> Tuple[list[dict[str, Any]], Path]:
    """Fetch and cache tools.json. Returns (tools_list, cache_path)."""
    cache_path = _base_cache_dir() / "tools.json"
    refresh = _should_refresh(force_refresh)

    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        # tools.json is a list of server tool specs
        if isinstance(data, list):
            return data, cache_path
        else:
            # Corruption or unexpected
            cache_path.unlink(missing_ok=True)

    try:
        data = _fetch_json(TOOLS_JSON_URL)
        if not isinstance(data, list):
            raise ValueError("tools.json content is not a list")
        _write_json(cache_path, data)
        return data, cache_path
    except (URLError, HTTPError, json.JSONDecodeError, ValueError) as e:
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data, cache_path
        raise RuntimeError(f"Unable to fetch tools.json: {e}")
