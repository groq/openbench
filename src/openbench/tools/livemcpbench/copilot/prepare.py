"""
Preparation utilities to prefetch and precompute Copilot caches.

"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from .server import _user_cache_dir, _ensure_parent_dir, _generate_embeddings_file
from .upstream_cache import (
    get_clean_config_cached,
    get_tools_json_cached,
    get_annotated_data_cached,
)
import shutil
import subprocess
import sys
from shutil import which


def _default_embeddings_path() -> Path:
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    abstract_model = os.getenv("ABSTRACT_MODEL", "gpt-4o-mini")
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

    return out


def _root_sandbox_dir() -> Path:
    return Path(os.path.expanduser("~/.cache/openbench/livemcpbench/root")).resolve()


def prepare_root_data(force_refresh: bool = False) -> Path:
    """Populate the sandbox root directory with annotated_data contents."""
    annotated = get_annotated_data_cached(force_refresh)
    root_dir = _root_sandbox_dir()
    root_dir.mkdir(parents=True, exist_ok=True)

    # Copy contents into root sandbox (flattened)
    for item in annotated.iterdir():
        target = root_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    return root_dir


def _default_playwright_browsers_path() -> Path:
    # Use platform default locations for Playwright browsers
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/ms-playwright"
    return Path(os.path.expanduser("~/.cache/ms-playwright")).resolve()


def install_playwright_browsers(
    browsers: Optional[list[str]] = None, browsers_path: Optional[Path] = None
) -> Path:
    """Install Playwright browsers using the MCP server's Playwright version.

    This installs @executeautomation/playwright-mcp-server in a local setup dir
    and runs its Playwright CLI to ensure browser revisions match.
    """
    if which("npm") is None:
        raise RuntimeError(
            "'npm' not found on PATH. Install Node.js (npm/npx) and retry."
        )

    setup_dir = Path.home() / ".cache/openbench/pw-setup"
    setup_dir.mkdir(parents=True, exist_ok=True)

    # Initialize package.json if needed
    if not (setup_dir / "package.json").exists():
        subprocess.run(["npm", "init", "-y"], cwd=setup_dir, check=True)

    # Install MCP server package to pin Playwright
    subprocess.run(
        ["npm", "i", "@executeautomation/playwright-mcp-server"],
        cwd=setup_dir,
        check=True,
    )

    cli = setup_dir / "node_modules/.bin/playwright"
    if sys.platform == "win32":
        cli = setup_dir / "node_modules/.bin/playwright.cmd"
    if not cli.exists():
        raise RuntimeError(f"Local Playwright CLI not found at {cli}")

    bp = browsers_path or _default_playwright_browsers_path()
    bp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(bp)

    args = [str(cli), "install"]
    if browsers and len(browsers) > 0:
        args.extend(browsers)
    else:
        args.append("chromium")

    subprocess.run(args, cwd=setup_dir, env=env, check=True)
    return bp
