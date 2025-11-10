"""
Helpers for integrating the external tau2-bench package.

We keep all tau2-specific bootstrapping logic here so the rest of the codebase
can assume the package (and its data assets) are ready before import.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

TAU2_PACKAGE_NAME = "tau2"
TAU2_REPO_URL = "https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT = "558e6cd066d7bf05db587fa2dc1509765c7d03bc"
TAU2_DATA_ENV = "TAU2_DATA_DIR"
DEFAULT_TAU2_DATA_DIR = Path("~/.openbench/tau2").expanduser()


class Tau2UnavailableError(RuntimeError):
    """Raised when the tau2 package (or its assets) are not available."""


def _install_hint() -> str:
    return "Install the tau-bench extra with: uv pip install -e '.[tau_bench]'"


def ensure_tau2_package() -> Any:
    """
    Import tau2, raising a helpful error if the optional dependency is missing.
    """
    try:
        import tau2  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on user env
        raise Tau2UnavailableError(
            "The 'tau2' package is required for tau-bench. " + _install_hint()
        ) from exc
    return tau2


def ensure_tau2_data_dir() -> Path:
    """
    Ensure TAU2_DATA_DIR points at a directory containing tau2's data assets.
    Downloads from the official repository if necessary.
    """
    data_dir_env = os.getenv(TAU2_DATA_ENV)
    if data_dir_env:
        path = Path(data_dir_env).expanduser()
        if not path.exists():
            raise Tau2UnavailableError(
                f"{TAU2_DATA_ENV}={path} does not exist. "
                "Either point it at a valid tau2 data checkout or unset it so "
                "openbench can download the assets."
            )
        return path

    target = DEFAULT_TAU2_DATA_DIR
    sentinel = target / "tau2" / "domains"
    if not sentinel.exists():
        _download_tau2_data(target)
    os.environ[TAU2_DATA_ENV] = str(target)
    return target


def ensure_tau2_ready() -> Any:
    """
    Convenience helper: ensure the package is installed and the data dir exists.
    Returns the imported tau2 module for convenience.
    """
    tau2 = ensure_tau2_package()
    ensure_tau2_data_dir()
    return tau2


def _download_tau2_data(target: Path) -> None:
    """Clone tau2-bench and copy its data directory into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="openbench_tau2_"))
    repo_dir = tmp_dir / "tau2-bench"
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                "main",
                TAU2_REPO_URL,
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
        )
        data_dir = repo_dir / "data"
        if not data_dir.exists():
            raise Tau2UnavailableError(
                f"Downloaded repository is missing the data directory at {data_dir}"
            )
        shutil.copytree(data_dir, target, dirs_exist_ok=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external git
        raise Tau2UnavailableError(
            f"Failed to download tau2 assets from {TAU2_REPO_URL}: {exc.stderr.decode().strip()}"
        ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
