"""Utilities for locating cached FActScore assets.

The FActScore benchmark expects a large Wikipedia SQLite database and prompt
entity lists that are distributed with the official project.
To run factscore these assets need to be staged under ``~/.openbench/factscore`` so
multiple runs can share the same assets.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
import time

from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)

from openbench.utils.text import FACTSCORE_DB_SHA256


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


def ensure_resources_exist(required_file: Path) -> None:
    """Validate that all required paths exist, raising a helpful error."""

    missing = not required_file.exists()
    if not missing:
        return

    raise FactScoreResourceError(
        "Missing FactScore resources. download the assets here: \n"
        " https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I\n"
        "Use bench cache upload to upload the assets to the respective cache directory.\n"
        "The following paths were not found:\n  - "
        f"{required_file}"
    )


def knowledge_db_path(
    cache_root: str | os.PathLike[str] | None = None,
    filename: str = "enwiki-20230401.db",
) -> Path:
    """Return the path to the expected Wikipedia SQLite database."""

    path = data_dir(cache_root) / filename
    ensure_resources_exist(path)
    return path


def _check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if there's enough disk space at the given path."""
    stat = shutil.disk_usage(path)
    return stat.free >= required_bytes


def _compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity verification."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_factscore_db(
    cache_root: str | os.PathLike[str] | None = None,
    expected_sha256: str | None = FACTSCORE_DB_SHA256,
    max_retries: int = 3,
) -> Path:
    """Download the FActScore Wikipedia database from Hugging Face.

    This function handles the automatic download of the ~20GB Wikipedia SQLite
    database required for FActScore evaluation, with built-in integrity verification.

    Args:
        cache_root: Root directory for FActScore cache. Uses default if None.
        expected_sha256: Expected SHA-256 hash for verification. Defaults to known-good hash.
            Set to None to skip verification (not recommended).
        max_retries: Maximum number of download attempts (default: 3).

    Returns:
        Path to the downloaded database file.

    Raises:
        RuntimeError: If download fails after retries or verification fails.
        OSError: If insufficient disk space.
    """
    console = Console()
    output_path = data_dir(cache_root) / "enwiki-20230401.db"

    filename = "enwiki-20230401.db"
    repo_id = "lvogel123/factscore-data"

    if output_path.exists():
        file_size_gb = output_path.stat().st_size / (1024**3)
        console.print(
            f"[green]✓[/green] FActScore database already exists at {output_path} "
            f"({file_size_gb:.1f} GB)"
        )
        return output_path

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    # Check disk space (estimate 20GB + 2GB buffer)
    required_space = 22 * 1024 * 1024 * 1024
    if not _check_disk_space(output_path.parent, required_space):
        raise OSError(
            f"Insufficient disk space. Need at least 22GB free at {output_path.parent}\n"
            f"Current free space: {shutil.disk_usage(output_path.parent).free / (1024**3):.1f} GB"
        )

    console.print("\n[bold cyan]FActScore Database Download[/bold cyan]")
    console.print(
        f"Downloading Wikipedia database (~20GB) to:\n[blue]{output_path}[/blue]\n"
    )
    console.print(
        "[yellow]This is a one-time download and may take a while...[/yellow]\n"
    )

    temp_dir = tempfile.mkdtemp(prefix="factscore_", dir=output_path.parent)
    temp_path = Path(temp_dir) / "enwiki-20230401.db"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                console.print(
                    f"\n[yellow]Retry attempt {attempt}/{max_retries}...[/yellow]"
                )

            console.print("[cyan]Downloading from Hugging Face...[/cyan]")

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=temp_dir,
            )

            path = Path(downloaded_path)

            if not path.exists():
                raise RuntimeError("Download completed but file is missing")

            file_size = path.stat().st_size
            if file_size == 0:
                raise RuntimeError("Downloaded file is empty")

            file_size_gb = file_size / (1024**3)
            console.print(
                f"\n[green]✓[/green] Download complete ({file_size_gb:.1f} GB)"
            )

            # Verify file integrity if hash provided
            if expected_sha256:
                console.print("[yellow]Verifying file integrity...[/yellow]")
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Computing SHA-256 hash...", total=None)
                    actual_hash = _compute_file_hash(temp_path)
                    progress.update(task, completed=True)

                if actual_hash != expected_sha256:
                    raise RuntimeError(
                        f"File integrity check failed!\n"
                        f"Expected SHA-256: {expected_sha256}\n"
                        f"Got SHA-256:      {actual_hash}\n"
                        f"The file may be corrupted or tampered with."
                    )
                console.print("[green]✓[/green] File integrity verified")

            # Move to final location
            console.print("[cyan]Moving file to cache directory...[/cyan]")
            shutil.move(str(temp_path), str(output_path))
            shutil.rmtree(temp_dir, ignore_errors=True)

            console.print(
                f"\n[bold green]✓ Success![/bold green] "
                f"Database ready at {output_path}\n"
            )
            return output_path

        except Exception as e:
            last_error = e
            console.print(f"[red]✗ Attempt {attempt} failed: {e}[/red]")

            # Clean up partial download
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

            if attempt < max_retries:
                wait_time = 5 * (2 ** (attempt - 1))
                console.print(f"[yellow]Retrying in {wait_time} seconds...[/yellow]")
                time.sleep(wait_time)

    # Clean up temp directory
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    # All retries failed
    raise RuntimeError(
        f"Failed to download FActScore database after {max_retries} attempts.\n"
        f"Last error: {last_error}\n\n"
        f"You can manually download from:\n"
        f"https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I\n"
        f"Then upload using:\n"
        f"  bench cache upload --db_file <path> --path factscore/data/enwiki-20230401.db"
    )
