import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import typer

from openbench.tools.progressivemcpbench.copilot.prepare import (
    prepare_copilot_cache,
    prepare_root_data,
)

PROGRESSIVEMCPBENCH_REPO_URL = "https://github.com/geelen/progressivemcpbench.git"
PROGRESSIVEMCPBENCH_DATA_ENV = "PROGRESSIVEMCPBENCH_DATA_DIR"
DEFAULT_PROGRESSIVEMCPBENCH_DATA_DIR = Path(
    "~/.openbench/progressivemcpbench/dataset"
).expanduser()


def _download_progressivemcpbench_data(target: Path) -> None:
    """Clone progressivemcpbench and copy its dataset directory into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="openbench_progressivemcpbench_"))
    repo_dir = tmp_dir / "progressivemcpbench"
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
                PROGRESSIVEMCPBENCH_REPO_URL,
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
        )
        dataset_dir = repo_dir / "dataset"
        if not dataset_dir.exists():
            raise ValueError(
                f"Downloaded repository is missing the dataset directory at {dataset_dir}"
            )
        shutil.copytree(dataset_dir, target, dirs_exist_ok=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Failed to download progressivemcpbench assets from {PROGRESSIVEMCPBENCH_REPO_URL}: "
            f"{exc.stderr.decode().strip()}"
        ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def get_progressivemcpbench_dataset_dir() -> Path:
    """Get the path to the progressivemcpbench dataset directory.

    Downloads from the official repository if necessary. The dataset contains:
    - config/servers.json: Server and tool definitions
    - tasks/progressivemcpbench.json: Evaluation tasks
    - data/: API mocks, files, web content
    """
    data_dir_env = os.getenv(PROGRESSIVEMCPBENCH_DATA_ENV)
    if data_dir_env:
        path = Path(data_dir_env).expanduser()
        if not path.exists():
            raise ValueError(
                f"{PROGRESSIVEMCPBENCH_DATA_ENV}={path} does not exist. "
                "Either point it at a valid progressivemcpbench dataset checkout or unset it "
                "so openbench can download the assets."
            )
        return path

    target = DEFAULT_PROGRESSIVEMCPBENCH_DATA_DIR
    sentinel = target / "tasks" / "progressivemcpbench.json"
    if not sentinel.exists():
        typer.secho(
            f"📥 Downloading ProgressiveMCPBench dataset to {target}...",
            fg=typer.colors.CYAN,
        )
        _download_progressivemcpbench_data(target)
        typer.secho("  ✅ Dataset downloaded", fg=typer.colors.GREEN)

    return target


def prepare_progressivemcpbench_cache(strategy: Optional[str] = None) -> Path:
    """Synchronously prepare all caches required by ProgressiveMCPBench before eval.

    Args:
        strategy: The strategy being used. If 'copilot', requires OPENAI_API_KEY
                  for embeddings generation. Other strategies only need upstream data.

    - For 'copilot': Verifies OPENAI_API_KEY and generates embeddings
    - For other strategies: Only fetches upstream JSONs (no embeddings needed)
    - Ensures root sandbox is staged with annotated_data
    """
    needs_embeddings = strategy == "copilot" or strategy is None

    if needs_embeddings and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for ProgressiveMCPBench with strategy=copilot "
            "(embeddings generation). Use -T strategy=directory or other strategies "
            "to skip embedding generation."
        )

    typer.secho("\n🔧 Preparing ProgressiveMCPBench caches...", fg=typer.colors.CYAN)

    if needs_embeddings:
        cache_path = prepare_copilot_cache(force_refresh=False, embeddings_path=None)
        typer.echo(f"  ✅ Embedding cache ready: {cache_path}")
        # Make sure the child MCP server uses this exact path
        os.environ["MCP_DATA_PATH"] = str(cache_path)

    root_path = prepare_root_data(force_refresh=False)
    typer.echo(f"  ✅ Root sandbox ready: {root_path}\n")

    return root_path


def _progressivemcpbench_root_dir() -> Path:
    """Return the root sandbox directory used by ProgressiveMCPBench tools.

    Kept in sync with copilot.prepare/_root_sandbox_dir and copilot.router.
    """
    return Path(os.path.expanduser("~/.openbench/progressivemcpbench/root")).resolve()


def clear_progressivemcpbench_root(quiet: bool = False) -> None:
    """Remove the ProgressiveMCPBench root sandbox directory (~/.openbench/progressivemcpbench/root).

    This is safe to run after an eval; the directory is re-created/populated
    during the next `prepare_progressivemcpbench_cache()` call.
    """
    root = _progressivemcpbench_root_dir()
    try:
        if root.exists():
            shutil.rmtree(root)
            if not quiet:
                typer.echo(f"🧹 Cleaned ProgressiveMCPBench root: {root}")
        else:
            if not quiet:
                typer.echo(f"(ProgressiveMCPBench root already clean: {root})")
    except Exception as e:
        # Don’t raise in cleanup; just inform if not quiet
        if not quiet:
            typer.echo(f"⚠️  Failed to clean ProgressiveMCPBench root ({root}): {e}")
