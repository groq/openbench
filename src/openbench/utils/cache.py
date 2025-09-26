import os
import shutil
from pathlib import Path
import typer

from openbench.tools.livemcpbench.copilot.prepare import (
    prepare_copilot_cache,
    prepare_root_data,
)


def prepare_livemcpbench_cache() -> Path:
    """Synchronously prepare all caches required by LiveMCPBench before eval.

    - Verifies OPENAI_API_KEY is present
    - Ensures upstream JSONs and embeddings exist (blocking)
    - Ensures root sandbox is staged with annotated_data
    - Exports MCP_DATA_PATH so the server uses the same embeddings file
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for LiveMCPBench (grading, key points, embeddings)."
        )

    typer.secho("\n🔧 Preparing LiveMCPBench caches...", fg=typer.colors.CYAN)

    cache_path = prepare_copilot_cache(force_refresh=False, embeddings_path=None)
    typer.echo(f"  ✅ Embedding cache ready: {cache_path}")

    # Make sure the child MCP server uses this exact path
    os.environ["MCP_DATA_PATH"] = str(cache_path)

    root_path = prepare_root_data(force_refresh=False)
    typer.echo(f"  ✅ Root sandbox ready: {root_path}\n")

    return root_path


def _livemcpbench_root_dir() -> Path:
    """Return the root sandbox directory used by LiveMCPBench tools.

    Kept in sync with copilot.prepare/_root_sandbox_dir and copilot.router.
    """
    return Path(os.path.expanduser("~/.openbench/livemcpbench/root")).resolve()


def clear_livemcpbench_root(quiet: bool = False) -> None:
    """Remove the LiveMCPBench root sandbox directory (~/.openbench/livemcpbench/root).

    This is safe to run after an eval; the directory is re-created/populated
    during the next `prepare_livemcpbench_cache()` call.
    """
    root = _livemcpbench_root_dir()
    try:
        if root.exists():
            shutil.rmtree(root)
            if not quiet:
                typer.echo(f"🧹 Cleaned LiveMCPBench root: {root}")
        else:
            if not quiet:
                typer.echo(f"(LiveMCPBench root already clean: {root})")
    except Exception as e:
        # Don’t raise in cleanup; just inform if not quiet
        if not quiet:
            typer.echo(f"⚠️  Failed to clean LiveMCPBench root ({root}): {e}")
