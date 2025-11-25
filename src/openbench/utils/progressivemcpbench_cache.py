import os
import shutil
from pathlib import Path
import typer

from openbench.tools.progressivemcpbench.copilot.prepare import (
    prepare_copilot_cache,
    prepare_root_data,
)
from openbench.tools.progressivemcpbench.copilot.upstream_cache import (
    get_clean_config_cached,
    get_tools_json_cached,
)


def prepare_progressivemcpbench_cache(strategy: str | None = None) -> Path:
    """Synchronously prepare all caches required by ProgressiveMCPBench before eval.

    - When using the copilot strategy, ensures embeddings exist (requires OPENAI_API_KEY/EMBEDDING_API_KEY)
    - For other strategies, only fetches upstream JSONs and prepares the root sandbox
    - Ensures root sandbox is staged with annotated_data
    - Exports MCP_DATA_PATH so the server uses the same embeddings file (copilot)
    """
    normalized = (strategy or "").lower()
    needs_embeddings = normalized in {"", "copilot", None}

    typer.secho("\n🔧 Preparing ProgressiveMCPBench caches...", fg=typer.colors.CYAN)

    if needs_embeddings:
        if not (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("ABSTRACT_API_KEY")
        ):
            raise RuntimeError(
                "OPENAI_API_KEY is required for the copilot strategy (embeddings generation)."
            )
        cache_path = prepare_copilot_cache(force_refresh=False, embeddings_path=None)
        typer.echo(f"  ✅ Embedding cache ready: {cache_path}")
        os.environ["MCP_DATA_PATH"] = str(cache_path)
    else:
        # Ensure upstream configs/tools exist even when embeddings are skipped
        get_clean_config_cached(force_refresh=False)
        get_tools_json_cached(force_refresh=False)
        typer.echo("  ✅ Config + tools cache ready (embeddings not required)")

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
