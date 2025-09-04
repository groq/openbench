"""
Utilities for managing the embedding cache.
"""

import click
from .embedding_cache import get_embedding_cache
from openbench.tools.livemcpbench import MCPToolsRegistry


@click.group()
def cli():
    """LiveMCPBench embedding cache management."""
    pass


@cli.command()
def stats():
    """Show cache statistics."""
    cache = get_embedding_cache()
    stats = cache.get_stats()

    click.echo("LiveMCPBench Embedding Cache Statistics")
    click.echo("=" * 40)
    click.echo(f"Cached embeddings: {stats['cached_embeddings']}")
    click.echo(f"Cache size: {stats['cache_size_mb']:.2f} MB")
    click.echo(f"Cache directory: {stats['cache_dir']}")


@cli.command()
def clear():
    """Clear all cached embeddings."""
    if click.confirm("Are you sure you want to clear all cached embeddings?"):
        cache = get_embedding_cache()
        cache.clear()
        click.echo("Cache cleared successfully.")
    else:
        click.echo("Cache clear cancelled.")


@cli.command()
@click.option("--force", is_flag=True, help="Force re-index without confirmation")
def reindex(force):
    """Re-index all tools (regenerate embeddings)."""
    if not force and not click.confirm(
        "Re-indexing will regenerate all embeddings. Continue?"
    ):
        click.echo("Re-indexing cancelled.")
        return

    # Clear cache first
    cache = get_embedding_cache()
    cache.clear()

    click.echo("Re-indexing tools...")
    registry = MCPToolsRegistry()
    registry.init_retriever()

    # Show new stats
    stats = cache.get_stats()
    click.echo(f"Re-indexing complete. Cached {stats['cached_embeddings']} embeddings.")


if __name__ == "__main__":
    cli()
