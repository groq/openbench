from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, List

import typer


cache_app = typer.Typer(help="Manage OpenBench caches")


def _cache_root() -> Path:
    """Return the root cache directory for all OpenBench caches."""
    return Path(os.path.expanduser("~/.openbench")).resolve()


def _discover_cache_types() -> List[str]:
    """Auto-discover available cache types under ~/.openbench/"""
    root = _cache_root()
    if not root.exists():
        return []
    return [p.name for p in root.iterdir() if p.is_dir()]


def _get_cache_path(cache_type: Optional[str] = None) -> Path:
    """Get path for specific cache type or root if None."""
    root = _cache_root()
    if cache_type:
        return root / cache_type
    return root


def _get_default_cache_type() -> Optional[str]:
    """Get default cache type for backward compatibility.

    Returns 'livemcpbench' if it's the only cache type, otherwise None.
    """
    cache_types = _discover_cache_types()
    if len(cache_types) == 1 and "livemcpbench" in cache_types:
        return "livemcpbench"
    return None


def _human_size(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            pass
    return total


@cache_app.command("info")
def cache_info(
    cache_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Cache type to show info for (e.g., 'livemcpbench'). If not specified, shows all caches.",
    ),
) -> None:
    """Show cache information and sizes for specific cache type or all caches."""
    if cache_type:
        base = _get_cache_path(cache_type)
        if not base.exists():
            typer.echo(f"No {cache_type} cache directory found.")
            return
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        total = _dir_size(base)
        typer.secho(f"Cache ({cache_type}): {base}", fg=typer.colors.CYAN)
        typer.echo(f"Total size: {_human_size(total)}")
        if not subdirs:
            return
        typer.echo("\nSubdirectories:")
        for sd in sorted(subdirs, key=lambda p: p.name):
            size = _dir_size(sd)
            typer.echo(f"- {sd.name:<16} {_human_size(size)}")
    else:
        root = _cache_root()
        cache_types = _discover_cache_types()
        if not cache_types:
            typer.echo("No cache directories found.")
            return

        total_size = _dir_size(root)
        typer.secho(f"Cache root: {root}", fg=typer.colors.CYAN)
        typer.echo(f"Total size: {_human_size(total_size)}")
        typer.echo("\nCache types:")

        for ct in sorted(cache_types):
            cache_path = root / ct
            size = _dir_size(cache_path)
            typer.echo(f"- {ct:<16} {_human_size(size)}")

            # Show subdirectories for each cache type
            if cache_path.exists() and cache_path.is_dir():
                subdirs = [p for p in cache_path.iterdir() if p.is_dir()]
                if subdirs:
                    for sd in sorted(subdirs, key=lambda p: p.name):
                        subdir_size = _dir_size(sd)
                        typer.echo(f"  └─ {sd.name:<14} {_human_size(subdir_size)}")


def _print_tree(path: Path, prefix: str = "") -> None:
    try:
        items = sorted(list(path.iterdir()), key=lambda p: (not p.is_dir(), p.name))
    except FileNotFoundError:
        typer.echo(f"Path not found: {path}")
        return
    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        typer.echo(prefix + connector + item.name)
        if item.is_dir():
            extension = "    " if i == len(items) - 1 else "│   "
            _print_tree(item, prefix + extension)


@cache_app.command("ls")
def cache_ls(
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Subpath to list within the cache directory",
    ),
    tree: bool = typer.Option(False, "--tree", help="Print tree view"),
    cache_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Cache type to list (e.g., 'livemcpbench', 'scicode'). If not specified, uses default or root.",
    ),
) -> None:
    """List cache contents (optionally as a tree)."""
    # Backward compatibility: if no cache_type specified, try to default
    if not cache_type:
        cache_type = _get_default_cache_type()

    if cache_type:
        base = _get_cache_path(cache_type)
    else:
        base = _cache_root()

    target = base if not path else (base / path)
    if not target.exists():
        typer.echo(f"Path not found: {target}")
        raise typer.Exit(1)

    cache_label = f" ({cache_type})" if cache_type else ""
    typer.secho(f"{target}{cache_label}", fg=typer.colors.CYAN)

    if tree and target.is_dir():
        _print_tree(target)
    elif target.is_dir():
        entries = list(target.iterdir())
        if not entries:
            typer.echo("(empty)")
        for e in sorted(entries, key=lambda p: (not p.is_dir(), p.name)):
            suffix = "/" if e.is_dir() else ""
            typer.echo(e.name + suffix)
    else:
        typer.echo("(file)")


@cache_app.command("clear")
def cache_clear(
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Subpath within the cache directory to remove",
    ),
    all: bool = typer.Option(
        False, "--all", help="Remove entire cache directory for the specified type"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Do not prompt for confirmation"
    ),
    cache_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Cache type to clear (e.g., 'livemcpbench', 'scicode'). If not specified, uses default.",
    ),
) -> None:
    """Remove selected cache data with confirmation."""
    if all and path:
        typer.echo("Specify either --all or --path, not both.")
        raise typer.Exit(2)
    if not all and not path:
        typer.echo("Specify --all to clear everything or --path to clear a subpath.")
        raise typer.Exit(2)

    # Backward compatibility: if no cache_type specified, try to default
    if not cache_type:
        cache_type = _get_default_cache_type()
        if not cache_type:
            typer.echo(
                "No cache type specified and no default available. Use --type to specify a cache type."
            )
            available_types = _discover_cache_types()
            if available_types:
                typer.echo(f"Available cache types: {', '.join(available_types)}")
            raise typer.Exit(2)

    base = _get_cache_path(cache_type)
    target = base if all else (base / path)  # type: ignore[operator]

    if not target.exists():
        typer.echo(f"Path not found: {target}")
        raise typer.Exit(1)

    # Confirm
    cache_label = f" ({cache_type})" if cache_type else ""
    msg = f"This will permanently delete{cache_label}: {target}\nProceed?"
    if not yes and not typer.confirm(msg):
        typer.echo("Aborted.")
        return

    try:
        if target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
        typer.echo("✅ Cleared.")
    except Exception as e:
        typer.echo(f"❌ Failed to clear: {e}")
        raise typer.Exit(1)
