from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import typer


cache_app = typer.Typer(help="Manage OpenBench caches (currently LiveMCPBench)")


def _lmcp_base() -> Path:
    return Path(os.path.expanduser("~/.openbench/livemcpbench")).resolve()


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
def cache_info() -> None:
    """Show total and per-subdir sizes under ~/.openbench/livemcpbench."""
    base = _lmcp_base()
    if not base.exists():
        typer.echo("No livemcpbench cache directory found.")
        return
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    total = _dir_size(base)
    typer.secho(f"Cache root: {base}", fg=typer.colors.CYAN)
    typer.echo(f"Total size: {_human_size(total)}")
    if not subdirs:
        return
    typer.echo("\nSubdirectories:")
    for sd in sorted(subdirs, key=lambda p: p.name):
        size = _dir_size(sd)
        typer.echo(f"- {sd.name:<16} {_human_size(size)}")


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
        help="Subpath under ~/.openbench/livemcpbench to list",
    ),
    tree: bool = typer.Option(False, "--tree", help="Print tree view"),
) -> None:
    """List cache contents (optionally as a tree)."""
    base = _lmcp_base()
    target = base if not path else (base / path)
    if not target.exists():
        typer.echo(f"Path not found: {target}")
        raise typer.Exit(1)
    typer.secho(str(target), fg=typer.colors.CYAN)
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
        help="Subpath under ~/.openbench/livemcpbench to remove",
    ),
    all: bool = typer.Option(
        False, "--all", help="Remove entire livemcpbench cache directory"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Do not prompt for confirmation"
    ),
) -> None:
    """Remove selected cache data with confirmation."""
    base = _lmcp_base()
    if all and path:
        typer.echo("Specify either --all or --path, not both.")
        raise typer.Exit(2)
    if not all and not path:
        typer.echo("Specify --all to clear everything or --path to clear a subpath.")
        raise typer.Exit(2)

    target = base if all else (base / path)  # type: ignore[operator]
    if not target.exists():
        typer.echo(f"Path not found: {target}")
        raise typer.Exit(1)

    # Confirm
    msg = f"This will permanently delete: {target}\nProceed?"
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
