"""
List command for benchmark evaluations.
"""

import typer
from typing import Optional, Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED

from openbench.config import (
    get_all_benchmarks,
    get_benchmarks_by_category,
    get_categories,
)
from openbench._cli.utils import (
    get_category_display_name,
    benchmark_to_eval_config,
    matches_search,
)


def list_evals(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category (core, math)"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Search evaluations by name, description, or tags"
    ),
    tags: bool = typer.Option(
        False, "--tags", "-t", help="Show tags for each benchmark"
    ),
    alpha: bool = typer.Option(
        False, "--alpha", help="Include experimental/alpha benchmarks"
    ),
) -> None:
    """List available benchmark evaluations with enhanced UI."""
    console = Console()

    # Get evaluations based on filters
    if category:
        if category not in get_categories():
            console.print(f"\n❌ [red]Unknown category: {category}[/red]")
            console.print(f"   Available: {', '.join(sorted(get_categories()))}\n")
            return
        benchmarks = get_benchmarks_by_category(category, include_alpha=alpha)
        evals = [benchmark_to_eval_config(meta) for meta in benchmarks.values()]
    else:
        all_benchmarks = get_all_benchmarks(include_alpha=alpha)
        evals = [benchmark_to_eval_config(meta) for meta in all_benchmarks.values()]

    # Apply search filter
    if search:
        evals = [e for e in evals if matches_search(e, search)]

    if not evals:
        console.print("\n💭 [yellow]No evaluations match your criteria.[/yellow]\n")
        return

    # Group by category
    categories: Dict[str, List[Any]] = {}
    for eval_config in evals:
        if eval_config.category not in categories:
            categories[eval_config.category] = []
        categories[eval_config.category].append(eval_config)

    # Header
    console.print()
    if search:
        header = Text(f"Search Results for '{search}'", style="bold blue")
    else:
        header = Text("Available Benchmarks", style="bold blue")
    console.print(Panel(header, expand=False, box=ROUNDED))
    console.print()

    # Display each category
    for cat_name in sorted(categories.keys()):
        display_name = get_category_display_name(cat_name)

        # Category header with count
        cat_count = len(categories[cat_name])
        console.print(
            f"[bold green]{display_name}[/bold green] [dim]({cat_count})[/dim]"
        )
        console.print("─" * 60)

        # Get task names for this category
        cat_evals_with_keys = [
            (k, v)
            for k, v in get_all_benchmarks(include_alpha=alpha).items()
            if v.name in [e.name for e in categories[cat_name]]
        ]
        cat_evals_with_keys = sorted(cat_evals_with_keys, key=lambda x: x[0])

        # Create table for this category
        table = Table(show_header=False, show_lines=False, padding=(0, 1), box=None)
        table.add_column("Key", style="cyan", width=18)
        table.add_column("Name", style="white", width=20)
        table.add_column("Description", style="dim")

        for task_key, benchmark_meta in cat_evals_with_keys:
            # Find the corresponding eval config
            eval_config = next(
                e for e in categories[cat_name] if e.name == benchmark_meta.name
            )

            # Format description
            desc = eval_config.description
            if len(desc) > 60:
                desc = desc[:57] + "..."

            # Add tags if requested
            if tags and eval_config.tags:
                tag_str = (
                    " [dim blue]"
                    + " · ".join(f"#{tag}" for tag in eval_config.tags[:3])
                    + "[/dim blue]"
                )
                desc += tag_str

            table.add_row(f"[bold cyan]{task_key}[/bold cyan]", eval_config.name, desc)

        console.print(table)
        console.print()

    # Footer with stats and help
    total_count = len(evals)
    console.print("─" * 60)
    status_msg = f"[dim]Total: {total_count} benchmark{'s' if total_count != 1 else ''}"
    if not alpha:
        status_msg += " (use --alpha to see experimental benchmarks)"
    status_msg += "[/dim]"
    console.print(status_msg)
    console.print()
    console.print("[dim]Commands:[/dim]")
    console.print("   bench describe <name> - Show detailed information")
    console.print("   bench eval <name>     - Run evaluation")
    console.print()
