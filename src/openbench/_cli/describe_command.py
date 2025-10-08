"""
Describe command for benchmark evaluations.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED

from openbench.eval_config import get_eval_config
from openbench.config import get_all_benchmarks, EVAL_GROUPS
from openbench._cli.utils import get_category_display_name


def describe_eval_group(name: str, console: Console) -> None:
    """Show detailed information about an eval group."""
    group = EVAL_GROUPS[name]

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold blue]{group.name}[/bold blue] [dim](Eval Group)[/dim]",
            expand=False,
            box=ROUNDED,
        )
    )

    # Group metadata
    console.print("\n[bold yellow]Group Information[/bold yellow]")
    console.print("─" * 60)

    info_table = Table(show_header=False, show_lines=False, padding=(0, 2), box=None)
    info_table.add_column("Property", style="cyan", width=15)
    info_table.add_column("Value", style="white")

    info_table.add_row("Description", group.description)
    info_table.add_row("Benchmarks", f"{len(group.benchmarks)} tasks")
    info_table.add_row("Command", f"[bold]bench eval {name}[/bold]")

    console.print(info_table)

    # List constituent benchmarks
    console.print("\n[bold yellow]Benchmark Subtasks[/bold yellow]")
    console.print("─" * 60)

    # Show first 15 benchmarks
    display_limit = 15
    for benchmark in group.benchmarks[:display_limit]:
        console.print(f"  • {benchmark}")

    if len(group.benchmarks) > display_limit:
        remaining = len(group.benchmarks) - display_limit
        console.print(f"  [dim]... and {remaining} more[/dim]")

    # Footer
    console.print()
    console.print(
        f"[dim]Run all {len(group.benchmarks)} benchmarks with: "
        f"[bold]bench eval {name}[/bold][/dim]"
    )
    console.print()


def describe_eval(name: str) -> None:
    """Show detailed information about a specific evaluation."""
    console = Console()

    # Check if it's an eval group first
    if name in EVAL_GROUPS:
        describe_eval_group(name, console)
        return

    if name not in get_all_benchmarks():
        console.print(f"\n[red]Unknown evaluation: {name}[/red]")

        # Suggest similar names from both benchmarks and groups
        all_benchmarks = list(get_all_benchmarks().keys())
        all_groups = list(EVAL_GROUPS.keys())
        all_names = all_benchmarks + all_groups
        similar = [n for n in all_names if name.lower() in n.lower()]
        if similar:
            console.print("\nDid you mean one of these?")
            for s in similar[:5]:
                # Indicate if it's a group
                suffix = " (group)" if s in EVAL_GROUPS else ""
                console.print(f"   • {s}{suffix}")
        console.print()
        return

    # Get both static and dynamic config
    config = get_eval_config(name, load_dynamic=True)
    if not config:
        console.print(f"\n[red]Failed to load configuration for {name}[/red]\n")
        return

    # Header
    console.print()
    console.print(
        Panel(f"[bold blue]{config.name}[/bold blue]", expand=False, box=ROUNDED)
    )

    # Static metadata section
    console.print("\n[bold yellow]Metadata[/bold yellow]")
    console.print("─" * 40)

    static_table = Table(show_header=False, show_lines=False, padding=(0, 2), box=None)
    static_table.add_column("Property", style="cyan", width=15)
    static_table.add_column("Value", style="white")

    static_table.add_row("Description", config.description)
    static_table.add_row("Category", get_category_display_name(config.category))
    static_table.add_row("Command", f"[bold]bench eval {name}[/bold]")

    if config.tags:
        tag_str = " ".join(f"[blue]#{tag}[/blue]" for tag in config.tags)
        static_table.add_row("Tags", tag_str)

    console.print(static_table)

    # Dynamic configuration section
    if config._dynamic_loaded:
        console.print("\n[bold yellow]Configuration[/bold yellow]")
        console.print("─" * 40)

        config_table = Table(
            show_header=False, show_lines=False, padding=(0, 2), box=None
        )
        config_table.add_column("Property", style="cyan", width=15)
        config_table.add_column("Value", style="white")

        # Show key configuration values
        if config.epochs is not None:
            config_table.add_row("Epochs", str(config.epochs))

        # Show all GenerateConfig fields that have values
        config_dict = config.__dict__

        # Get all fields from config dict, excluding private/internal fields
        for field, value in sorted(config_dict.items()):
            # Skip private fields, None values, and non-GenerateConfig fields
            if (
                field.startswith("_")
                or value is None
                or field
                in [
                    "name",
                    "description",
                    "category",
                    "tags",
                    "epochs",
                    "sandbox",
                    "dataset_size",
                    "task_args",
                ]
            ):
                continue

            # Format the value nicely
            if isinstance(value, float):
                formatted_value = f"{value:.2f}" if value < 10 else f"{value:,.0f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}" if value > 999 else str(value)
            else:
                formatted_value = str(value)

            # Pretty field name
            field_name = field.replace("_", " ").title()
            config_table.add_row(field_name, formatted_value)

        if config.sandbox:
            config_table.add_row("Sandbox", config.sandbox)

        if config.dataset_size:
            config_table.add_row("Dataset Size", f"{config.dataset_size:,}")

        console.print(config_table)

        # Show task-specific arguments if any
        if config.task_args:
            console.print("\n[bold yellow]Task Arguments[/bold yellow]")
            console.print("─" * 40)

            args_table = Table(
                show_header=False, show_lines=False, padding=(0, 2), box=None
            )
            args_table.add_column("Argument", style="cyan", width=15)
            args_table.add_column("Default Value", style="white")

            for arg_name, arg_value in sorted(config.task_args.items()):
                # Pretty argument name
                display_name = arg_name.replace("_", " ").title()

                # Format the value
                if isinstance(arg_value, float):
                    formatted_value = (
                        f"{arg_value:.2f}" if arg_value < 10 else f"{arg_value:,.0f}"
                    )
                elif isinstance(arg_value, int):
                    formatted_value = (
                        f"{arg_value:,}" if arg_value > 999 else str(arg_value)
                    )
                elif isinstance(arg_value, bool):
                    formatted_value = "Yes" if arg_value else "No"
                else:
                    formatted_value = str(arg_value)

                args_table.add_row(display_name, formatted_value)

            console.print(args_table)
    else:
        console.print("\n[red]Failed to load dynamic configuration[/red]")

    # Footer
    console.print()
    console.print(f"[dim]Run with: [bold]bench eval {name}[/bold][/dim]")
    console.print()
