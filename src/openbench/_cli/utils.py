"""
Utility functions shared across CLI commands.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import yaml
from openbench.config import BenchmarkMetadata


def get_category_display_name(category: str) -> str:
    """Generate display name for category."""
    display = category.replace("_", " ").replace("-", " ").title()
    if "benchmark" not in display.lower():
        display += " Benchmarks"
    return display


def benchmark_to_eval_config(meta: BenchmarkMetadata):
    """Convert BenchmarkMetadata to a simple eval config for display."""
    return SimpleNamespace(
        name=meta.name,
        description=meta.description,
        category=meta.category,
        tags=meta.tags,
    )


def matches_search(eval_config, query: str) -> bool:
    """Check if an evaluation matches the search query."""
    query = query.lower()
    return (
        query in eval_config.name.lower()
        or query in eval_config.description.lower()
        or any(query in tag.lower() for tag in eval_config.tags)
    )


def parse_cli_args(
    args: Optional[List[str]], force_str: bool = False
) -> Dict[str, Any]:
    """Parse CLI arguments in the format key=value.

    Args:
        args: List of arguments in the format key=value
        force_str: Force all values to be strings

    Returns:
        Dictionary of parsed arguments
    """
    params: Dict[str, Any] = {}
    if args:
        for arg in args:
            parts = arg.split("=", 1)
            if len(parts) == 2:
                key = parts[0].replace("-", "_")
                try:
                    # Try to parse with yaml for proper type conversion
                    value = yaml.safe_load(parts[1])
                    # Handle comma-separated values as lists
                    if isinstance(value, str) and "," in value:
                        value = value.split(",")
                        value = value if len(value) > 1 else value[0]
                except yaml.YAMLError:
                    # If parsing fails, treat as string
                    value = parts[1]
                params[key] = str(value) if force_str else value
            else:
                # If no '=' found, this is an invalid argument format
                raise ValueError(f"Invalid argument format: {arg}. Expected key=value")
    return params
