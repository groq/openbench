#!/usr/bin/env python3
"""
Export openbench benchmark metadata for the docs catalog.

Outputs (fixed paths):
- docs/data/benchmarks.json
- docs/snippets/benchmarks.data.mdx (benchmarksData and evalGroupsData exports)

Fields included:
Benchmarks: name, description, category, tags, function_name, is_alpha
Eval Groups: name, description, id, benchmark_count, benchmarks
"""

from __future__ import annotations
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_benchmark(record: Any) -> Dict[str, Any]:
    """Normalize a BenchmarkMetadata into a plain dict with required fields only."""

    def clean_str(field: str) -> str:
        val = getattr(record, field, None)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"Missing or invalid {field} in benchmark {getattr(record, 'name', '<unknown>')}"
            )
        return val.strip()

    name = clean_str("name")
    description = clean_str("description")
    category = clean_str("category")
    function_name = clean_str("function_name")

    tags = getattr(record, "tags", None)
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(t, str) and t.strip() for t in tags)
    ):
        raise ValueError(
            f"Tags must be a non-empty list of strings in benchmark {name}"
        )

    is_alpha = getattr(record, "is_alpha", None)
    if not isinstance(is_alpha, bool):
        raise ValueError(f"is_alpha must be a boolean in benchmark {name}")

    return {
        "name": name,
        "description": description,
        "category": category,
        "tags": [t.strip() for t in tags],
        "function_name": function_name,
        "is_alpha": is_alpha,
    }


def normalize_eval_group(group_id: str, group: Any) -> Dict[str, Any]:
    """Normalize an EvalGroup into a plain dict."""
    name = getattr(group, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Missing or invalid name in eval group {group_id}")

    description = getattr(group, "description", None)
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Missing or invalid description in eval group {group_id}")

    benchmarks = getattr(group, "benchmarks", None)
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError(f"Missing or invalid benchmarks list in eval group {group_id}")

    return {
        "name": name.strip(),
        "description": description.strip(),
        "category": "eval-group",
        "tags": ["eval-group"],
        "id": group_id,
        "benchmark_count": len(benchmarks),
        "benchmarks": benchmarks,
    }


def export_benchmarks(
    snippet_mdx_path: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect, normalize, sort, and write benchmark and eval group metadata."""
    # Include alpha in the export; filtering happens in the UI
    config_module = importlib.import_module("openbench.config")
    all_benchmarks = getattr(config_module, "BENCHMARKS", None)
    if not isinstance(all_benchmarks, dict):
        raise AttributeError("Error loading benchmarks from openbench.config")

    all_groups = getattr(config_module, "EVAL_GROUPS", None)
    if not isinstance(all_groups, dict):
        raise AttributeError("Error loading eval groups from openbench.config")

    # Normalize benchmarks
    benchmark_rows: List[Dict[str, Any]] = []
    for eval_name, metadata in all_benchmarks.items():
        benchmark_rows.append(normalize_benchmark(metadata))

    # Normalize eval groups
    group_rows: List[Dict[str, Any]] = []
    for group_id, group in all_groups.items():
        group_rows.append(normalize_eval_group(group_id, group))

    # Sort benchmarks by name
    benchmark_rows.sort(key=lambda r: r.get("name", ""))
    # Sort groups by name
    group_rows.sort(key=lambda r: r.get("name", ""))

    # Write to MDX snippet for docs to source benchmarks info
    snippet_mdx_path.parent.mkdir(parents=True, exist_ok=True)
    snippet_mdx_path.write_text(
        "export const benchmarksData = "
        + json.dumps(benchmark_rows, ensure_ascii=False, indent=2)
        + ";\n\n"
        + "export const evalGroupsData = "
        + json.dumps(group_rows, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )

    return benchmark_rows, group_rows


def main() -> None:
    """Export benchmarks to MDX file under openbench/docs/snippets/"""
    snippet_mdx = Path("docs/snippets/benchmarks.data.mdx")
    export_benchmarks(snippet_mdx_path=snippet_mdx)


if __name__ == "__main__":
    main()
