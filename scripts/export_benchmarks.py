#!/usr/bin/env python3
"""
Export OpenBench benchmark metadata for the docs catalog.

Outputs (fixed paths):
- docs/data/benchmarks.json
- docs/snippets/benchmarks.data.mdx (benchmarksData export)

Fields included:
- name, description, category, tags, function_name, is_alpha
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


def export_benchmarks(snippet_mdx_path: Path) -> List[Dict[str, Any]]:
    """Collect, normalize, sort, and write benchmark metadata JSON."""
    # Include alpha in the export; filtering happens in the UI
    config_module = importlib.import_module("openbench.config")
    all_benchmarks = getattr(config_module, "BENCHMARKS", None)
    if not isinstance(all_benchmarks, dict):
        raise AttributeError("Error loading benchmarks from openbench.config")

    rows: List[Dict[str, Any]] = []
    for eval_name, metadata in all_benchmarks.items():
        rows.append(normalize_benchmark(metadata))

    # Output sorted by category, then name
    rows.sort(key=lambda r: (r.get("category", ""), r.get("name", "")))

    # write to MDX snippet for docs to source benchmarks info
    snippet_mdx_path.parent.mkdir(parents=True, exist_ok=True)
    snippet_mdx_path.write_text(
        "export const benchmarksData = "
        + json.dumps(rows, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )

    return rows


def main() -> None:
    """Export benchmarks to MDX file under openbench/docs/snippets/"""
    snippet_mdx = Path("docs/snippets/benchmarks.data.mdx")
    export_benchmarks(snippet_mdx_path=snippet_mdx)


if __name__ == "__main__":
    main()
