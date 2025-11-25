#!/usr/bin/env python
"""
Aggregate ProgressiveMCPBench success logs into per-task summaries.

Usage:
    python scripts/progressivemcpbench_extract_success.py --input path/to/success_log.json --output summaries.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records.extend(data)
    return records


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        task_id = str(rec.get("task_id"))
        by_task[task_id].append(rec)

    summaries: list[dict[str, Any]] = []
    for task_id, recs in by_task.items():
        server_counter: Counter[str] = Counter()
        tool_counter: Counter[str] = Counter()
        strategies: Counter[str] = Counter()

        for rec in recs:
            for server in rec.get("servers") or []:
                server_counter[server] += 1
            for tool in rec.get("tools") or []:
                tool_counter[tool] += 1
            strategy = rec.get("strategy")
            if strategy:
                strategies[strategy] += 1

        most_common_server = server_counter.most_common(1)[0][0] if server_counter else None
        top_tools = tool_counter.most_common()

        summaries.append(
            {
                "task_id": task_id,
                "runs": len(recs),
                "most_common_server": most_common_server,
                "server_counts": dict(server_counter),
                "tool_counts": dict(tool_counter),
                "top_tools": top_tools,
                "strategies": dict(strategies),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ProgressiveMCPBench success logs."
    )
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        help="Path to a success log JSON file (can be repeated)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Where to write the aggregated summary JSON.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)

    records = load_records(input_paths)
    summaries = aggregate(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))

    print(
        f"Wrote {len(summaries)} task summaries from {len(records)} success records to {output_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
