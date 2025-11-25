"""
Utilities for extracting ProgressiveMCPBench success usage logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from inspect_ai.log import EvalLog, EvalSample

from openbench.utils.progressivemcp_output import parse_progressivemcp_output


def _first_score(sample: EvalSample):
    if sample.scores:
        return next(iter(sample.scores.values()))
    return None


def _tool_calls_from_sample(sample: EvalSample) -> list[dict[str, Any]]:
    # Prefer scorer-provided metadata
    if sample.scores:
        for score in sample.scores.values():
            meta = score.metadata or {}
            calls = meta.get("tool_calls")
            if isinstance(calls, list):
                return calls

    # Fallback: parse completion directly
    raw = getattr(sample.output, "completion", None) if sample.output else None
    parsed = parse_progressivemcp_output(raw) if raw else None
    calls = parsed.get("tool_calls") if parsed else None
    return calls if isinstance(calls, list) else []


def extract_progressivemcp_success(
    eval_logs: Iterable[EvalLog], *, min_score: float = 1.0
) -> list[dict[str, Any]]:
    """Extract successful tool usage records from eval logs."""
    records: list[dict[str, Any]] = []
    for log in eval_logs:
        task_name = log.eval.task.split("/")[-1] if log.eval and log.eval.task else ""
        if task_name != "progressivemcpbench":
            continue

        strategy = None
        if log.eval and log.eval.task_args:
            strategy = log.eval.task_args.get("strategy")

        for sample in log.samples or []:
            score_obj = _first_score(sample)
            score_value = score_obj.value if score_obj else None
            if score_value is None or score_value < min_score:
                continue

            tool_calls = _tool_calls_from_sample(sample)
            servers = sorted(
                {
                    c.get("server_name")
                    for c in tool_calls
                    if isinstance(c, dict) and c.get("server_name")
                }
            )
            tools = sorted(
                {
                    f"{c.get('server_name')}::{c.get('tool_name')}"
                    for c in tool_calls
                    if isinstance(c, dict)
                    and c.get("server_name")
                    and c.get("tool_name")
                }
            )

            records.append(
                {
                    "task_id": sample.id,
                    "category": (sample.metadata or {}).get("category")
                    if sample.metadata
                    else None,
                    "strategy": strategy,
                    "score": score_value,
                    "servers": servers,
                    "tools": tools,
                    "tool_calls": tool_calls,
                }
            )
    return records


def write_progressivemcp_success_log(
    records: List[dict[str, Any]], output_path: Path
) -> Path:
    """Write records to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return output_path
