#!/usr/bin/env python3
"""
Extract detailed statistics from openbench/Inspect AI evaluation logs.

This script parses .eval files (ZIP archives) or .json log files and extracts:
- Timing statistics: total time, LLM HTTP time, tool time
- Token usage: input, output, total tokens
- Cache statistics: cached tokens (OpenAI), cache read/creation (Anthropic)
- Provider-specific timing: queue_time, prompt_time, completion_time (Groq)

Usage:
    python scripts/extract_eval_stats.py <eval_log_path> [-o output.json] [--per-sample] [-v]

Examples:
    python scripts/extract_eval_stats.py logs/my_eval.eval
    python scripts/extract_eval_stats.py logs/ -o stats.json --per-sample
"""

import argparse
import json
import statistics
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SampleStats:
    """Statistics for a single sample."""

    sample_id: str
    epoch: int | None = None
    score: float | None = None
    total_time: float | None = None
    working_time: float | None = None

    # Model call counts
    n_model_calls: int = 0
    n_tool_calls: int = 0

    # LLM timing (from HTTP calls)
    llm_http_time_total: float = 0.0
    llm_http_times: list[float] = field(default_factory=list)

    # Provider-specific timing (Groq)
    llm_provider_time_total: float = 0.0
    llm_queue_time_total: float = 0.0
    llm_prompt_time_total: float = 0.0
    llm_completion_time_total: float = 0.0

    # Token usage
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    total_tokens: int = 0

    # OpenAI caching
    openai_prompt_tokens_total: int = 0
    openai_cached_tokens_total: int = 0

    # Anthropic caching
    anthropic_input_tokens_total: int = 0
    anthropic_cache_read_tokens_total: int = 0
    anthropic_cache_creation_tokens_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "epoch": self.epoch,
            "score": self.score,
            "total_time": self.total_time,
            "working_time": self.working_time,
            "n_model_calls": self.n_model_calls,
            "n_tool_calls": self.n_tool_calls,
            "llm_http_time_total": self.llm_http_time_total,
            "llm_http_time_mean": (
                statistics.mean(self.llm_http_times) if self.llm_http_times else None
            ),
            "llm_provider_time_total": self.llm_provider_time_total,
            "llm_queue_time_total": self.llm_queue_time_total,
            "llm_prompt_time_total": self.llm_prompt_time_total,
            "llm_completion_time_total": self.llm_completion_time_total,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "total_tokens": self.total_tokens,
            "openai_prompt_tokens_total": self.openai_prompt_tokens_total,
            "openai_cached_tokens_total": self.openai_cached_tokens_total,
            "openai_cache_hit_rate": (
                self.openai_cached_tokens_total / self.openai_prompt_tokens_total
                if self.openai_prompt_tokens_total > 0
                else None
            ),
            "anthropic_input_tokens_total": self.anthropic_input_tokens_total,
            "anthropic_cache_read_tokens_total": self.anthropic_cache_read_tokens_total,
            "anthropic_cache_creation_tokens_total": self.anthropic_cache_creation_tokens_total,
        }


@dataclass
class EvalStats:
    """Aggregated statistics for an entire evaluation."""

    log_path: str
    model: str | None = None
    task: str | None = None
    num_samples: int = 0
    sample_stats: list[SampleStats] = field(default_factory=list)

    def aggregate(self) -> dict[str, Any]:
        """Compute aggregate statistics across all samples."""
        if not self.sample_stats:
            return {"log_path": self.log_path, "num_samples": 0}

        # Filter for valid values
        total_times = [
            s.total_time for s in self.sample_stats if s.total_time is not None
        ]
        working_times = [
            s.working_time for s in self.sample_stats if s.working_time is not None
        ]
        scores = [s.score for s in self.sample_stats if s.score is not None]

        # Counts
        total_model_calls = sum(s.n_model_calls for s in self.sample_stats)
        total_tool_calls = sum(s.n_tool_calls for s in self.sample_stats)

        # LLM timing - collect all individual call times for averaging
        all_http_times = []
        all_queue_times = []
        all_prompt_times = []
        all_completion_times = []
        all_provider_times = []

        for s in self.sample_stats:
            all_http_times.extend(s.llm_http_times)
            # Provider timing is summed per sample, so we need per-call averages
            if s.n_model_calls > 0:
                all_queue_times.append(s.llm_queue_time_total / s.n_model_calls)
                all_prompt_times.append(s.llm_prompt_time_total / s.n_model_calls)
                all_completion_times.append(
                    s.llm_completion_time_total / s.n_model_calls
                )
                all_provider_times.append(
                    s.llm_provider_time_total / s.n_model_calls
                )

        # Token sums and rates
        input_tokens_sum = sum(s.input_tokens_total for s in self.sample_stats)
        output_tokens_sum = sum(s.output_tokens_total for s in self.sample_stats)
        total_tokens_sum = sum(s.total_tokens for s in self.sample_stats)

        # Calculate throughput (tokens per second)
        llm_completion_time_sum = sum(
            s.llm_completion_time_total for s in self.sample_stats
        )
        output_tokens_per_sec = (
            output_tokens_sum / llm_completion_time_sum
            if llm_completion_time_sum > 0
            else None
        )

        llm_prompt_time_sum = sum(s.llm_prompt_time_total for s in self.sample_stats)
        input_tokens_per_sec = (
            input_tokens_sum / llm_prompt_time_sum if llm_prompt_time_sum > 0 else None
        )

        # OpenAI cache sums
        openai_prompt_sum = sum(s.openai_prompt_tokens_total for s in self.sample_stats)
        openai_cached_sum = sum(s.openai_cached_tokens_total for s in self.sample_stats)

        # Anthropic cache sums
        anthropic_input_sum = sum(
            s.anthropic_input_tokens_total for s in self.sample_stats
        )
        anthropic_cache_read_sum = sum(
            s.anthropic_cache_read_tokens_total for s in self.sample_stats
        )
        anthropic_cache_creation_sum = sum(
            s.anthropic_cache_creation_tokens_total for s in self.sample_stats
        )

        # Time breakdown
        llm_http_time_sum = sum(s.llm_http_time_total for s in self.sample_stats)
        tool_time_approx = (
            sum(working_times) - llm_http_time_sum if working_times else None
        )

        return {
            "log_path": self.log_path,
            "model": self.model,
            "task": self.task,
            "num_samples": len(self.sample_stats),
            # Scores
            "score_mean": statistics.mean(scores) if scores else None,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            # Sample timing
            "total_time_sum": sum(total_times) if total_times else None,
            "total_time_mean": statistics.mean(total_times) if total_times else None,
            "total_time_p50": (
                statistics.median(total_times) if total_times else None
            ),
            "total_time_p95": (
                sorted(total_times)[int(len(total_times) * 0.95)]
                if len(total_times) >= 20
                else None
            ),
            "working_time_sum": sum(working_times) if working_times else None,
            "working_time_mean": (
                statistics.mean(working_times) if working_times else None
            ),
            # LLM call timing (averages per call)
            "llm_http_time_mean": (
                statistics.mean(all_http_times) if all_http_times else None
            ),
            "llm_queue_time_mean": (
                statistics.mean(all_queue_times) if all_queue_times else None
            ),
            "llm_prompt_time_mean": (
                statistics.mean(all_prompt_times) if all_prompt_times else None
            ),
            "llm_completion_time_mean": (
                statistics.mean(all_completion_times) if all_completion_times else None
            ),
            "llm_provider_time_mean": (
                statistics.mean(all_provider_times) if all_provider_times else None
            ),
            # Time breakdown (for understanding where time goes)
            "llm_http_time_sum": llm_http_time_sum,
            "tool_time_approx": tool_time_approx,
            "llm_time_fraction": (
                llm_http_time_sum / sum(working_times)
                if working_times and sum(working_times) > 0
                else None
            ),
            # Counts
            "total_model_calls": total_model_calls,
            "total_tool_calls": total_tool_calls,
            # Tokens
            "input_tokens_sum": input_tokens_sum,
            "output_tokens_sum": output_tokens_sum,
            "total_tokens_sum": total_tokens_sum,
            # Throughput
            "output_tokens_per_sec": output_tokens_per_sec,
            "input_tokens_per_sec": input_tokens_per_sec,
            # OpenAI caching
            "openai_prompt_tokens_sum": openai_prompt_sum,
            "openai_cached_tokens_sum": openai_cached_sum,
            "openai_cache_hit_rate": (
                openai_cached_sum / openai_prompt_sum if openai_prompt_sum > 0 else None
            ),
            # Anthropic caching
            "anthropic_input_tokens_sum": anthropic_input_sum,
            "anthropic_cache_read_tokens_sum": anthropic_cache_read_sum,
            "anthropic_cache_creation_tokens_sum": anthropic_cache_creation_sum,
        }


def extract_usage_from_response(response: dict[str, Any]) -> dict[str, Any]:
    """
    Extract token usage and caching info from a raw model response.

    Handles both OpenAI and Anthropic response formats.
    """
    result: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "openai_prompt_tokens": 0,
        "openai_cached_tokens": 0,
        "anthropic_input_tokens": 0,
        "anthropic_cache_read_tokens": 0,
        "anthropic_cache_creation_tokens": 0,
    }

    usage = response.get("usage") or {}

    # OpenAI format
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    result["input_tokens"] = prompt_tokens
    result["output_tokens"] = completion_tokens
    result["total_tokens"] = total_tokens
    result["openai_prompt_tokens"] = prompt_tokens

    # OpenAI cached tokens
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens") or 0
    result["openai_cached_tokens"] = cached_tokens

    # Anthropic format (check for anthropic-specific fields)
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        result["anthropic_input_tokens"] = usage.get("input_tokens") or 0
        result["anthropic_cache_read_tokens"] = (
            usage.get("cache_read_input_tokens") or 0
        )
        result["anthropic_cache_creation_tokens"] = (
            usage.get("cache_creation_input_tokens") or 0
        )
        # For Anthropic, recalculate input tokens
        result["input_tokens"] = usage.get("input_tokens") or 0
        result["output_tokens"] = usage.get("output_tokens") or 0
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]

    return result


def extract_provider_timing(metadata: dict[str, Any]) -> dict[str, float]:
    """Extract provider-specific timing info (e.g., Groq timing)."""
    return {
        "queue_time": metadata.get("queue_time") or 0.0,
        "prompt_time": metadata.get("prompt_time") or 0.0,
        "completion_time": metadata.get("completion_time") or 0.0,
        "total_time": metadata.get("total_time") or 0.0,
    }


def build_span_hierarchy(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a mapping of span_id -> span info from span_begin events."""
    span_info: dict[str, dict[str, Any]] = {}
    for evt in events:
        if evt.get("event") == "span_begin":
            span_info[evt["id"]] = {
                "type": evt.get("type", ""),
                "name": evt.get("name", ""),
                "parent_id": evt.get("parent_id"),
            }
    return span_info


def get_root_span_type(span_id: str, span_info: dict[str, dict[str, Any]]) -> str:
    """Walk up the span hierarchy to find the root span type (solvers, scorers, init)."""
    visited: set[str] = set()
    while span_id and span_id not in visited:
        visited.add(span_id)
        info = span_info.get(span_id)
        if not info:
            return "unknown"
        if info["type"] in ("solvers", "scorers", "init"):
            return info["type"]
        span_id = info.get("parent_id", "")
    return "unknown"


def extract_sample_stats(sample_data: dict[str, Any], verbose: bool = False) -> SampleStats:
    """Extract statistics from a single sample.
    
    Only includes model calls from the 'solvers' span, excluding scorer LLM calls.
    """
    stats = SampleStats(
        sample_id=sample_data.get("id", "unknown"),
        epoch=sample_data.get("epoch"),
        total_time=sample_data.get("total_time"),
        working_time=sample_data.get("working_time"),
    )

    # Extract score
    scores = sample_data.get("scores") or {}
    for scorer_name, scorer_data in scores.items():
        if isinstance(scorer_data, dict) and "value" in scorer_data:
            stats.score = scorer_data["value"]
            break

    # Process events
    events = sample_data.get("events") or []
    
    # Build span hierarchy to distinguish solver vs scorer calls
    span_info = build_span_hierarchy(events)

    for event in events:
        event_type = event.get("event") or event.get("type") or event.get("kind", "")

        if event_type == "model" or "model" in str(event_type).lower():
            # Check if this is a solver call (not a scorer call)
            span_id = event.get("span_id", "")
            root_span_type = get_root_span_type(span_id, span_info)
            
            # Only count model calls from solvers, not scorers
            if root_span_type != "solvers":
                if verbose and root_span_type == "scorers":
                    pass  # Skip scorer calls silently unless debugging
                continue
            
            stats.n_model_calls += 1

            # HTTP time from call
            call = event.get("call") or {}
            http_time = call.get("time") or 0.0
            if http_time:
                stats.llm_http_time_total += http_time
                stats.llm_http_times.append(http_time)

            # Output contains usage and metadata
            output = event.get("output") or {}

            # Token usage from output.usage
            usage = output.get("usage") or {}
            stats.input_tokens_total += usage.get("input_tokens") or 0
            stats.output_tokens_total += usage.get("output_tokens") or 0
            stats.total_tokens += usage.get("total_tokens") or 0

            # Check for cache info in raw response
            response = call.get("response") or {}
            if response:
                cache_info = extract_usage_from_response(response)
                stats.openai_prompt_tokens_total += cache_info["openai_prompt_tokens"]
                stats.openai_cached_tokens_total += cache_info["openai_cached_tokens"]
                stats.anthropic_input_tokens_total += cache_info[
                    "anthropic_input_tokens"
                ]
                stats.anthropic_cache_read_tokens_total += cache_info[
                    "anthropic_cache_read_tokens"
                ]
                stats.anthropic_cache_creation_tokens_total += cache_info[
                    "anthropic_cache_creation_tokens"
                ]

            # Provider timing from output.metadata (Groq)
            metadata = output.get("metadata") or {}
            provider_timing = extract_provider_timing(metadata)
            stats.llm_provider_time_total += provider_timing["total_time"]
            stats.llm_queue_time_total += provider_timing["queue_time"]
            stats.llm_prompt_time_total += provider_timing["prompt_time"]
            stats.llm_completion_time_total += provider_timing["completion_time"]

        elif event_type == "tool" or "tool" in str(event_type).lower():
            stats.n_tool_calls += 1

    # Also check model_usage at sample level (fallback)
    model_usage = sample_data.get("model_usage") or {}
    if not stats.input_tokens_total and model_usage:
        for model_name, usage in model_usage.items():
            stats.input_tokens_total += usage.get("input_tokens") or 0
            stats.output_tokens_total += usage.get("output_tokens") or 0
            stats.total_tokens += usage.get("total_tokens") or 0

    return stats


def parse_eval_file(eval_file: Path, verbose: bool = False) -> EvalStats:
    """Parse a .eval ZIP file and extract statistics."""
    eval_stats = EvalStats(log_path=str(eval_file))

    try:
        with zipfile.ZipFile(eval_file, "r") as zf:
            # Read start journal for metadata
            try:
                with zf.open("_journal/start.json") as f:
                    start_data = json.load(f)
                    eval_info = start_data.get("eval") or {}
                    eval_stats.model = eval_info.get("model")
                    eval_stats.task = eval_info.get("task")
            except (KeyError, json.JSONDecodeError):
                pass

            # Find all sample files
            sample_files = [
                name
                for name in zf.namelist()
                if name.startswith("samples/") and name.endswith(".json")
            ]

            for sample_file in sample_files:
                try:
                    with zf.open(sample_file) as f:
                        sample_data = json.load(f)
                        sample_stats = extract_sample_stats(sample_data, verbose)
                        eval_stats.sample_stats.append(sample_stats)
                except (json.JSONDecodeError, KeyError) as e:
                    if verbose:
                        print(
                            f"Warning: Could not parse {sample_file}: {e}",
                            file=sys.stderr,
                        )

            eval_stats.num_samples = len(eval_stats.sample_stats)

    except zipfile.BadZipFile as e:
        print(f"Warning: Could not read ZIP file {eval_file}: {e}", file=sys.stderr)

    return eval_stats


def parse_json_file(json_file: Path, verbose: bool = False) -> EvalStats:
    """Parse a .json log file and extract statistics."""
    eval_stats = EvalStats(log_path=str(json_file))

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Try to find samples in various locations
        samples = []
        if "results" in data and "samples" in data["results"]:
            samples = data["results"]["samples"]
        elif "samples" in data:
            samples = data["samples"]

        for sample_data in samples:
            sample_stats = extract_sample_stats(sample_data, verbose)
            eval_stats.sample_stats.append(sample_stats)

        eval_stats.num_samples = len(eval_stats.sample_stats)

        # Try to get model/task info
        if "eval" in data:
            eval_stats.model = data["eval"].get("model")
            eval_stats.task = data["eval"].get("task")

    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not parse {json_file}: {e}", file=sys.stderr)

    return eval_stats


def find_log_files(log_path: Path) -> list[Path]:
    """Find all log files (.eval or .json) in a path."""
    if log_path.is_file():
        if log_path.suffix in (".eval", ".json"):
            return [log_path]
        return []

    log_files = []
    for f in log_path.rglob("*.eval"):
        log_files.append(f)
    for f in log_path.rglob("*.json"):
        if "samples" not in f.parts:  # Skip sample subdirectory files
            log_files.append(f)

    return sorted(log_files)


def format_time(seconds: float | None) -> str:
    """Format seconds as human-readable time (e.g., '2h 20m', '45s', '1.23s')."""
    if seconds is None:
        return "N/A"
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if secs == 0:
            return f"{minutes}m"
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        # Round to nearest 10 minutes for long durations
        if hours >= 3:
            minutes = round(minutes / 10) * 10
            if minutes == 60:
                hours += 1
                minutes = 0
        if minutes == 0:
            return f"{hours}h"
        return f"{hours}h {minutes}m"


def format_tokens(count: int | None) -> str:
    """Format token count as human-readable number."""
    if count is None or count == 0:
        return "0"
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count/1000:.1f}K"
    else:
        return f"{count/1_000_000:.2f}M"


def format_percentage(value: float | None) -> str:
    """Format a fraction as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_rate(tokens_per_sec: float | None) -> str:
    """Format tokens per second as human-readable rate."""
    if tokens_per_sec is None:
        return "N/A"
    if tokens_per_sec >= 1000:
        return f"{tokens_per_sec/1000:.1f}K tok/s"
    return f"{tokens_per_sec:.0f} tok/s"


def print_summary(eval_stats: EvalStats) -> None:
    """Print a human-readable summary of eval statistics."""
    agg = eval_stats.aggregate()

    print(f"\n{'=' * 60}")
    print(f"EVAL: {agg['log_path']}")
    print(f"{'=' * 60}")

    if agg["model"]:
        print(f"  Model: {agg['model']}")
    if agg["task"]:
        print(f"  Task: {agg['task']}")

    print(f"\n  Samples: {agg['num_samples']}")
    if agg["score_mean"] is not None:
        print(
            f"  Score: mean={agg['score_mean']:.3f} (min={agg['score_min']:.3f}, max={agg['score_max']:.3f})"
        )

    print(f"\n  CALLS:")
    print(f"    Model calls: {agg['total_model_calls']}")
    print(f"    Tool calls: {agg['total_tool_calls']}")

    print(f"\n  TIME BREAKDOWN:")
    print(f"    Total eval time: {format_time(agg['total_time_sum'])}")
    print(
        f"    LLM time: {format_time(agg['llm_http_time_sum'])} ({format_percentage(agg['llm_time_fraction'])})"
    )
    print(f"    Tool time (approx): {format_time(agg['tool_time_approx'])}")

    # LLM call performance (averages)
    if agg["llm_http_time_mean"] is not None:
        print(f"\n  LLM CALL PERFORMANCE (avg per call):")
        print(f"    HTTP round-trip: {format_time(agg['llm_http_time_mean'])}")
        if agg["llm_queue_time_mean"] is not None and agg["llm_queue_time_mean"] > 0:
            print(f"    Time to first token (queue): {format_time(agg['llm_queue_time_mean'])}")
        if agg["llm_prompt_time_mean"] is not None and agg["llm_prompt_time_mean"] > 0:
            print(f"    Prompt processing: {format_time(agg['llm_prompt_time_mean'])}")
        if agg["llm_completion_time_mean"] is not None and agg["llm_completion_time_mean"] > 0:
            print(f"    Completion generation: {format_time(agg['llm_completion_time_mean'])}")

    print(f"\n  TOKENS:")
    print(f"    Input: {format_tokens(agg['input_tokens_sum'])}")
    print(f"    Output: {format_tokens(agg['output_tokens_sum'])}")
    print(f"    Total: {format_tokens(agg['total_tokens_sum'])}")

    # Throughput
    if agg["output_tokens_per_sec"] is not None or agg["input_tokens_per_sec"] is not None:
        print(f"\n  THROUGHPUT:")
        if agg["input_tokens_per_sec"] is not None:
            print(f"    Input processing: {format_rate(agg['input_tokens_per_sec'])}")
        if agg["output_tokens_per_sec"] is not None:
            print(f"    Output generation: {format_rate(agg['output_tokens_per_sec'])}")

    if agg["openai_cached_tokens_sum"] and agg["openai_cached_tokens_sum"] > 0:
        print(f"\n  CACHING (OpenAI-compatible):")
        print(f"    Prompt tokens: {format_tokens(agg['openai_prompt_tokens_sum'])}")
        print(f"    Cached tokens: {format_tokens(agg['openai_cached_tokens_sum'])}")
        print(f"    Cache hit rate: {format_percentage(agg['openai_cache_hit_rate'])}")

    if (
        agg["anthropic_cache_read_tokens_sum"]
        and agg["anthropic_cache_read_tokens_sum"] > 0
    ) or (
        agg["anthropic_cache_creation_tokens_sum"]
        and agg["anthropic_cache_creation_tokens_sum"] > 0
    ):
        print(f"\n  CACHING (Anthropic):")
        print(f"    Input tokens: {format_tokens(agg['anthropic_input_tokens_sum'])}")
        print(
            f"    Cache read: {format_tokens(agg['anthropic_cache_read_tokens_sum'])}"
        )
        print(
            f"    Cache creation: {format_tokens(agg['anthropic_cache_creation_tokens_sum'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract statistics from openbench evaluation logs"
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to log directory or individual log file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file for JSON statistics",
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        help="Include per-sample statistics in output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"Error: Path does not exist: {args.log_path}", file=sys.stderr)
        sys.exit(1)

    log_files = find_log_files(args.log_path)
    if not log_files:
        print(f"Error: No log files found in {args.log_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(log_files)} log file(s)")

    all_eval_stats: list[EvalStats] = []

    for log_file in log_files:
        if args.verbose:
            print(f"Processing {log_file}")

        if log_file.suffix == ".eval":
            eval_stats = parse_eval_file(log_file, args.verbose)
        else:
            eval_stats = parse_json_file(log_file, args.verbose)

        all_eval_stats.append(eval_stats)
        print_summary(eval_stats)

    # Write JSON output if requested
    if args.output:
        output_data: dict[str, Any] = {
            "evals": [es.aggregate() for es in all_eval_stats],
        }
        if args.per_sample:
            output_data["samples"] = []
            for es in all_eval_stats:
                for sample in es.sample_stats:
                    sample_dict = sample.to_dict()
                    sample_dict["eval_log_path"] = es.log_path
                    output_data["samples"].append(sample_dict)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nJSON output written to: {args.output}")


if __name__ == "__main__":
    main()
