#!/usr/bin/env python3
"""
Extract successful task completions from ProgressiveMCPBench eval logs.

This script parses the eval output to extract:
- task_id: The task identifier
- servers_used: Set of server names that were called
- tools_used: Set of tool names that were called

The output can be used to annotate the dataset with which servers/tools
are needed for each task, enabling the minimal-servers and minimal-tools strategies.

Usage:
    python scripts/extract_success_logs.py <eval_log_dir> -o success_annotations.json

Example:
    python scripts/extract_success_logs.py logs/progressivemcpbench-2025-01-15 -o annotations.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def extract_tool_calls_from_output(output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from the model's JSON output."""
    tool_calls = []

    # Try to get tool_calls from the final answer
    completion = output.get("completion", "")
    if not completion:
        return tool_calls

    # Parse the JSON output
    try:
        # Try to extract JSON from the completion
        json_match = re.search(r"\{.*\}", completion, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed.get("tool_calls"), list):
                tool_calls = parsed["tool_calls"]
    except (json.JSONDecodeError, TypeError):
        pass

    return tool_calls


def extract_server_and_tool_names(
    tool_calls: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Extract server names and tool names from tool calls."""
    servers = set()
    tools = set()

    for call in tool_calls:
        # Handle copilot-style tool calls
        if "server_name" in call:
            servers.add(call["server_name"])
        if "tool_name" in call:
            tools.add(call["tool_name"])

        # Handle directory-style tool calls
        if "tool_path" in call:
            path = call["tool_path"]
            if path.startswith("/tools/"):
                parts = path[7:].split("/")
                if len(parts) >= 2:
                    servers.add(parts[0])
                    tool_name = parts[1]
                    if tool_name.endswith(".md"):
                        tool_name = tool_name[:-3]
                    tools.add(tool_name)

    return servers, tools


def parse_log_file(log_file: Path) -> list[dict[str, Any]]:
    """Parse a single log file and extract sample data."""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not parse {log_file}: {e}", file=sys.stderr)
        return []

    samples = []
    results = data.get("results", {})

    for sample in results.get("samples", []):
        sample_id = sample.get("id", "")
        score = sample.get("score", {})
        score_value = score.get("value", 0)

        # Only process successful samples (score >= 0.5 for partial credit)
        if score_value < 0.5:
            continue

        output = sample.get("output", {})
        tool_calls = extract_tool_calls_from_output(output)
        servers, tools = extract_server_and_tool_names(tool_calls)

        # Also try to extract from messages if output doesn't have tool_calls
        if not servers and not tools:
            messages = sample.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        if tc.get("function", {}).get("name") == "execute-tool":
                            try:
                                args = json.loads(tc["function"].get("arguments", "{}"))
                                if "server_name" in args:
                                    servers.add(args["server_name"])
                                if "tool_name" in args:
                                    tools.add(args["tool_name"])
                                if "tool_path" in args:
                                    path = args["tool_path"]
                                    if path.startswith("/tools/"):
                                        parts = path[7:].split("/")
                                        if len(parts) >= 2:
                                            servers.add(parts[0])
                                            tool_name = parts[1]
                                            if tool_name.endswith(".md"):
                                                tool_name = tool_name[:-3]
                                            tools.add(tool_name)
                            except (json.JSONDecodeError, KeyError):
                                pass

        if servers or tools:
            samples.append(
                {
                    "task_id": sample_id,
                    "score": score_value,
                    "servers_used": sorted(servers),
                    "tools_used": sorted(tools),
                }
            )

    return samples


def find_log_files(log_dir: Path) -> list[Path]:
    """Find all JSON log files in a directory."""
    if log_dir.is_file() and log_dir.suffix == ".json":
        return [log_dir]

    log_files = []
    for f in log_dir.rglob("*.json"):
        # Skip samples subdirectory files which are individual sample logs
        if "samples" not in f.parts:
            log_files.append(f)

    return sorted(log_files)


def merge_annotations(
    existing: dict[str, dict[str, Any]], new_samples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Merge new annotations into existing, keeping best scores."""
    for sample in new_samples:
        task_id = sample["task_id"]
        if task_id not in existing or sample["score"] > existing[task_id].get(
            "score", 0
        ):
            existing[task_id] = {
                "servers_used": sample["servers_used"],
                "tools_used": sample["tools_used"],
                "score": sample["score"],
            }
        else:
            # Merge sets
            existing[task_id]["servers_used"] = sorted(
                set(existing[task_id]["servers_used"]) | set(sample["servers_used"])
            )
            existing[task_id]["tools_used"] = sorted(
                set(existing[task_id]["tools_used"]) | set(sample["tools_used"])
            )
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Extract tool usage from successful ProgressiveMCPBench runs"
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
        default=Path("success_annotations.json"),
        help="Output file for annotations (default: success_annotations.json)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing output file if it exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"Error: Log path does not exist: {args.log_path}", file=sys.stderr)
        sys.exit(1)

    log_files = find_log_files(args.log_path)
    if not log_files:
        print(f"Error: No log files found in {args.log_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(log_files)} log file(s)")

    # Load existing annotations if merging
    annotations: dict[str, dict[str, Any]] = {}
    if args.merge and args.output.exists():
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            if args.verbose:
                print(f"Loaded {len(annotations)} existing annotations")
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load existing annotations: {e}", file=sys.stderr)

    # Process log files
    total_samples = 0
    for log_file in log_files:
        if args.verbose:
            print(f"Processing {log_file}")
        samples = parse_log_file(log_file)
        total_samples += len(samples)
        annotations = merge_annotations(annotations, samples)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(
        f"Extracted {total_samples} successful samples from {len(log_files)} log file(s)"
    )
    print(f"Total unique task annotations: {len(annotations)}")
    print(f"Output written to: {args.output}")

    # Summary statistics
    if annotations:
        all_servers = set()
        all_tools = set()
        for data in annotations.values():
            all_servers.update(data.get("servers_used", []))
            all_tools.update(data.get("tools_used", []))
        print(f"Unique servers: {len(all_servers)}")
        print(f"Unique tools: {len(all_tools)}")


if __name__ == "__main__":
    main()
