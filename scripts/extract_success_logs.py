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
import zipfile
from pathlib import Path
from typing import Any


def extract_tool_calls_from_output(output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from the model's JSON output."""
    tool_calls: list[dict[str, Any]] = []

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


def extract_from_messages(messages: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Extract server and tool names from message tool calls.

    Only extracts actual MCP tool calls (meta__execute-tool), not meta-tools
    like meta__ls, meta__read-tool-file, or meta__route.

    Also handles minimal strategy tools which are prefixed with server names
    (e.g., 'filesystem__read_file').
    """
    servers: set[str] = set()
    tools: set[str] = set()

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        for tc in msg.get("tool_calls", []):
            func_name = tc.get("function", "")
            args = tc.get("arguments", {})

            # Handle meta__execute-tool calls (copilot and directory strategies)
            if func_name == "meta__execute-tool":
                tool_path = args.get("tool_path", "")
                if tool_path.startswith("/tools/"):
                    parts = tool_path[7:].split("/")
                    if len(parts) >= 2:
                        servers.add(parts[0])
                        tool_name = parts[1]
                        if tool_name.endswith(".md"):
                            tool_name = tool_name[:-3]
                        tools.add(tool_name)

                # Also handle copilot-style meta__execute-tool (with server_name, tool_name)
                if "server_name" in args:
                    servers.add(args["server_name"])
                if "tool_name" in args:
                    tools.add(args["tool_name"])

            # Handle minimal strategy tools (prefixed with server_name__)
            # e.g., 'filesystem__read_file' -> server='filesystem', tool='read_file'
            elif "__" in func_name and not func_name.startswith("meta__"):
                parts = func_name.split("__", 1)
                if len(parts) == 2:
                    server_name, tool_name = parts
                    servers.add(server_name)
                    tools.add(tool_name)

    return servers, tools


def parse_sample_json(sample_data: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single sample JSON and extract tool usage if successful."""
    sample_id = sample_data.get("id", "")

    # Get score from the scores dict
    scores = sample_data.get("scores", {})
    scorer_data = scores.get("progressivemcpbench_scorer", {})
    score_value = scorer_data.get("value", 0)

    # Only process successful samples (score >= 0.5 for partial credit)
    if score_value < 0.5:
        return None

    # Extract tool calls from messages - this is the authoritative source
    # We don't use the model's self-reported tool_calls from the output JSON
    # because the model sometimes makes mistakes in reporting
    messages = sample_data.get("messages", [])
    servers, tools = extract_from_messages(messages)

    if servers or tools:
        return {
            "task_id": sample_id,
            "score": score_value,
            "servers_used": sorted(servers),
            "tools_used": sorted(tools),
        }

    return None


def parse_eval_file(eval_file: Path, verbose: bool = False) -> list[dict[str, Any]]:
    """Parse an .eval file (ZIP archive) and extract sample data."""
    samples = []

    try:
        with zipfile.ZipFile(eval_file, "r") as zf:
            # Find all sample JSON files
            sample_files = [
                name
                for name in zf.namelist()
                if name.startswith("samples/") and name.endswith(".json")
            ]

            # Track best score per task_id (across epochs)
            best_samples: dict[str, dict[str, Any]] = {}

            for sample_file in sample_files:
                try:
                    with zf.open(sample_file) as f:
                        sample_data = json.load(f)
                        result = parse_sample_json(sample_data)
                        if result:
                            task_id = result["task_id"]
                            # Keep the best scoring run for each task
                            if (
                                task_id not in best_samples
                                or result["score"] > best_samples[task_id]["score"]
                            ):
                                best_samples[task_id] = result
                            else:
                                # Merge tools from this run
                                existing = best_samples[task_id]
                                existing["servers_used"] = sorted(
                                    set(existing["servers_used"])
                                    | set(result["servers_used"])
                                )
                                existing["tools_used"] = sorted(
                                    set(existing["tools_used"])
                                    | set(result["tools_used"])
                                )
                except (json.JSONDecodeError, KeyError) as e:
                    if verbose:
                        print(
                            f"Warning: Could not parse {sample_file}: {e}",
                            file=sys.stderr,
                        )

            samples = list(best_samples.values())

    except zipfile.BadZipFile as e:
        print(f"Warning: Could not read ZIP file {eval_file}: {e}", file=sys.stderr)
    except OSError as e:
        print(f"Warning: Could not open {eval_file}: {e}", file=sys.stderr)

    return samples


def parse_log_file(log_file: Path, verbose: bool = False) -> list[dict[str, Any]]:
    """Parse a log file (.eval ZIP or .json) and extract sample data."""
    if log_file.suffix == ".eval":
        return parse_eval_file(log_file, verbose)

    # Fall back to JSON parsing for .json files
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

        if score_value < 0.5:
            continue

        output = sample.get("output", {})
        tool_calls = extract_tool_calls_from_output(output)
        servers, tools = extract_server_and_tool_names(tool_calls)

        messages = sample.get("messages", [])
        msg_servers, msg_tools = extract_from_messages(messages)
        servers.update(msg_servers)
        tools.update(msg_tools)

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
    """Find all log files (.eval or .json) in a directory."""
    if log_dir.is_file():
        if log_dir.suffix in (".eval", ".json"):
            return [log_dir]
        return []

    log_files = []
    # Find .eval files (ZIP archives)
    for f in log_dir.rglob("*.eval"):
        log_files.append(f)
    # Also find .json files (legacy format)
    for f in log_dir.rglob("*.json"):
        # Skip samples subdirectory files
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
        samples = parse_log_file(log_file, verbose=args.verbose)
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
