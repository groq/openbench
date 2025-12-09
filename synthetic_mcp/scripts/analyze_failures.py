#!/usr/bin/env python3
"""
Analyze failure modes in ProgressiveMCPBench evaluation logs.

This script:
1. Parses all progressivemcpbench .eval files (including "invalid" runs)
2. Extracts failed samples (score == 0.0)
3. Uses gpt-oss-120b on Groq to categorize failure modes
4. Produces a summary report of all failure types

Usage:
    # Analyze all progressivemcpbench logs
    python scripts/analyze_failures.py

    # Analyze a specific log file
    python scripts/analyze_failures.py --log-file logs/my-evaluation.eval

    # Limit to specific number of failures per run
    python scripts/analyze_failures.py --max-per-run 5

    # Skip LLM categorization (just extract failures)
    python scripts/analyze_failures.py --no-llm

    # Output to specific file
    python scripts/analyze_failures.py -o temp/my-analysis.json

Environment:
    GROQ_API_KEY must be set for LLM categorization

Output:
    Writes JSON analysis to temp/failure-analysis.json by default
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Groq client for LLM categorization
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None  # type: ignore


@dataclass
class FailedSample:
    """Represents a failed sample from an evaluation."""

    eval_file: str
    model: str
    strategy: str
    task: str
    sample_id: str
    epoch: int
    input_prompt: str
    target: list[str]
    final_answer: str | None
    tool_calls_made: list[dict[str, Any]]
    error_messages: list[str]
    timeout: bool
    score: float
    category: str | None  # From task metadata

    # LLM-assigned categorization
    failure_category: str | None = None
    failure_explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_file": self.eval_file,
            "model": self.model,
            "strategy": self.strategy,
            "task": self.task,
            "sample_id": self.sample_id,
            "epoch": self.epoch,
            "input_prompt": self.input_prompt,
            "target": self.target,
            "final_answer": self.final_answer,
            "tool_calls_made": self.tool_calls_made,
            "error_messages": self.error_messages,
            "timeout": self.timeout,
            "score": self.score,
            "category": self.category,
            "failure_category": self.failure_category,
            "failure_explanation": self.failure_explanation,
        }


def extract_tool_calls(messages: list[dict]) -> list[dict[str, Any]]:
    """Extract tool calls from message history."""
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(
                    {
                        "function": tc.get("function"),
                        "arguments": tc.get("arguments"),
                    }
                )
        if msg.get("role") == "tool":
            # Include tool responses
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            tool_calls.append(
                {
                    "function": msg.get("function"),
                    "response_preview": content[:500] if content else None,
                }
            )
    return tool_calls


def extract_error_messages(messages: list[dict]) -> list[str]:
    """Extract error messages from message history."""
    errors = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        else:
            text = ""

        # Look for error patterns
        lower_text = text.lower()
        if any(
            pattern in lower_text
            for pattern in ["error:", "exception:", "timeout", "failed", "timed out"]
        ):
            errors.append(text[:500])
    return errors


def extract_final_answer(messages: list[dict]) -> str | None:
    """Extract the final answer from the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:1000]
            elif isinstance(content, list):
                texts = [c.get("text", "") for c in content if isinstance(c, dict)]
                return " ".join(texts)[:1000]
    return None


def check_timeout(sample: dict) -> bool:
    """Check if sample timed out."""
    # Check for timeout in various places
    if sample.get("error_retries"):
        for error in sample["error_retries"]:
            if "timeout" in str(error).lower():
                return True

    # Check total_time vs some threshold (e.g., > 600 seconds suggests timeout)
    total_time = sample.get("total_time", 0)
    if total_time and total_time > 600:
        return True

    return False


def parse_sample(
    sample: dict,
    eval_file: str,
    model: str,
    strategy: str,
    task: str,
) -> FailedSample | None:
    """Parse a sample and return FailedSample if it failed."""
    scores = sample.get("scores", {})
    scorer = scores.get("progressivemcpbench_scorer", {})
    score = scorer.get("value", 1.0)  # Assume pass if no score

    # Only process failures
    if score > 0.0:
        return None

    messages = sample.get("messages", [])

    return FailedSample(
        eval_file=eval_file,
        model=model,
        strategy=strategy,
        task=task,
        sample_id=sample.get("id", "unknown"),
        epoch=sample.get("epoch", 0),
        input_prompt=sample.get("input", "")[:500],
        target=sample.get("target", []),
        final_answer=extract_final_answer(messages),
        tool_calls_made=extract_tool_calls(messages),
        error_messages=extract_error_messages(messages),
        timeout=check_timeout(sample),
        score=score,
        category=sample.get("metadata", {}).get("category"),
    )


def parse_eval_file(eval_path: Path, verbose: bool = False) -> list[FailedSample]:
    """Parse an .eval file and extract all failed samples."""
    failures = []

    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
            # Get metadata
            start_data = {}
            try:
                with zf.open("_journal/start.json") as f:
                    start_data = json.load(f)
            except (KeyError, json.JSONDecodeError):
                pass

            eval_info = start_data.get("eval", {})
            model = eval_info.get("model", "unknown")
            task = eval_info.get("task", "unknown")

            # Extract strategy from task_args
            task_args = eval_info.get("task_args", {})
            strategy = task_args.get("strategy", "unknown")

            if verbose:
                print(f"  Model: {model}, Strategy: {strategy}")

            # Parse all sample files
            sample_files = [
                n
                for n in zf.namelist()
                if n.startswith("samples/") and n.endswith(".json")
            ]

            for sample_file in sample_files:
                try:
                    with zf.open(sample_file) as f:
                        sample = json.load(f)

                    failed = parse_sample(
                        sample,
                        eval_file=eval_path.name,
                        model=model,
                        strategy=strategy,
                        task=task,
                    )
                    if failed:
                        failures.append(failed)

                except (json.JSONDecodeError, KeyError) as e:
                    if verbose:
                        print(f"    Warning: Could not parse {sample_file}: {e}")

    except zipfile.BadZipFile as e:
        print(f"Warning: Could not open {eval_path}: {e}", file=sys.stderr)

    return failures


CATEGORIZATION_PROMPT = """You are analyzing a failed task from an AI agent benchmark that tests MCP (Model Context Protocol) tool usage.

The agent was given a task and had access to MCP tools. The task failed (score=0). Analyze WHY it failed.

## Task Details
- Model: {model}
- Strategy: {strategy}
- Task Category: {category}
- Input: {input_prompt}
- Expected Answer: {target}
- Agent's Answer: {final_answer}

## Tool Calls Made
{tool_calls}

## Error Messages Detected
{errors}

## Timeout
{timeout}

---

Categorize this failure into ONE of these categories:

1. **TOOL_NOT_FOUND** - Agent couldn't find the right tool via routing/discovery
2. **TOOL_CALL_ERROR** - Tool was found but returned an error when called
3. **TOOL_TIMEOUT** - Tool call timed out
4. **WRONG_TOOL_USED** - Agent used the wrong tool for the task
5. **INCOMPLETE_ANSWER** - Agent got partial info but didn't complete the task
6. **PARSING_ERROR** - Agent got correct data but couldn't parse/format the answer correctly
7. **REASONING_ERROR** - Agent made a logical/reasoning mistake
8. **NO_TOOLS_CALLED** - Agent didn't call any tools at all
9. **INFRASTRUCTURE_ERROR** - System/infrastructure failure (not agent's fault)
10. **GAVE_UP** - Agent explicitly stated it couldn't complete the task

Respond with JSON only:
{{
  "category": "<one of the categories above>",
  "explanation": "<1-2 sentence explanation of what went wrong>"
}}"""


async def categorize_failure(
    client: AsyncGroq, failure: FailedSample, semaphore: asyncio.Semaphore
) -> None:
    """Use LLM to categorize a single failure."""
    async with semaphore:
        tool_calls_str = (
            json.dumps(failure.tool_calls_made[:10], indent=2)
            if failure.tool_calls_made
            else "None"
        )
        errors_str = (
            "\n".join(failure.error_messages[:5]) if failure.error_messages else "None"
        )

        prompt = CATEGORIZATION_PROMPT.format(
            model=failure.model,
            strategy=failure.strategy,
            category=failure.category or "Unknown",
            input_prompt=failure.input_prompt,
            target=failure.target,
            final_answer=failure.final_answer or "None",
            tool_calls=tool_calls_str,
            errors=errors_str,
            timeout="Yes" if failure.timeout else "No",
        )

        try:
            response = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )

            content = response.choices[0].message.content
            if content:
                # Try to parse JSON from response
                try:
                    # Handle potential markdown code blocks
                    text = content.strip()
                    if "```" in text:
                        # Extract content between first ``` and next ```
                        parts = text.split("```")
                        if len(parts) >= 2:
                            text = parts[1]
                            # Remove optional language identifier
                            if text.startswith("json"):
                                text = text[4:]
                            elif text.startswith("\n"):
                                text = text[1:]
                    # Find JSON object in the text
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        text = text[start:end]
                    result = json.loads(text.strip())
                    failure.failure_category = result.get("category", "UNKNOWN")
                    failure.failure_explanation = result.get("explanation", "")
                except json.JSONDecodeError:
                    # Try to extract category with regex as fallback
                    match = re.search(r'"category"\s*:\s*"([^"]+)"', content)
                    if match:
                        failure.failure_category = match.group(1)
                        # Try to extract explanation too
                        exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)', content)
                        failure.failure_explanation = (
                            exp_match.group(1) if exp_match else content[:200]
                        )
                    else:
                        failure.failure_category = "PARSE_ERROR"
                        failure.failure_explanation = content[:200]
            else:
                failure.failure_category = "EMPTY_RESPONSE"
                failure.failure_explanation = "LLM returned empty response"

        except Exception as e:
            error_str = str(e)
            if "rate" in error_str.lower() or "limit" in error_str.lower():
                failure.failure_category = "RATE_LIMITED"
            else:
                failure.failure_category = "LLM_ERROR"
            failure.failure_explanation = error_str[:200]


async def categorize_failures(
    failures: list[FailedSample], concurrency: int = 10
) -> None:
    """Categorize all failures using LLM."""
    if AsyncGroq is None:
        print(
            "Error: groq package not installed. Run: pip install groq", file=sys.stderr
        )
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set", file=sys.stderr)
        return

    client = AsyncGroq(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    print(f"Categorizing {len(failures)} failures using gpt-oss-120b...")

    tasks = [categorize_failure(client, f, semaphore) for f in failures]

    # Process with progress
    completed = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        if completed % 50 == 0:
            print(f"  Progress: {completed}/{len(failures)}")

    print(f"  Done: {completed}/{len(failures)} categorized")


def generate_summary(failures: list[FailedSample]) -> dict[str, Any]:
    """Generate summary statistics from failures."""
    # Count by category
    category_counts: dict[str, int] = {}
    for f in failures:
        cat = f.failure_category or "UNCATEGORIZED"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Count by model
    model_counts: dict[str, int] = {}
    for f in failures:
        model_counts[f.model] = model_counts.get(f.model, 0) + 1

    # Count by strategy
    strategy_counts: dict[str, int] = {}
    for f in failures:
        strategy_counts[f.strategy] = strategy_counts.get(f.strategy, 0) + 1

    # Count by task category
    task_category_counts: dict[str, int] = {}
    for f in failures:
        tc = f.category or "Unknown"
        task_category_counts[tc] = task_category_counts.get(tc, 0) + 1

    # Infrastructure vs genuine failures
    infra_categories = {"TOOL_CALL_ERROR", "TOOL_TIMEOUT", "INFRASTRUCTURE_ERROR"}
    infra_failures = sum(1 for f in failures if f.failure_category in infra_categories)
    genuine_failures = len(failures) - infra_failures

    return {
        "total_failures": len(failures),
        "infrastructure_failures": infra_failures,
        "genuine_failures": genuine_failures,
        "by_category": dict(sorted(category_counts.items(), key=lambda x: -x[1])),
        "by_model": dict(sorted(model_counts.items(), key=lambda x: -x[1])),
        "by_strategy": dict(sorted(strategy_counts.items(), key=lambda x: -x[1])),
        "by_task_category": dict(
            sorted(task_category_counts.items(), key=lambda x: -x[1])
        ),
    }


def find_progressivemcpbench_logs(logs_dir: Path) -> list[Path]:
    """Find all progressivemcpbench .eval files."""
    return sorted(logs_dir.glob("*progressivemcpbench*.eval"))


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze failure modes in ProgressiveMCPBench logs"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing .eval files (default: logs)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Analyze a specific .eval file instead of all files in logs directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("temp/failure-analysis.json"),
        help="Output JSON file (default: temp/failure-analysis.json)",
    )
    parser.add_argument(
        "--max-per-run",
        type=int,
        default=None,
        help="Max failures to extract per eval file (for testing)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM categorization",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="LLM API concurrency (default: 10)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Determine which eval files to analyze
    if args.log_file:
        # Analyze only the specified file
        if not args.log_file.exists():
            print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
            sys.exit(1)
        eval_files = [args.log_file]
        print(f"Analyzing specific log file: {args.log_file.name}")
    else:
        # Find all eval files in logs directory
        eval_files = find_progressivemcpbench_logs(args.logs_dir)
        print(f"Found {len(eval_files)} progressivemcpbench eval files")

    # Extract failures from all files
    all_failures: list[FailedSample] = []
    for eval_file in eval_files:
        if args.verbose:
            print(f"Processing {eval_file.name}...")

        failures = parse_eval_file(eval_file, verbose=args.verbose)

        if args.max_per_run and len(failures) > args.max_per_run:
            failures = failures[: args.max_per_run]

        all_failures.extend(failures)

        if not args.verbose:
            print(f"  {eval_file.name}: {len(failures)} failures")

    print(f"\nTotal failures extracted: {len(all_failures)}")

    # Categorize with LLM
    if not args.no_llm and all_failures:
        await categorize_failures(all_failures, args.concurrency)

    # Generate summary
    summary = generate_summary(all_failures)

    # Build output
    output = {
        "generated": str(Path(__file__).name),
        "summary": summary,
        "failures": [f.to_dict() for f in all_failures],
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total failures: {summary['total_failures']}")
    print(f"Infrastructure failures: {summary['infrastructure_failures']}")
    print(f"Genuine failures: {summary['genuine_failures']}")

    print("\nBy Category:")
    for cat, count in summary["by_category"].items():
        pct = count / summary["total_failures"] * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print("\nBy Model:")
    for model, count in list(summary["by_model"].items())[:5]:
        print(f"  {model}: {count}")

    print("\nBy Strategy:")
    for strategy, count in summary["by_strategy"].items():
        print(f"  {strategy}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
