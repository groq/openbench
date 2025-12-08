#!/usr/bin/env python3
"""
Deterministic categorization of ProgressiveMCPBench evaluation results.

This script categorizes each task result into one of:
- SUCCESS: Model completed and answer was correct
- INCORRECT: Model completed but gave wrong answer
- GAVE_UP: Model explicitly declined to answer
- PARSING_ERROR: Model output couldn't be parsed (runtime_error)
- TOOL_VALIDATION_ERROR: Tool call validation failed
- API_ERROR: External API error (503, rate limits, etc.)
- SCHEMA_ERROR: JSON schema validation error
- UNKNOWN: Could not categorize (prints to console for analysis)

Usage:
    python scripts/categorize_results.py [--log-file FILE] [--logs-dir DIR]

No LLM dependencies - purely deterministic categorization.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskResult:
    """A single task result from an evaluation."""

    eval_file: str
    sample_id: str
    task_input: str
    target: str
    score: float
    answer: str
    grade: str | None
    reason: str | None
    execution_error: str | None
    error_message: str | None
    sample_error: str | None
    last_assistant_message: str | None
    model: str
    strategy: str
    category: str | None

    # Assigned by categorizer
    result_category: str = "UNKNOWN"


def extract_last_assistant_message(messages: list[dict]) -> str | None:
    """Extract the final assistant message content."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if content and len(content.strip()) > 0:
                return content.strip()
    return None


def parse_sample(sample: dict, eval_file: str, model: str, strategy: str) -> TaskResult:
    """Parse a sample into a TaskResult."""
    score_data = sample.get("scores", {}).get("progressivemcpbench_scorer", {})
    meta = score_data.get("metadata", {})
    dmeta = sample.get("metadata", {})

    messages = sample.get("messages", [])
    last_msg = extract_last_assistant_message(messages)

    error = sample.get("error")
    error_str = str(error) if error else None

    return TaskResult(
        eval_file=eval_file,
        sample_id=sample.get("id", "unknown"),
        task_input=sample.get("input", "")[:200],
        target=str(sample.get("target", "")),
        score=score_data.get("value", 0.0),
        answer=score_data.get("answer", ""),
        grade=meta.get("grade"),
        reason=meta.get("reason"),
        execution_error=dmeta.get("execution_error"),
        error_message=dmeta.get("error_message"),
        sample_error=error_str,
        last_assistant_message=last_msg,
        model=model,
        strategy=strategy,
        category=dmeta.get("category"),
    )


def categorize_result(result: TaskResult) -> str:
    """
    Deterministically categorize a task result.

    Returns one of the category strings.
    """
    # Success case
    if result.score > 0:
        return "SUCCESS"

    # Check for grade
    if result.grade == "correct":
        return "SUCCESS"

    if result.grade == "incorrect":
        return "INCORRECT"

    # Check last assistant message for specific patterns first
    last_msg = result.last_assistant_message or ""
    last_msg_lower = last_msg.lower()

    # Define gave-up phrases for reuse
    gave_up_phrases = [
        "i could not determine",
        "i couldn't determine",
        "unable to determine",
        "cannot determine",
        "i don't know",
        "i do not know",
        "i'm unable to",
        "i am unable to",
        "failed to find",
        "could not find",
        "couldn't find",
        "no answer available",
        "unable to find",
        "unable to complete",
        "cannot complete",
        "i give up",
        "was unable to retrieve",
        "unable to retrieve",
    ]

    # Check for "reduce length" error in last message (API rejected)
    if "reduce the length" in last_msg_lower or "please reduce" in last_msg_lower:
        return "API_ERROR_TOO_LONG"

    # Check error message patterns
    err_msg = result.error_message or ""
    err_msg_lower = err_msg.lower()

    # API/Infrastructure errors
    if "503" in err_msg or "service unavailable" in err_msg_lower:
        return "API_ERROR_503"

    if "400 invalid_argument" in err_msg_lower:
        return "API_ERROR_INVALID_ARG"

    if "connection closed" in err_msg_lower:
        return "API_ERROR_CONNECTION"

    if "413" in err_msg or "request entity too large" in err_msg_lower:
        return "API_ERROR_TOO_LARGE"

    if "reduce the length" in err_msg_lower:
        return "API_ERROR_TOO_LONG"

    if "internalservererror" in err_msg_lower:
        return "API_ERROR_INTERNAL"

    if "badrequesterror" in err_msg_lower and "invalid schema" in err_msg_lower:
        return "SCHEMA_ERROR"

    # Tool validation errors
    if "tool call validation failed" in err_msg_lower:
        return "TOOL_VALIDATION_ERROR"

    if "description not provided for parameter" in err_msg_lower:
        return "TOOL_SCHEMA_MISSING_DESC"

    if "failed to parse tool call arguments as json" in err_msg_lower:
        return "TOOL_ARGS_PARSE_ERROR"

    if "failed to call a function" in err_msg_lower:
        return "TOOL_CALL_FAILED"

    # Parsing errors
    if result.execution_error == "runtime_error":
        if "parsing failed" in err_msg_lower:
            return "PARSING_ERROR"
        return "RUNTIME_ERROR"

    # Sample-level errors (e.g., JSONSchema validation)
    if result.sample_error:
        if "jsonschema" in result.sample_error.lower():
            return "SCHEMA_ERROR"
        if "validation error" in result.sample_error.lower():
            return "VALIDATION_ERROR"

    # Check if model gave up explicitly
    if last_msg:
        for phrase in gave_up_phrases:
            if phrase in last_msg_lower:
                return "GAVE_UP"

    # Not attempted with no clear reason
    if result.grade == "not_attempted" or result.grade is None:
        if result.reason == "Failed to extract final_answer from output":
            if last_msg is None or last_msg == "":
                return "NO_OUTPUT"
            return "OUTPUT_PARSE_FAILED"
        # Model gave output but was marked not_attempted without a reason
        # This typically means the answer format was unexpected
        if last_msg and "final_answer" in last_msg:
            # Check if it's a gave-up response
            for phrase in gave_up_phrases:
                if phrase in last_msg_lower:
                    return "GAVE_UP"
            return "ANSWER_FORMAT_ISSUE"
        if last_msg is None or last_msg == "":
            return "NO_OUTPUT"
        return "NOT_ATTEMPTED"

    return "UNKNOWN"


def parse_eval_file(eval_path: Path, verbose: bool = False) -> list[TaskResult]:
    """Parse an .eval file and categorize all samples."""
    results: list[TaskResult] = []

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
            task_args = eval_info.get("task_args", {})
            strategy = task_args.get("strategy", "unknown")

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

                    result = parse_sample(
                        sample,
                        eval_file=eval_path.name,
                        model=model,
                        strategy=strategy,
                    )
                    result.result_category = categorize_result(result)
                    results.append(result)

                except (json.JSONDecodeError, KeyError) as e:
                    if verbose:
                        print(f"  Warning: Could not parse {sample_file}: {e}")

    except zipfile.BadZipFile as e:
        print(f"Warning: Could not open {eval_path}: {e}", file=sys.stderr)

    return results


def find_eval_files(
    logs_dir: Path, pattern: str = "*progressivemcpbench*.eval"
) -> list[Path]:
    """Find all matching .eval files."""
    return sorted(logs_dir.glob(pattern))


def print_summary(results: list[TaskResult], verbose: bool = False) -> None:
    """Print summary statistics."""
    category_counts: Counter[str] = Counter()
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_results: list[TaskResult] = []

    for r in results:
        category_counts[r.result_category] += 1
        by_model[r.model][r.result_category] += 1
        by_strategy[r.strategy][r.result_category] += 1
        if r.result_category == "UNKNOWN":
            unknown_results.append(r)

    total = len(results)

    print("=" * 70)
    print("RESULT CATEGORIZATION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {total}")
    print()

    print("BY CATEGORY:")
    for cat, count in category_counts.most_common():
        pct = count / total * 100 if total else 0
        print(f"  {cat:30} {count:5} ({pct:5.1f}%)")
    print()

    # Group categories for high-level summary
    success = category_counts.get("SUCCESS", 0)
    incorrect = category_counts.get("INCORRECT", 0)
    gave_up = category_counts.get("GAVE_UP", 0)
    infra_errors = sum(
        category_counts.get(c, 0)
        for c in [
            "API_ERROR_503",
            "API_ERROR_INTERNAL",
            "API_ERROR_CONNECTION",
            "API_ERROR_TOO_LARGE",
            "API_ERROR_TOO_LONG",
            "SCHEMA_ERROR",
            "VALIDATION_ERROR",
        ]
    )
    tool_errors = sum(
        category_counts.get(c, 0)
        for c in [
            "TOOL_VALIDATION_ERROR",
            "TOOL_SCHEMA_MISSING_DESC",
            "TOOL_ARGS_PARSE_ERROR",
            "TOOL_CALL_FAILED",
            "API_ERROR_INVALID_ARG",
        ]
    )
    output_errors = sum(
        category_counts.get(c, 0)
        for c in [
            "PARSING_ERROR",
            "RUNTIME_ERROR",
            "OUTPUT_PARSE_FAILED",
            "NO_OUTPUT",
            "ANSWER_FORMAT_ISSUE",
        ]
    )
    not_attempted = category_counts.get("NOT_ATTEMPTED", 0)
    other = (
        total
        - success
        - incorrect
        - gave_up
        - infra_errors
        - tool_errors
        - output_errors
        - not_attempted
    )

    print("HIGH-LEVEL SUMMARY:")
    print(f"  {'SUCCESS':30} {success:5} ({success / total * 100:5.1f}%)")
    print(f"  {'INCORRECT':30} {incorrect:5} ({incorrect / total * 100:5.1f}%)")
    print(f"  {'GAVE_UP':30} {gave_up:5} ({gave_up / total * 100:5.1f}%)")
    print(
        f"  {'Infrastructure Errors':30} {infra_errors:5} ({infra_errors / total * 100:5.1f}%)"
    )
    print(f"  {'Tool Errors':30} {tool_errors:5} ({tool_errors / total * 100:5.1f}%)")
    print(
        f"  {'Output/Parsing Errors':30} {output_errors:5} ({output_errors / total * 100:5.1f}%)"
    )
    if not_attempted:
        print(
            f"  {'NOT_ATTEMPTED':30} {not_attempted:5} ({not_attempted / total * 100:5.1f}%)"
        )
    if other:
        print(f"  {'Other/Unknown':30} {other:5} ({other / total * 100:5.1f}%)")
    print()

    if verbose and by_strategy:
        print("BY STRATEGY:")
        for strategy, counts in sorted(by_strategy.items()):
            strat_total = sum(counts.values())
            success_rate = (
                counts.get("SUCCESS", 0) / strat_total * 100 if strat_total else 0
            )
            print(f"  {strategy}: {strat_total} samples, {success_rate:.1f}% success")
        print()

    # Print unknown cases for investigation
    if unknown_results:
        print("=" * 70)
        print(f"UNKNOWN CASES ({len(unknown_results)} samples) - need investigation:")
        print("=" * 70)
        for r in unknown_results[:20]:
            print(f"\n  File: {r.eval_file}")
            print(f"  Sample: {r.sample_id}")
            print(f"  Grade: {r.grade}, Reason: {r.reason}")
            print(f"  Exec Error: {r.execution_error}")
            print(f"  Error Msg: {r.error_message[:100] if r.error_message else None}")
            print(f"  Sample Error: {r.sample_error[:100] if r.sample_error else None}")
            print(
                f"  Last Msg: {r.last_assistant_message[:150] if r.last_assistant_message else None!r}"
            )

        if len(unknown_results) > 20:
            print(f"\n  ... and {len(unknown_results) - 20} more unknown cases")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Categorize ProgressiveMCPBench evaluation results"
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
        help="Analyze a specific .eval file",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*progressivemcpbench*.eval",
        help="Glob pattern for finding eval files (default: *progressivemcpbench*.eval)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON file for detailed results",
    )

    args = parser.parse_args()

    # Find eval files
    if args.log_file:
        if not args.log_file.exists():
            print(f"Error: File not found: {args.log_file}", file=sys.stderr)
            sys.exit(1)
        eval_files = [args.log_file]
    else:
        eval_files = find_eval_files(args.logs_dir, args.pattern)

    print(f"Found {len(eval_files)} eval files")

    # Parse and categorize
    all_results: list[TaskResult] = []
    for eval_file in eval_files:
        if args.verbose:
            print(f"Processing {eval_file.name}...")
        results = parse_eval_file(eval_file, verbose=args.verbose)
        all_results.extend(results)
        if not args.verbose:
            success = sum(1 for r in results if r.result_category == "SUCCESS")
            print(f"  {eval_file.name}: {len(results)} samples, {success} success")

    print(f"\nTotal samples: {len(all_results)}")
    print()

    # Print summary
    print_summary(all_results, verbose=args.verbose)

    # Write detailed output if requested
    if args.output:
        output_data = {
            "total_samples": len(all_results),
            "results": [
                {
                    "eval_file": r.eval_file,
                    "sample_id": r.sample_id,
                    "category": r.result_category,
                    "score": r.score,
                    "grade": r.grade,
                    "model": r.model,
                    "strategy": r.strategy,
                    "task_category": r.category,
                }
                for r in all_results
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to: {args.output}")


if __name__ == "__main__":
    main()
