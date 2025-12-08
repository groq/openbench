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

# ANSI color codes
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"


@dataclass
class EvalMetadata:
    """Metadata about an evaluation run."""

    eval_file: str
    model: str
    strategy: str
    task: str
    created: str
    total_samples: int


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

    # Additional fields for deeper error analysis
    message_count: int = 0
    tool_errors: list[str] | None = None

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


def extract_tool_errors(messages: list[dict]) -> list[str]:
    """Extract error messages from tool response messages."""
    errors = []
    for msg in messages:
        if msg.get("role") == "tool":
            error = msg.get("error")
            if error:
                if isinstance(error, dict):
                    err_msg = error.get("message", str(error))
                else:
                    err_msg = str(error)
                errors.append(err_msg)
    return errors


def parse_sample(sample: dict, eval_file: str, model: str, strategy: str) -> TaskResult:
    """Parse a sample into a TaskResult."""
    score_data = sample.get("scores", {}).get("progressivemcpbench_scorer", {})
    meta = score_data.get("metadata", {})
    dmeta = sample.get("metadata", {})

    messages = sample.get("messages", [])
    last_msg = extract_last_assistant_message(messages)
    tool_errors = extract_tool_errors(messages)

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
        message_count=len(messages),
        tool_errors=tool_errors if tool_errors else None,
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

    # Check for samples that never started (0 messages)
    if result.message_count == 0:
        return "NEVER_STARTED"

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

    # Check error message patterns (from metadata)
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

    # Tool validation errors from metadata error_message
    if "tool call validation failed" in err_msg_lower:
        # More specific categorization
        if (
            "attempted to call tool" in err_msg_lower
            and "not in request.tools" in err_msg_lower
        ):
            return "TOOL_NOT_AVAILABLE"
        if "parameters for tool" in err_msg_lower:
            return "TOOL_PARAM_VALIDATION_ERROR"
        return "TOOL_VALIDATION_ERROR"

    if "description not provided for parameter" in err_msg_lower:
        return "TOOL_SCHEMA_MISSING_DESC"

    if "failed to parse tool call arguments as json" in err_msg_lower:
        return "TOOL_ARGS_PARSE_ERROR"

    if "failed to call a function" in err_msg_lower:
        return "TOOL_CALL_FAILED"

    # Parsing errors from metadata
    if "parsing failed" in err_msg_lower:
        return "PARSING_ERROR"

    # Check for execution_error types
    if result.execution_error == "runtime_error":
        # Check for more specific patterns in error_message
        if err_msg:
            return "RUNTIME_ERROR"
        # Fall through to check tool errors

    if result.execution_error == "missing_annotation":
        return "MISSING_ANNOTATION"

    # Sample-level errors (e.g., JSONSchema validation)
    if result.sample_error:
        if "jsonschema" in result.sample_error.lower():
            return "SCHEMA_ERROR"
        if "validation error" in result.sample_error.lower():
            return "VALIDATION_ERROR"

    # Check tool errors from message history
    if result.tool_errors:
        tool_err_str = " ".join(result.tool_errors).lower()
        if "server not found" in tool_err_str:
            return "TOOL_SERVER_NOT_FOUND"
        if "robots.txt" in tool_err_str or "robot" in tool_err_str:
            return "TOOL_BLOCKED_BY_ROBOTS"
        if "ddg detected an anomaly" in tool_err_str:
            return "TOOL_RATE_LIMITED"
        if "access denied" in tool_err_str:
            return "TOOL_ACCESS_DENIED"
        if "enoent" in tool_err_str or "no such file" in tool_err_str:
            return "TOOL_FILE_NOT_FOUND"
        if "tool not found" in tool_err_str:
            return "TOOL_NOT_FOUND"
        # Generic tool error
        return "TOOL_ERROR"

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


def parse_eval_file(
    eval_path: Path, verbose: bool = False
) -> tuple[EvalMetadata | None, list[TaskResult]]:
    """Parse an .eval file and categorize all samples."""
    results: list[TaskResult] = []
    metadata: EvalMetadata | None = None

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
            task = eval_info.get("task", "unknown")
            created = eval_info.get("created", "unknown")
            dataset_info = eval_info.get("dataset", {})
            total_samples = dataset_info.get("samples", 0)

            metadata = EvalMetadata(
                eval_file=eval_path.name,
                model=model,
                strategy=strategy,
                task=task,
                created=created,
                total_samples=total_samples,
            )

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

    return metadata, results


def find_eval_files(
    logs_dir: Path, pattern: str = "*progressivemcpbench*.eval"
) -> list[Path]:
    """Find all matching .eval files."""
    return sorted(logs_dir.glob(pattern))


def print_header(metadata_list: list[EvalMetadata], results: list[TaskResult]) -> None:
    """Print the report header with eval info in green."""
    total = len(results)
    success = sum(1 for r in results if r.result_category == "SUCCESS")
    success_rate = success / total * 100 if total else 0

    print()
    print(f"{GREEN}{BOLD}{'═' * 70}{RESET}")
    print(f"{GREEN}{BOLD}  PROGRESSIVEMCPBENCH EVALUATION REPORT{RESET}")
    print(f"{GREEN}{BOLD}{'═' * 70}{RESET}")

    if len(metadata_list) == 1:
        m = metadata_list[0]
        print(f"{GREEN}  Eval:     {m.task}{RESET}")
        print(f"{GREEN}  Model:    {m.model}{RESET}")
        print(f"{GREEN}  Strategy: {m.strategy}{RESET}")
        print(f"{GREEN}  Date:     {m.created}{RESET}")
        print(f"{GREEN}  Samples:  {total}{RESET}")
        print(
            f"{GREEN}{BOLD}  Score:    {success_rate:.1f}% ({success}/{total} correct){RESET}"
        )
    else:
        models = set(m.model for m in metadata_list)
        strategies = set(m.strategy for m in metadata_list)
        dates = sorted(
            set(m.created[:10] for m in metadata_list if m.created != "unknown")
        )

        print(f"{GREEN}  Evals:      {len(metadata_list)} files{RESET}")
        print(f"{GREEN}  Models:     {', '.join(sorted(models))}{RESET}")
        print(f"{GREEN}  Strategies: {', '.join(sorted(strategies))}{RESET}")
        if dates:
            print(f"{GREEN}  Date Range: {dates[0]} to {dates[-1]}{RESET}")
        print(f"{GREEN}  Samples:    {total}{RESET}")
        print(
            f"{GREEN}{BOLD}  Score:      {success_rate:.1f}% ({success}/{total} correct){RESET}"
        )

    print(f"{GREEN}{BOLD}{'═' * 70}{RESET}")
    print()


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
            "NEVER_STARTED",
        ]
    )
    tool_errors = sum(
        category_counts.get(c, 0)
        for c in [
            "TOOL_VALIDATION_ERROR",
            "TOOL_SCHEMA_MISSING_DESC",
            "TOOL_ARGS_PARSE_ERROR",
            "TOOL_CALL_FAILED",
            "TOOL_NOT_AVAILABLE",
            "TOOL_PARAM_VALIDATION_ERROR",
            "TOOL_SERVER_NOT_FOUND",
            "TOOL_BLOCKED_BY_ROBOTS",
            "TOOL_RATE_LIMITED",
            "TOOL_ACCESS_DENIED",
            "TOOL_FILE_NOT_FOUND",
            "TOOL_NOT_FOUND",
            "TOOL_ERROR",
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
    missing_annotation = category_counts.get("MISSING_ANNOTATION", 0)
    other = (
        total
        - success
        - incorrect
        - gave_up
        - infra_errors
        - tool_errors
        - output_errors
        - not_attempted
        - missing_annotation
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
    if missing_annotation:
        print(
            f"  {'MISSING_ANNOTATION':30} {missing_annotation:5} ({missing_annotation / total * 100:5.1f}%)"
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


def print_verbose_details(results: list[TaskResult]) -> None:
    """Print detailed breakdown of each failure category."""
    # Group results by category
    by_category: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        if r.result_category != "SUCCESS":
            by_category[r.result_category].append(r)

    if not by_category:
        return

    print()
    print(f"{CYAN}{'=' * 70}{RESET}")
    print(f"{CYAN}{BOLD}DETAILED FAILURE BREAKDOWN{RESET}")
    print(f"{CYAN}{'=' * 70}{RESET}")

    # INCORRECT - show answer vs expected, deduped
    if "INCORRECT" in by_category:
        print(f"\n{YELLOW}▸ INCORRECT ({len(by_category['INCORRECT'])} cases){RESET}")
        print(f"  {DIM}Model gave wrong answer{RESET}")
        answer_pairs: Counter[tuple[str, str]] = Counter()
        for r in by_category["INCORRECT"]:
            answer_pairs[(r.answer[:80], r.target[:80])] += 1
        for (answer, expected), count in answer_pairs.most_common(15):
            print(f"\n  [{count}x] Got: {answer!r}")
            print(f"       Expected: {expected!r}")
        if len(answer_pairs) > 15:
            print(f"\n  ... and {len(answer_pairs) - 15} more distinct answer pairs")

    # GAVE_UP - show the messages
    if "GAVE_UP" in by_category:
        print(f"\n{YELLOW}▸ GAVE_UP ({len(by_category['GAVE_UP'])} cases){RESET}")
        print(f"  {DIM}Model declined to answer{RESET}")
        messages: Counter[str] = Counter()
        for r in by_category["GAVE_UP"]:
            msg = r.last_assistant_message or ""
            # Extract the "gave up" part
            msg_preview = msg[:100].replace("\n", " ")
            messages[msg_preview] += 1
        for msg, count in messages.most_common(10):
            print(f"\n  [{count}x] {msg!r}")

    # Tool errors - show specific error messages
    tool_cats = [
        "TOOL_VALIDATION_ERROR",
        "TOOL_SCHEMA_MISSING_DESC",
        "TOOL_ARGS_PARSE_ERROR",
        "TOOL_CALL_FAILED",
        "TOOL_NOT_AVAILABLE",
        "TOOL_PARAM_VALIDATION_ERROR",
        "TOOL_SERVER_NOT_FOUND",
        "TOOL_BLOCKED_BY_ROBOTS",
        "TOOL_RATE_LIMITED",
        "TOOL_ACCESS_DENIED",
        "TOOL_FILE_NOT_FOUND",
        "TOOL_NOT_FOUND",
        "TOOL_ERROR",
    ]
    for cat in tool_cats:
        if cat in by_category:
            print(f"\n{YELLOW}▸ {cat} ({len(by_category[cat])} cases){RESET}")
            tool_errors: Counter[str] = Counter()
            for r in by_category[cat]:
                # Use error_message if available, otherwise check tool_errors
                if r.error_message:
                    err = r.error_message[:120].replace("\n", " ")
                elif r.tool_errors:
                    err = " | ".join(r.tool_errors)[:120].replace("\n", " ")
                else:
                    err = "(no error details)"
                tool_errors[err] += 1
            for err, count in tool_errors.most_common(10):
                print(f"\n  [{count}x] {err}")

    # API errors
    api_cats = [
        "API_ERROR_503",
        "API_ERROR_INVALID_ARG",
        "API_ERROR_CONNECTION",
        "API_ERROR_TOO_LARGE",
        "API_ERROR_TOO_LONG",
        "API_ERROR_INTERNAL",
    ]
    for cat in api_cats:
        if cat in by_category:
            print(f"\n{YELLOW}▸ {cat} ({len(by_category[cat])} cases){RESET}")
            api_errors: Counter[str] = Counter()
            for r in by_category[cat]:
                err = (r.error_message or r.last_assistant_message or "")[:100].replace(
                    "\n", " "
                )
                api_errors[err] += 1
            for err, count in api_errors.most_common(5):
                print(f"\n  [{count}x] {err}")

    # Output/Parsing errors
    output_cats = [
        "OUTPUT_PARSE_FAILED",
        "NO_OUTPUT",
        "ANSWER_FORMAT_ISSUE",
        "PARSING_ERROR",
        "RUNTIME_ERROR",
    ]
    for cat in output_cats:
        if cat in by_category:
            print(f"\n{YELLOW}▸ {cat} ({len(by_category[cat])} cases){RESET}")
            if cat == "OUTPUT_PARSE_FAILED":
                print(
                    f"  {DIM}Model gave output but couldn't extract final_answer{RESET}"
                )
                samples: Counter[str] = Counter()
                for r in by_category[cat]:
                    msg = (r.last_assistant_message or "")[:80].replace("\n", " ")
                    samples[msg] += 1
                for msg, count in samples.most_common(10):
                    print(f"\n  [{count}x] {msg!r}")
            elif cat == "NO_OUTPUT":
                print(f"  {DIM}Model produced no output{RESET}")
                # Group by task input to see which tasks are problematic
                tasks: Counter[str] = Counter()
                for r in by_category[cat]:
                    tasks[r.task_input[:60]] += 1
                for task, count in tasks.most_common(10):
                    print(f"\n  [{count}x] Task: {task!r}")
            elif cat == "ANSWER_FORMAT_ISSUE":
                print(f"  {DIM}Model gave answer but format was unexpected{RESET}")
                format_samples: Counter[str] = Counter()
                for r in by_category[cat]:
                    msg = (r.last_assistant_message or "")[:80].replace("\n", " ")
                    format_samples[msg] += 1
                for msg, count in format_samples.most_common(10):
                    print(f"\n  [{count}x] {msg!r}")
            else:
                parse_errors: Counter[str] = Counter()
                for r in by_category[cat]:
                    err = (r.error_message or "")[:100].replace("\n", " ")
                    parse_errors[err] += 1
                for err, count in parse_errors.most_common(10):
                    print(f"\n  [{count}x] {err}")

    # Schema/Validation errors
    for cat in ["SCHEMA_ERROR", "VALIDATION_ERROR"]:
        if cat in by_category:
            print(f"\n{YELLOW}▸ {cat} ({len(by_category[cat])} cases){RESET}")
            schema_errors: Counter[str] = Counter()
            for r in by_category[cat]:
                err = (r.sample_error or r.error_message or "")[:100].replace("\n", " ")
                schema_errors[err] += 1
            for err, count in schema_errors.most_common(5):
                print(f"\n  [{count}x] {err}")

    # NEVER_STARTED
    if "NEVER_STARTED" in by_category:
        print(
            f"\n{YELLOW}▸ NEVER_STARTED ({len(by_category['NEVER_STARTED'])} cases){RESET}"
        )
        print(f"  {DIM}Sample had 0 messages - never executed{RESET}")
        tasks_ns: Counter[str] = Counter()
        for r in by_category["NEVER_STARTED"]:
            tasks_ns[r.task_input[:60]] += 1
        for task, count in tasks_ns.most_common(10):
            print(f"\n  [{count}x] Task: {task!r}")

    # MISSING_ANNOTATION
    if "MISSING_ANNOTATION" in by_category:
        print(
            f"\n{YELLOW}▸ MISSING_ANNOTATION ({len(by_category['MISSING_ANNOTATION'])} cases){RESET}"
        )
        print(f"  {DIM}Missing annotation data for scoring{RESET}")
        tasks_ma: Counter[str] = Counter()
        for r in by_category["MISSING_ANNOTATION"]:
            tasks_ma[r.task_input[:60]] += 1
        for task, count in tasks_ma.most_common(10):
            print(f"\n  [{count}x] Task: {task!r}")

    # NOT_ATTEMPTED
    if "NOT_ATTEMPTED" in by_category:
        print(
            f"\n{YELLOW}▸ NOT_ATTEMPTED ({len(by_category['NOT_ATTEMPTED'])} cases){RESET}"
        )
        print(f"  {DIM}Could not categorize these failures{RESET}")
        for r in by_category["NOT_ATTEMPTED"][:5]:
            print(f"\n  Sample: {r.sample_id}")
            print(f"  Task: {r.task_input[:60]!r}")
            print(f"  Last msg: {(r.last_assistant_message or '')[:80]!r}")

    print()


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
        help="Show detailed breakdown of each failure category",
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

    if not args.verbose:
        print(f"Found {len(eval_files)} eval files")

    # Parse and categorize
    all_results: list[TaskResult] = []
    all_metadata: list[EvalMetadata] = []

    for eval_file in eval_files:
        metadata, results = parse_eval_file(eval_file, verbose=args.verbose)
        if metadata:
            all_metadata.append(metadata)
        all_results.extend(results)
        if not args.verbose:
            success = sum(1 for r in results if r.result_category == "SUCCESS")
            print(f"  {eval_file.name}: {len(results)} samples, {success} success")

    if not args.verbose:
        print(f"\nTotal samples: {len(all_results)}")

    # Print header
    print_header(all_metadata, all_results)

    # Print summary
    print_summary(all_results, verbose=args.verbose)

    # Print detailed breakdown with -v
    if args.verbose:
        print_verbose_details(all_results)

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
