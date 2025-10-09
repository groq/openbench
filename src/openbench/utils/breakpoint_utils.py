"""
Breakpoint utilities for code manipulation and test execution.

Ported from https://github.com/Uzay-G/breakpoint
Key functions for AST-based code manipulation and pytest JSON report parsing.
"""

import ast
import json
import re
from typing import Any


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def extract_function_info(source: str, function_name: str) -> dict[str, Any]:
    """
    Extract metadata about a function from source code using AST.

    Args:
        source: Source code to parse
        function_name: Name of function to find. May include class prefix
                      (e.g., "Checker.checkDeadScopes" or "checkDeadScopes")

    Returns dict with:
    - func_start: Line where function/decorator starts (0-indexed)
    - func_def_end: Line where docstring ends (where body should start)
    - node_end_lineno: Original function end line
    - indent: Number of spaces for indentation
    """
    # Strip class prefix if present (e.g., "Checker.checkDeadScopes" → "checkDeadScopes")
    # This handles method names that include the class name in the dataset
    method_name = (
        function_name.split(".")[-1] if "." in function_name else function_name
    )

    tree = ast.parse(source)
    lines = source.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            # Find decorators
            decorator_start = node.lineno - 1  # 0-indexed
            if node.decorator_list:
                decorator_start = node.decorator_list[0].lineno - 1

            # Calculate indentation
            func_def_line = lines[node.lineno - 1]
            indent = len(func_def_line) - len(func_def_line.lstrip())

            # Find where function body starts (after docstring if present)
            func_def_end = node.lineno  # Line after "def function(...):"

            # Check if there's a docstring
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
                and node.body[0].end_lineno is not None
            ):
                # Skip past docstring
                func_def_end = node.body[0].end_lineno

            return {
                "func_start": decorator_start,
                "func_def_end": func_def_end,
                "node_end_lineno": node.end_lineno,
                "indent": indent,
            }

    # Use original function_name in error for clarity
    raise ValueError(
        f"Function '{function_name}' not found in source "
        f"(searched for method name '{method_name}')"
    )


def remove_functions_in_file(file_path: str, function_to_remove: str) -> dict[str, Any]:
    """
    Remove function body from file, replacing with 'pass'.

    Keeps:
    - Decorators
    - Function signature
    - Docstring

    Replaces body with 'pass'.

    Returns metadata needed for later insertion.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.split("\n")

    # Extract function metadata
    info = extract_function_info(source, function_to_remove)

    # Build new content: keep everything up to docstring end, add pass, skip rest
    def_end = info["func_def_end"]  # 1-indexed: line where docstring ends
    node_end = info["node_end_lineno"]  # 1-indexed: line where function ends
    new_indent = " " * (info["indent"] + 4)
    pass_line = f"{new_indent}pass"

    # Reconstruct file
    # AST line numbers are 1-indexed, but lines array is 0-indexed
    # To keep up to and including 1-indexed line N: use lines[:N]
    # To skip 1-indexed lines up to and including line N: use lines[N:]
    new_lines = lines[:def_end] + [pass_line] + lines[node_end:]

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    return info


def apply_code_with_indentation(code: str, required_indent: int) -> str:
    """
    Adjust code indentation to match required indent level.

    Detects the user's base indentation and adjusts all lines by the delta
    to match the required indentation.
    """
    lines = code.split("\n")

    # Find the minimum indentation in non-empty lines (user's base indent)
    min_indent: float = float("inf")
    for line in lines:
        if line.strip():  # Skip empty lines
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)

    user_indent = 0 if min_indent == float("inf") else int(min_indent)

    # Calculate delta
    indent_delta = required_indent - user_indent

    # Apply delta to all lines
    adjusted_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            if indent_delta >= 0:
                # Add spaces
                adjusted_lines.append(" " * indent_delta + line)
            else:
                # Remove spaces (dedent)
                spaces_to_remove = abs(indent_delta)
                if line[:spaces_to_remove].strip() == "":
                    adjusted_lines.append(line[spaces_to_remove:])
                else:
                    # Can't dedent that much, keep as is
                    adjusted_lines.append(line)
        else:
            # Empty line
            adjusted_lines.append(line)

    return "\n".join(adjusted_lines)


def insert_function_code(
    new_function_code: str,
    start_line: int,
    end_skip_range: int,
    indent: int,
    path: str,
) -> None:
    """
    Insert new function code into file at specified location.

    Args:
        new_function_code: The function body to insert
        start_line: Line number where function starts (0-indexed)
        end_skip_range: Line number where old function ends (0-indexed)
        indent: Required indentation level
        path: File path to modify
    """
    # Adjust indentation
    adjusted_fn = apply_code_with_indentation(new_function_code, indent)

    # Read existing file
    with open(path, "r", encoding="utf-8") as f:
        old_lines = f.readlines()

    # Reconstruct: lines before + new code + lines after
    new_lines = []
    for i, line in enumerate(old_lines):
        if i == start_line:
            # Insert new function code
            new_lines.append(adjusted_fn + "\n")
        elif start_line <= i <= end_skip_range:
            # Skip old function lines
            pass
        else:
            # Keep other lines
            new_lines.append(line)

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(new_lines))


def parse_pytest_json_report(log: str, stderr: str | None = None) -> dict[str, Any]:
    """
    Parse pytest JSON report log (from --report-log flag).

    Returns:
        Dict with:
        - success: bool - all tests passed
        - failed: int - number of failed tests
        - passed: int - number of passed tests
        - failures_info: list[str] - detailed failure messages
        - had_execution_error: bool - syntax/import errors
        - error_message: str - error details if execution failed
        - failed_tests: list[tuple] - (test_file, test_function) for failures
    """
    failed = 0
    passed = 0
    failures_info = []
    had_execution_error = False
    error_message = ""
    failed_tests = []
    detailed_failures = 3  # Number of detailed failures to include

    # Check stderr for execution errors
    if stderr:
        had_execution_error = True
        error_message = strip_ansi(stderr)

    # Parse each JSON line
    for line in log.splitlines():
        if not line.strip():
            continue

        try:
            entry = json.loads(line)

            # Process test results
            if entry.get("$report_type") == "TestReport":
                if entry.get("outcome") == "failed":
                    failed += 1

                    # Extract failure details for first N failures
                    if failed <= detailed_failures:
                        test_name = entry.get("nodeid", "")

                        # Parse nodeid: "tests/test_file.py::test_function"
                        parts = test_name.split("::")
                        test_file = parts[0] if len(parts) > 0 else ""
                        test_function = parts[-1] if len(parts) > 1 else ""
                        failed_tests.append((test_file, test_function))

                        # Extract error details from longrepr
                        if "longrepr" in entry:
                            longrepr = entry["longrepr"]
                            if isinstance(longrepr, dict):
                                crash_info = longrepr.get("reprcrash", {})
                                error_path = crash_info.get("path", "")
                                error_line = crash_info.get("lineno", "")
                                error_msg = crash_info.get("message", "")

                                failure_detail = f"Test: {test_name}\n"
                                failure_detail += (
                                    f"Location: {error_path}:{error_line}\n"
                                )
                                failure_detail += f"Error: {error_msg}\n"
                            else:
                                # String representation - take last 50 lines
                                failure_detail = "\n".join(
                                    str(longrepr).split("\n")[-50:]
                                )

                            failures_info.append(failure_detail)

                elif entry.get("outcome") == "passed":
                    passed += 1

            # Check for collection errors
            elif entry.get("$report_type") == "CollectReport":
                if entry.get("outcome") == "failed":
                    had_execution_error = True
                    if "longrepr" in entry:
                        longrepr = entry["longrepr"]
                        if isinstance(longrepr, dict):
                            error_message = longrepr.get("reprcrash", {}).get(
                                "message", ""
                            )
                        else:
                            error_message = str(longrepr)

        except json.JSONDecodeError:
            continue

    success = failed == 0 and not had_execution_error

    return {
        "success": success,
        "failed": failed,
        "passed": passed,
        "failures_info": failures_info,
        "had_execution_error": had_execution_error,
        "error_message": error_message,
        "failed_tests": failed_tests,
    }


def parse_function_from_completion(completion: str) -> str | None:
    """
    Extract function code from LLM completion.

    Looks for:
    1. Python code blocks (```python ... ```)
    2. Generic code blocks (``` ... ```)
    3. Raw function definitions (def ...)

    Returns the function code or None if not found.
    """
    # Try to find code blocks first
    code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(code_block_pattern, completion, re.DOTALL)

    if matches:
        # Return the last code block (most likely the final answer)
        return matches[-1].strip()

    # Try to find function definition directly
    if "def " in completion:
        # Find the function definition and extract it
        lines = completion.split("\n")
        func_lines = []
        in_function = False

        for line in lines:
            if line.strip().startswith("def "):
                in_function = True

            if in_function:
                func_lines.append(line)

        if func_lines:
            return "\n".join(func_lines)

    return None
