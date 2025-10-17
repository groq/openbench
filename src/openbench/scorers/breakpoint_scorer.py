"""
Breakpoint scorer for executing tests and calculating normalized scores.
"""

import os
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox
from openbench.utils.breakpoint_utils import (
    parse_pytest_json_report,
)


@scorer(metrics=[accuracy(), stderr()])
def breakpoint_scorer() -> Scorer:
    """
    Score function implementations by running pytest with binary scoring.

    Scoring formula (per paper): Binary score for whether all tests pass

    Returns:
    - 1.0 if all tests pass (perfect solution)
    - 0.0 otherwise (any failures or execution errors)
    """

    async def score(state: TaskState, target: Target) -> Score:
        # For agent-based tasks, check if submit_solution succeeded by parsing messages
        # Look for success indicator in tool responses
        if hasattr(state, "messages") and state.messages:
            for msg in state.messages:
                # Check tool responses for success message
                if (
                    hasattr(msg, "role")
                    and msg.role == "tool"
                    and hasattr(msg, "content")
                    and msg.content
                    and "✅" in msg.content
                    and "SUCCESS! All tests passed!" in msg.content
                ):
                    return Score(
                        value=1.0,
                        answer=state.output.completion[:500]
                        if state.output and state.output.completion
                        else "",
                        explanation="Success! Agent's submit_solution passed all tests.",
                    )

        # Get metadata (with type assertions)
        metadata = state.metadata or {}
        repo_url: str = metadata.get("repo_url", "")
        repo_commit: str | None = metadata.get("repo_commit")
        repo_name: str = metadata.get("repo_name", "repo")
        fpath: str = metadata.get("fpath", "")
        function_name: str = metadata.get("function_name", "")
        test_command: str = metadata.get("test_command", "pytest")
        baseline_failures: int = max(
            int(metadata.get("baseline_failures", 1)), 1
        )  # Ensure at least 1 to avoid division by zero
        parsed_code: str | None = metadata.get("parsed_function_code")

        # Check if we have the model's code
        if not parsed_code:
            return Score(
                value=0.0,
                answer=state.output.completion[:500],
                explanation="Agent did not successfully submit a solution, and no parseable code found",
            )

        # Type narrowing: parsed_code is now str (not None)
        assert parsed_code is not None

        # Create working directory
        work_dir = f"/tmp/breakpoint_{repo_name}_{os.urandom(8).hex()}"

        try:
            # Step 1: Clone repository
            clone_result = await sandbox().exec(
                cmd=["git", "clone", repo_url, work_dir],
                timeout=120,
            )
            if not clone_result.success:
                return Score(
                    value=0.0,
                    answer=parsed_code[:500],
                    explanation=f"Failed to clone repository: {clone_result.stderr}",
                )

            # Step 2: Checkout specific commit
            if repo_commit:
                checkout_result = await sandbox().exec(
                    cmd=["git", "checkout", repo_commit],
                    cwd=work_dir,
                    timeout=30,
                )
                if not checkout_result.success:
                    return Score(
                        value=0.0,
                        answer=parsed_code[:500],
                        explanation=f"Failed to checkout commit: {checkout_result.stderr}",
                    )

            # Step 3: Setup virtual environment
            venv_result = await sandbox().exec(
                cmd=["python3", "-m", "venv", "venv"],
                cwd=work_dir,
                timeout=60,
            )
            if not venv_result.success:
                return Score(
                    value=0.0,
                    answer=parsed_code[:500],
                    explanation=f"Failed to create venv: {venv_result.stderr}",
                )

            # Step 4: Install dependencies and pytest-reportlog
            install_commands = [
                # Upgrade pip
                "source venv/bin/activate && pip install --upgrade pip",
                # Install pytest-reportlog (required for JSON reports)
                "source venv/bin/activate && pip install pytest pytest-reportlog",
            ]

            # Check for requirements.txt or setup.py
            for install_file in ["requirements.txt", "setup.py", "pyproject.toml"]:
                check_result = await sandbox().exec(
                    cmd=["test", "-f", install_file],
                    cwd=work_dir,
                )
                if check_result.success:
                    if install_file == "requirements.txt":
                        install_commands.append(
                            "source venv/bin/activate && pip install -r requirements.txt"
                        )
                    elif install_file in ["setup.py", "pyproject.toml"]:
                        install_commands.append(
                            "source venv/bin/activate && pip install -e ."
                        )
                    break

            # Run installation
            for install_cmd in install_commands:
                install_result = await sandbox().exec(
                    cmd=["bash", "-c", install_cmd],
                    cwd=work_dir,
                    timeout=300,  # 5 minutes for dependency installation
                )
                if not install_result.success:
                    # Continue even if some installs fail - tests might still run
                    pass

            # Step 5: Get absolute path to target file
            abs_fpath = os.path.join(work_dir, fpath)

            # Step 6: Copy file to temp location for manipulation
            # (We can't use Python functions directly in sandbox, so we'll use shell commands)

            # Write the code to a file in the sandbox to avoid quoting issues
            code_file = os.path.join(work_dir, "new_function_code.py")
            await sandbox().write_file(code_file, parsed_code.encode("utf-8"))

            # Use repr() for proper Python string escaping
            # Inline utility functions to avoid import issues in sandbox
            manipulation_script = f"""import ast
from typing import Any

def extract_function_info(source: str, function_name: str) -> dict[str, Any]:
    \"\"\"Extract metadata about a function from source code using AST.\"\"\"
    method_name = (
        function_name.split(".")[-1] if "." in function_name else function_name
    )

    tree = ast.parse(source)
    lines = source.split("\\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            decorator_start = node.lineno - 1
            if node.decorator_list:
                decorator_start = node.decorator_list[0].lineno - 1

            func_def_line = lines[node.lineno - 1]
            indent = len(func_def_line) - len(func_def_line.lstrip())

            func_def_end = node.lineno

            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
                and node.body[0].end_lineno is not None
            ):
                func_def_end = node.body[0].end_lineno

            return {{
                "func_start": decorator_start,
                "func_def_end": func_def_end,
                "node_end_lineno": node.end_lineno,
                "indent": indent,
            }}

    raise ValueError(
        f"Function '{{function_name}}' not found in source "
        f"(searched for method name '{{method_name}}')"
    )

def remove_functions_in_file(file_path: str, function_to_remove: str) -> dict[str, Any]:
    \"\"\"Remove function body from file, replacing with 'pass'.\"\"\"
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.split("\\n")
    info = extract_function_info(source, function_to_remove)

    def_end = info["func_def_end"]
    node_end = info["node_end_lineno"]
    new_indent = " " * (info["indent"] + 4)
    pass_line = f"{{new_indent}}pass"

    new_lines = lines[:def_end] + [pass_line] + lines[node_end:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\\n".join(new_lines))

    return info

def apply_code_with_indentation(code: str, required_indent: int) -> str:
    \"\"\"Adjust code indentation to match required indent level.\"\"\"
    lines = code.split("\\n")

    min_indent: float = float("inf")
    for line in lines:
        if line.strip():
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)

    user_indent = 0 if min_indent == float("inf") else int(min_indent)
    indent_delta = required_indent - user_indent

    adjusted_lines = []
    for line in lines:
        if line.strip():
            if indent_delta >= 0:
                adjusted_lines.append(" " * indent_delta + line)
            else:
                spaces_to_remove = abs(indent_delta)
                if line[:spaces_to_remove].strip() == "":
                    adjusted_lines.append(line[spaces_to_remove:])
                else:
                    adjusted_lines.append(line)
        else:
            adjusted_lines.append(line)

    return "\\n".join(adjusted_lines)

def insert_function_code(
    new_function_code: str,
    start_line: int,
    end_skip_range: int,
    indent: int,
    path: str,
) -> None:
    \"\"\"Insert new function code into file at specified location.\"\"\"
    adjusted_fn = apply_code_with_indentation(new_function_code, indent)

    with open(path, "r", encoding="utf-8") as f:
        old_lines = f.readlines()

    new_lines = []
    for i, line in enumerate(old_lines):
        if i < start_line:
            new_lines.append(line)
        elif i == start_line:
            new_lines.append(adjusted_fn + "\\n")
        elif i <= end_skip_range:
            pass
        else:
            new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(new_lines))

# Read the new code from file
with open({repr(code_file)}, 'r', encoding='utf-8') as f:
    new_code = f.read()

info = remove_functions_in_file({repr(abs_fpath)}, {repr(function_name)})

# Use node_end_lineno - 1 for end_skip_range (convert 1-indexed to 0-indexed)
insert_function_code(
    new_code,
    info['func_start'],
    info['node_end_lineno'] - 1,
    info['indent'],
    {repr(abs_fpath)}
)

print("Code inserted successfully")
"""

            # Write script using bash
            script_path = os.path.join(work_dir, "manipulate_code.py")
            write_script_cmd = (
                f"cat > {script_path} << 'EOFPYTHON'\n{manipulation_script}\nEOFPYTHON"
            )

            write_result = await sandbox().exec(
                cmd=["bash", "-c", write_script_cmd],
                cwd=work_dir,
                timeout=10,
            )
            if not write_result.success:
                return Score(
                    value=0.0,
                    answer=parsed_code[:500],
                    explanation=f"Failed to write manipulation script: {write_result.stderr}",
                )

            # Run manipulation script
            manip_result = await sandbox().exec(
                cmd=[
                    "bash",
                    "-c",
                    "source venv/bin/activate && python3 manipulate_code.py",
                ],
                cwd=work_dir,
                timeout=30,
            )
            if not manip_result.success:
                return Score(
                    value=0.0,
                    answer=parsed_code[:500],
                    explanation=f"Failed to insert code: {manip_result.stderr}",
                )

            # Step 7: Run pytest with JSON report
            report_log = f"/tmp/pytest-report-{os.urandom(8).hex()}.jsonl"
            test_cmd = (
                f"source venv/bin/activate && {test_command} --report-log={report_log}"
            )

            test_result = await sandbox().exec(
                cmd=["bash", "-c", test_cmd],
                cwd=work_dir,
                timeout=240,  # 4 minutes for test execution
            )

            # Step 8: Read pytest JSON report
            read_report_result = await sandbox().exec(
                cmd=["cat", report_log],
                timeout=10,
            )
            if not read_report_result.success:
                return Score(
                    value=0.0,
                    answer=parsed_code[:500],
                    explanation=f"Failed to read pytest report: {read_report_result.stderr}",
                )

            report_content = read_report_result.stdout

            # Step 9: Parse pytest results
            test_output = parse_pytest_json_report(
                report_content,
                stderr=test_result.stderr,
            )

            # Step 10: Calculate score (binary: 1.0 if all pass, 0.0 otherwise)
            if test_output.get("had_execution_error"):
                score_value = 0.0
                explanation = f"Execution error: {test_output.get('error_message', 'Unknown error')}"
            else:
                current_failures = test_output.get("failed", 0)
                if current_failures == 0:
                    score_value = 1.0
                    explanation = f"Success! All tests passed ({test_output.get('passed', 0)} passed)"
                else:
                    score_value = 0.0
                    explanation = (
                        f"Failed: {current_failures} tests failed, "
                        f"{test_output.get('passed', 0)} passed (baseline: {baseline_failures} failures)"
                    )

                # Add failure details if available
                if test_output.get("failures_info"):
                    explanation += "\n\nFailure samples:\n" + "\n".join(
                        test_output["failures_info"][:2]
                    )

            # Cleanup
            await sandbox().exec(
                cmd=["rm", "-rf", work_dir, report_log],
                timeout=30,
            )

            return Score(
                value=score_value,
                answer=parsed_code[:500],
                explanation=explanation,
            )

        except Exception as e:
            # Cleanup on error
            await sandbox().exec(
                cmd=["rm", "-rf", work_dir],
                timeout=30,
            )
            return Score(
                value=0.0,
                answer=parsed_code[:500]
                if parsed_code
                else state.output.completion[:500],
                explanation=f"Scorer error: {str(e)}",
            )

    return score
