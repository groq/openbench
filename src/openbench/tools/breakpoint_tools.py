"""
Breakpoint evaluation tool for code submission and testing.
"""

import os
import re
from inspect_ai.tool import Tool, tool
from inspect_ai.util import sandbox, store


@tool
def submit_solution() -> Tool:
    """Submit a solution for testing.

    This tool:
    1. Takes the provided function code
    2. Inserts it into the target file using AST manipulation
    3. Runs pytest to check if tests pass
    4. Returns test results with pass/fail details
    5. Counts toward max_attempts budget (default: 4)
    """

    async def run(code: str) -> str:
        """Submit your function implementation for testing. Takes your complete function code, inserts it into the target file, runs tests, and returns results.

        Args:
            code: Complete function implementation to test

        Returns:
            Test results (pass/fail with detailed output)
        """
        # Use Inspect AI's store for persistent state across tool calls
        state = store()

        # Track attempts
        attempts = state.get("attempts", 0)
        # Default: 4 attempts as specified in Breakpoint paper (Section 3.2)
        max_attempts = state.get("max_attempts", 4)

        if attempts >= max_attempts:
            return f"❌ Maximum attempts ({max_attempts}) exhausted. No more submissions allowed."

        # Increment attempt counter
        attempts += 1
        state.set("attempts", attempts)

        # Get problem metadata (set by sample metadata)
        target_file = state.get("fpath", "")
        function_name = state.get("function_name", "")
        test_command = state.get("repo_test_command", "pytest")
        repo_name = state.get("repo_name", "unknown")

        if not target_file or not function_name:
            return "❌ Error: Missing target file or function name in metadata"

        # Find the repository directory dynamically
        # Agent should have cloned it, so we search for target_file
        find_result = await sandbox().exec(
            cmd=["find", ".", "-name", os.path.basename(target_file), "-type", "f"],
            timeout=10,  # Short timeout for filesystem search
        )

        if not find_result.success or not find_result.stdout.strip():
            return f"""❌ [Attempt {attempts}/{max_attempts}] Could not find target file: {target_file}

Make sure you have:
1. Cloned the repository
2. Checked out the correct commit
3. The file path matches: {target_file}

Use 'find . -name "{os.path.basename(target_file)}"' to locate the file."""

        # Get first match (should be the one in the cloned repo)
        all_matches = find_result.stdout.strip().split("\n")
        found_path = all_matches[0].strip()

        # Remove leading "./" if present
        if found_path.startswith("./"):
            found_path = found_path[2:]

        # found_path is like "repo_123/recordlinkage/algorithms/string.py"
        # target_file is like "recordlinkage/algorithms/string.py"
        # We need to find the repo root by removing the target_file suffix from found_path

        # Navigate up from found_path to find repo root
        # Count how many directory levels are in target_file
        target_parts = target_file.split("/")

        # Start from found_path and go up by the number of parts in target_file
        repo_root = found_path
        for _ in range(len(target_parts)):
            repo_root = os.path.dirname(repo_root)

        # If repo_root is empty, use current directory
        if not repo_root:
            repo_root = "."

        # Write code to temp file
        code_file = os.path.join(repo_root, ".breakpoint_solution.py")
        await sandbox().write_file(code_file, code.encode("utf-8"))

        # Inline AST manipulation utilities (to avoid import issues)
        # Use repr() for proper escaping to prevent syntax errors from special characters
        manipulation_script = f"""
import ast
import sys

def extract_function_info(source, function_name):
    \"\"\"Extract function metadata using AST.\"\"\"
    method_name = function_name.split(".")[-1] if "." in function_name else function_name
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

            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str) and
                node.body[0].end_lineno is not None):
                func_def_end = node.body[0].end_lineno

            return {{
                "func_start": decorator_start,
                "func_def_end": func_def_end,
                "node_end_lineno": node.end_lineno,
                "indent": indent,
            }}

    raise ValueError(f"Function '{{function_name}}' not found")

def apply_indentation(code, required_indent):
    \"\"\"Adjust code indentation.\"\"\"
    lines = code.split("\\n")
    min_indent = float("inf")

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

# Read new code
with open('.breakpoint_solution.py', 'r') as f:
    new_code = f.read()

# Read target file (use target_file which is relative to repo_root/cwd)
with open({repr(target_file)}, 'r') as f:
    source = f.read()

# Extract function info
info = extract_function_info(source, {repr(function_name)})

# Adjust indentation
adjusted_code = apply_indentation(new_code, info['indent'])

# Reconstruct file
lines = source.split("\\n")
new_lines = lines[:info['func_start']] + [adjusted_code] + lines[info['node_end_lineno']:]

# Write back
with open({repr(target_file)}, 'w') as f:
    f.write("\\n".join(new_lines))

print("Code inserted successfully")
"""

        # Execute manipulation script with cwd=repo_root
        # This makes all paths in the script relative to repo_root
        manip_result = await sandbox().exec(
            cmd=["python3", "-c", manipulation_script],
            cwd=repo_root,
            timeout=30,
        )

        if not manip_result.success:
            return f"❌ [Attempt {attempts}/{max_attempts}] Failed to insert code:\n{manip_result.stderr}"

        # Install dependencies if not already done
        # Create isolated venv per repository to prevent dependency conflicts
        deps_installed_key = f"deps_installed_{repo_root}"
        venv_path = os.path.join(repo_root, ".venv")
        install_notes = []

        if not state.get(deps_installed_key, False):
            # Check for pyproject.toml, setup.py, or requirements.txt
            check_files_result = await sandbox().exec(
                cmd=[
                    "bash",
                    "-c",
                    "ls pyproject.toml setup.py requirements.txt 2>/dev/null || true",
                ],
                cwd=repo_root,
                timeout=5,
            )

            files_found = (
                check_files_result.stdout.strip() if check_files_result.stdout else ""
            )

            if files_found:
                # Create isolated virtual environment for this repository
                venv_create_result = await sandbox().exec(
                    cmd=["python3", "-m", "venv", ".venv"],
                    cwd=repo_root,
                    timeout=60,
                )

                if not venv_create_result.success:
                    install_notes.append(
                        f"⚠ Failed to create venv: {venv_create_result.stderr[:200]}"
                    )
                else:
                    install_notes.append("✓ Created isolated venv")

                    # Install the package + pytest in the isolated venv
                    # pytest is always needed to run tests, and most test repos
                    # only include it as an optional dependency
                    install_cmd = ".venv/bin/pip install -e . pytest 2>&1"

                    install_result = await sandbox().exec(
                        cmd=["bash", "-c", install_cmd],
                        cwd=repo_root,
                        timeout=180,  # 3 minutes for dependency installation
                    )

                    # Capture installation output for debugging
                    install_output = install_result.stdout or ""
                    install_errors = install_result.stderr or ""

                    # Check if installation succeeded or had issues
                    if (
                        install_result.returncode == 0
                        or "Successfully installed" in install_output
                    ):
                        install_notes.append("✓ Dependencies installed in isolated venv")
                    else:
                        # Installation failed or partially failed
                        # Extract useful error info
                        error_lines = []
                        for line in (install_output + install_errors).split("\n"):
                            if any(
                                keyword in line.lower()
                                for keyword in [
                                    "error",
                                    "failed",
                                    "could not",
                                    "no module",
                                ]
                            ):
                                error_lines.append(line.strip())

                        if error_lines:
                            install_notes.append(
                                "⚠ Dependency installation issues:\n"
                                + "\n".join(error_lines[:5])
                            )
                        else:
                            install_notes.append(
                                f"⚠ Dependency installation returned code {install_result.returncode}"
                            )

                # Mark as installed (even if partial failure)
                state.set(deps_installed_key, True)

        # Run pytest using the isolated venv's pytest
        # This ensures tests run with the correct dependencies installed
        #
        # Check if isolated venv exists, otherwise fall back to system pytest
        venv_pytest = os.path.join(venv_path, "bin", "pytest")

        # Strip venv activation commands and path prefixes from test_command
        # since we'll use the venv's pytest directly
        clean_test_command = test_command
        if "source" in test_command and "activate" in test_command:
            # Remove "source venv/bin/activate &&" prefix
            clean_test_command = re.sub(
                r"source\s+[^\s]+/activate\s+&&\s+", "", test_command
            )
        # Also handle "./venv/bin/pytest" -> "pytest" and similar patterns
        clean_test_command = re.sub(
            r"\./[^\s]+/bin/(pytest|python)", r"\1", clean_test_command
        )

        # Use venv's pytest if it exists, otherwise use system pytest
        # Check if venv pytest exists
        check_venv_pytest = await sandbox().exec(
            cmd=["test", "-f", venv_pytest],
            cwd=repo_root,
            timeout=5,
        )

        if check_venv_pytest.success:
            # Replace 'pytest' with venv's pytest path
            if clean_test_command.startswith("pytest"):
                test_cmd = clean_test_command.replace("pytest", venv_pytest, 1)
            else:
                # If test command doesn't start with pytest, prepend venv path
                test_cmd = f"{venv_pytest} {clean_test_command}"
            test_cmd += " -v"
        else:
            # Fall back to system pytest (no venv created or pytest not in venv)
            test_cmd = f"{clean_test_command} -v"

        test_result = await sandbox().exec(
            cmd=["bash", "-c", test_cmd],
            cwd=repo_root,
            timeout=240,  # 4 minutes for test execution
        )

        # Parse pytest output directly
        # Exit code 0 = all passed, 1 = some failed, other = error
        output = test_result.stdout if test_result.stdout else ""

        # Count passed and failed from pytest output
        # Pytest shows: "X passed" and/or "X failed" in summary
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        # If pytest didn't run at all (missing deps), that's also a failure
        if test_result.returncode != 0 and passed == 0 and failed == 0:
            install_info = (
                "\n".join(install_notes)
                if install_notes
                else "No dependency installation attempted"
            )
            return f"""❌ [Attempt {attempts}/{max_attempts}] Tests failed to run.

{install_info}

Test command: {test_cmd}
Exit code: {test_result.returncode}

Output (first 1500 chars):
{output[:1500]}

Stderr (first 500 chars):
{test_result.stderr[:500] if test_result.stderr else "(empty)"}

Likely issues:
- Missing test dependencies
- Check installation notes above for dependency errors"""

        if failed == 0:
            return f"""✅ [Attempt {attempts}/{max_attempts}] SUCCESS! All tests passed!

Repository: {repo_name}
Function: {function_name}
Tests: {passed} passed, 0 failed

Your solution is correct!"""
        else:
            # Extract some failure details
            failure_preview = (
                test_result.stdout[:1500] if test_result.stdout else "No output"
            )

            return f"""❌ [Attempt {attempts}/{max_attempts}] Tests failed

Repository: {repo_name}
Function: {function_name}
Tests: {passed} passed, {failed} failed

Test output (first 1500 chars):
{failure_preview}

Try again with a different approach."""

    return run
