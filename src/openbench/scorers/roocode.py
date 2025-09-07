"""
Scorer for Roo-Code-Evals tasks.

This scorer evaluates the results of code execution from the roocode_solver,
parsing test results to determine task success/failure and providing detailed feedback.
"""

import re
from typing import Optional, Dict, Any

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    CORRECT,
    INCORRECT,
    stderr,
    std,
    scorer,
)
from inspect_ai.solver import TaskState
from openbench.metrics.grouped import grouped


def extract_execution_results(completion: str) -> Optional[Dict[str, Any]]:
    """
    Extract execution results from the solver's completion output.

    Args:
        completion: The full completion text from the roocode_agent_solver

    Returns:
        Dictionary containing parsed execution results, or None if not found
    """
    # Look for the final test results section added by the agent solver
    results_pattern = r"\[FINAL_TEST_RESULTS\]\s*\n(.*?)(?:\n\n|\Z)"
    match = re.search(results_pattern, completion, re.DOTALL)

    if not match:
        return None

    results_text = match.group(1).strip()

    # Parse the execution results
    result = {
        "exit_code": None,
        "success": False,
        "stdout": "",
        "stderr": "",
        "raw_output": results_text,
    }

    # Extract exit code
    exit_code_match = re.search(r"Exit Code:\s*(\d+)", results_text)
    if exit_code_match:
        result["exit_code"] = int(exit_code_match.group(1))

    # Extract success status
    success_match = re.search(r"Success:\s*(True|False)", results_text)
    if success_match:
        result["success"] = success_match.group(1) == "True"

    # Extract stdout
    stdout_match = re.search(
        r"--- STDOUT ---\s*\n(.*?)(?=\n--- STDERR ---|$)", results_text, re.DOTALL
    )
    if stdout_match:
        result["stdout"] = stdout_match.group(1).strip()

    # Extract stderr
    stderr_match = re.search(r"--- STDERR ---\s*\n(.*?)$", results_text, re.DOTALL)
    if stderr_match:
        result["stderr"] = stderr_match.group(1).strip()

    return result


def analyze_test_results(
    execution_results: Dict[str, Any], language: str
) -> Dict[str, Any]:
    """
    Analyze test execution results to determine success and extract details.

    Args:
        execution_results: Parsed execution results from extract_execution_results
        language: Programming language of the task

    Returns:
        Dictionary with analysis results including success, test counts, errors, etc.
    """
    analysis: Dict[str, Any] = {
        "overall_success": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "compilation_error": False,
        "runtime_error": False,
        "timeout_error": False,
        "error_details": [],
        "test_details": [],
    }

    # Check for basic execution success
    if not execution_results["success"] or execution_results["exit_code"] != 0:
        analysis["overall_success"] = False
    else:
        analysis["overall_success"] = True

    stdout = execution_results["stdout"]
    stderr = execution_results["stderr"]

    # Check for timeout
    if "timed out" in stderr.lower():
        analysis["timeout_error"] = True
        analysis["error_details"].append("Execution timed out")

    # Language-specific test result parsing
    if language == "python":
        analysis.update(_analyze_python_tests(stdout, stderr))
    elif language == "go":
        analysis.update(_analyze_go_tests(stdout, stderr))
    elif language == "javascript":
        analysis.update(_analyze_javascript_tests(stdout, stderr))
    elif language == "java":
        analysis.update(_analyze_java_tests(stdout, stderr))
    elif language == "rust":
        analysis.update(_analyze_rust_tests(stdout, stderr))

    # If we couldn't parse specific test results, use general success/failure
    if analysis["tests_total"] == 0:
        if analysis["overall_success"]:
            analysis["tests_passed"] = 1
            analysis["tests_total"] = 1
        else:
            analysis["tests_failed"] = 1
            analysis["tests_total"] = 1

    return analysis


def _analyze_python_tests(stdout: str, stderr: str) -> Dict[str, Any]:
    """Analyze Python pytest output."""
    analysis: Dict[str, Any] = {
        "compilation_error": False,
        "runtime_error": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "error_details": [],
        "overall_success": True,
    }

    # Check for compilation/syntax errors
    if any(
        error in stderr for error in ["SyntaxError", "IndentationError", "ImportError"]
    ):
        analysis["compilation_error"] = True
        analysis["error_details"] = [stderr]

    # Parse pytest results
    # Look for patterns like "1 failed, 2 passed" or "3 passed"
    test_summary = re.search(r"(\d+) failed.*?(\d+) passed|(\d+) passed", stdout)
    if test_summary:
        if test_summary.group(1):  # Has failures
            analysis["tests_failed"] = int(test_summary.group(1))
            analysis["tests_passed"] = int(test_summary.group(2))
        else:  # Only passes
            analysis["tests_passed"] = int(test_summary.group(3))
            analysis["tests_failed"] = 0
        analysis["tests_total"] = analysis["tests_passed"] + analysis["tests_failed"]

    return analysis


def _analyze_go_tests(stdout: str, stderr: str) -> Dict[str, Any]:
    """Analyze Go test output."""
    analysis: Dict[str, Any] = {
        "compilation_error": False,
        "runtime_error": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "error_details": [],
        "overall_success": True,
    }

    # Check for compilation errors
    if "build failed" in stderr or "cannot find package" in stderr:
        analysis["compilation_error"] = True
        analysis["error_details"] = [stderr]

    # Parse go test results
    if "PASS" in stdout and "FAIL" not in stdout:
        analysis["tests_passed"] = 1
        analysis["tests_total"] = 1
    elif "FAIL" in stdout:
        analysis["tests_failed"] = 1
        analysis["tests_total"] = 1
        analysis["overall_success"] = False

    return analysis


def _analyze_javascript_tests(stdout: str, stderr: str) -> Dict[str, Any]:
    """Analyze JavaScript/Node.js test output."""
    analysis: Dict[str, Any] = {
        "compilation_error": False,
        "runtime_error": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "error_details": [],
        "overall_success": True,
    }

    # Check for syntax/import errors
    if any(
        error in stderr
        for error in ["SyntaxError", "Cannot find module", "ReferenceError"]
    ):
        analysis["compilation_error"] = True
        analysis["error_details"] = [stderr]

    # Parse test results (Jest, Mocha, etc.)
    # Look for patterns like "Tests: 1 failed, 2 passed" or "✓ 3 tests passed"
    if "passed" in stdout.lower() and "failed" not in stdout.lower():
        # All tests passed
        passed_match = re.search(r"(\d+).*?passed", stdout.lower())
        if passed_match:
            analysis["tests_passed"] = int(passed_match.group(1))
            analysis["tests_total"] = analysis["tests_passed"]
    elif "failed" in stdout.lower():
        analysis["overall_success"] = False
        analysis["tests_failed"] = 1
        analysis["tests_total"] = 1

    return analysis


def _analyze_java_tests(stdout: str, stderr: str) -> Dict[str, Any]:
    """Analyze Java/Gradle test output."""
    analysis: Dict[str, Any] = {
        "compilation_error": False,
        "runtime_error": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "error_details": [],
        "overall_success": True,
    }

    # Check for compilation errors
    if "BUILD FAILED" in stdout or "compilation failed" in stderr:
        analysis["compilation_error"] = True
        analysis["error_details"] = [stderr]

    # Parse Gradle test results
    if "BUILD SUCCESSFUL" in stdout:
        analysis["tests_passed"] = 1
        analysis["tests_total"] = 1
    elif "BUILD FAILED" in stdout:
        analysis["tests_failed"] = 1
        analysis["tests_total"] = 1
        analysis["overall_success"] = False

    return analysis


def _analyze_rust_tests(stdout: str, stderr: str) -> Dict[str, Any]:
    """Analyze Rust/Cargo test output."""
    analysis: Dict[str, Any] = {
        "compilation_error": False,
        "runtime_error": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_total": 0,
        "error_details": [],
        "overall_success": True,
    }

    # Check for compilation errors
    if "error[E" in stderr or "could not compile" in stderr:
        analysis["compilation_error"] = True
        analysis["error_details"] = [stderr]

    # Parse cargo test results
    test_result = re.search(r"test result: (\w+)\. (\d+) passed; (\d+) failed", stdout)
    if test_result:
        result_type = test_result.group(1)
        passed = int(test_result.group(2))
        failed = int(test_result.group(3))

        analysis["tests_passed"] = passed
        analysis["tests_failed"] = failed
        analysis["tests_total"] = passed + failed

        if result_type != "ok" or failed > 0:
            analysis["overall_success"] = False

    return analysis


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        std(),
        grouped(group_key="language", metric=[accuracy(), stderr(), std()]),
    ]
)
def roocode_scorer() -> Scorer:
    """
    Scorer for Roo-Code-Evals tasks.

    This scorer parses the execution results from the roocode_solver to determine
    if the coding task was completed successfully by analyzing test results.

    Returns:
        Scorer function for Roo-Code tasks
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Score a Roo-Code task based on test execution results.

        Args:
            state: Task state containing completion with execution results
            target: Target (not used for code execution tasks)

        Returns:
            Score indicating success/failure with detailed explanation
        """
        completion = state.output.completion
        language = state.metadata.get("language", "unknown")
        task_name = state.metadata.get("task_name", "unknown")

        # Check for execution error in the completion
        if "ERROR:" in completion and "[FINAL_TEST_RESULTS]" not in completion:
            error_match = re.search(r"ERROR: (.*?)(?:\n|$)", completion)
            error_msg = (
                error_match.group(1) if error_match else "Unknown execution error"
            )

            return Score(
                value=INCORRECT,
                answer=None,
                explanation=f"Execution failed for {language}/{task_name}: {error_msg}",
                metadata={
                    "language": language,
                    "task_name": task_name,
                    "error_type": "execution_error",
                    "error_message": error_msg,
                },
            )

        # Extract execution results from completion
        execution_results = extract_execution_results(completion)

        if not execution_results:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation=f"No final test results found for {language}/{task_name}. Agent may not have completed the task.",
                metadata={
                    "language": language,
                    "task_name": task_name,
                    "error_type": "no_results",
                },
            )

        # Analyze the test results
        analysis = analyze_test_results(execution_results, language)

        # Determine overall success
        task_success = (
            analysis["overall_success"]
            and analysis["tests_passed"] > 0
            and analysis["tests_failed"] == 0
            and not analysis["compilation_error"]
            and not analysis["timeout_error"]
        )

        # Build explanation
        explanation_parts = [f"Task: {language}/{task_name}"]

        if task_success:
            explanation_parts.append(
                f"All tests passed ({analysis['tests_passed']}/{analysis['tests_total']})"
            )
        else:
            if analysis["compilation_error"]:
                explanation_parts.append("Compilation/syntax error occurred")
            elif analysis["timeout_error"]:
                explanation_parts.append("Execution timed out")
            elif analysis["tests_failed"] > 0:
                explanation_parts.append(
                    f"Tests failed ({analysis['tests_failed']}/{analysis['tests_total']})"
                )
            else:
                explanation_parts.append("Task execution failed")

        # Add error details if available
        if analysis["error_details"]:
            explanation_parts.append("\nError details:")
            for error in analysis["error_details"][:3]:  # Limit to first 3 errors
                explanation_parts.append(
                    f"  • {error[:200]}..."
                )  # Truncate long errors

        # Add execution output for context (truncated)
        if execution_results["stdout"]:
            explanation_parts.append(
                f"\nStdout (first 300 chars):\n{execution_results['stdout'][:300]}..."
            )
        if execution_results["stderr"] and not analysis["compilation_error"]:
            explanation_parts.append(
                f"\nStderr (first 300 chars):\n{execution_results['stderr'][:300]}..."
            )

        return Score(
            value=CORRECT if task_success else INCORRECT,
            answer=f"{'PASS' if task_success else 'FAIL'}: {analysis['tests_passed']}/{analysis['tests_total']} tests passed",
            explanation="\n".join(explanation_parts),
            metadata={
                "language": language,
                "task_name": task_name,
                "tests_passed": analysis["tests_passed"],
                "tests_failed": analysis["tests_failed"],
                "tests_total": analysis["tests_total"],
                "compilation_error": analysis["compilation_error"],
                "timeout_error": analysis["timeout_error"],
                "exit_code": execution_results["exit_code"],
                "execution_success": execution_results["success"],
            },
        )

    return score
