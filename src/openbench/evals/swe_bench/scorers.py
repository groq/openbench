"""
Scoring logic for SWE-Bench evaluation.
"""

import logging
import re
import shlex
from collections import Counter

from inspect_ai.scorer import Score, Scorer, Target, scorer, INCORRECT, CORRECT
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import ExecResult, sandbox

logger = logging.getLogger(__name__)

# Test execution timeout (30 minutes - conservative)
EVAL_SCRIPT_TIMEOUT = 60 * 30


@scorer(metrics=[accuracy(), stderr()])
def swe_bench_scorer() -> Scorer:
    """
    Scores changes made by a solver when solving a SWE-bench instance.
    Runs the test suite to check if the instance is correctly solved.
    """

    async def scorer_fn(state: TaskState, target: Target) -> Score:
        # Get the model's patch for logging
        await sandbox().exec(
            [
                "bash",
                "-c",
                create_model_patch_cmd(base_commit=state.metadata["base_commit"]),
            ]
        )

        try:
            agent_patch = await sandbox().exec(
                ["bash", "-c", "cat /testbed/model.patch"]
            )
        except UnicodeDecodeError:
            agent_patch = ExecResult(
                success=True,
                returncode=0,
                stdout="Agent patch could not be decoded due to binary content.",
                stderr="",
            )

        # Create and run evaluation script
        eval_script = create_eval_script(
            test_patch=state.metadata["test_patch"],
            repo=state.metadata["repo"],
            version=state.metadata["version"],
            base_commit=state.metadata["base_commit"],
        )

        logger.debug("Running evaluation script")
        try:
            eval_script_result = await sandbox().exec(
                ["bash", "-c", eval_script], timeout=EVAL_SCRIPT_TIMEOUT
            )
        except TimeoutError as e:
            explanation = (
                f"The evaluation script timed out after {EVAL_SCRIPT_TIMEOUT} seconds."
            )
            return Score(
                value=INCORRECT,
                explanation=explanation,
                metadata={
                    "model_patch": agent_patch.stdout,
                    "eval_script": eval_script,
                    "eval_script_timeout": str(e),
                },
            )

        if not eval_script_result.success:
            logger.error(f"Test run failed: {eval_script_result.stderr}")
            # Still try to parse results even if script returns non-zero

        # Parse test output
        value, explanation, pass_to_pass_results, fail_to_pass_results = (
            parse_test_output(
                eval_script_result.stdout + "\n" + eval_script_result.stderr, state
            )
        )

        return Score(
            value=value,
            explanation=explanation,
            metadata={
                "model_patch": agent_patch.stdout,
                "eval_script": eval_script,
                "test_results": {
                    "pass_to_pass": pass_to_pass_results,
                    "fail_to_pass": fail_to_pass_results,
                },
                "eval_script_result": {
                    "stdout": eval_script_result.stdout,
                    "stderr": eval_script_result.stderr,
                    "returncode": eval_script_result.returncode,
                },
            },
        )

    return scorer_fn


def parse_test_output(
    test_output: str, state: TaskState
) -> tuple[str, str, dict, dict]:
    """
    Parse the test output to determine if the issue was solved correctly.

    Returns:
        Tuple of (score_value, explanation, pass_to_pass_results, fail_to_pass_results)
    """
    # Try to import swebench - make it optional
    try:
        from swebench.harness.constants import (  # type: ignore
            APPLY_PATCH_FAIL,
            RESET_FAILED,
            TESTS_ERROR,
            TESTS_TIMEOUT,
        )
        from swebench.harness.grading import MAP_REPO_TO_PARSER  # type: ignore
    except ImportError:
        logger.warning("swebench package not available, using basic test parsing")
        # Basic parsing when swebench is not available
        return parse_test_output_basic(test_output, state)

    # Check for error strings
    error_string_search = {
        x: x in test_output
        for x in [
            APPLY_PATCH_FAIL,
            RESET_FAILED,
            TESTS_ERROR,
            TESTS_TIMEOUT,
            "Failed to reset task environment",
        ]
    }

    empty_results: tuple[dict, dict] = {}, {}
    if any(error_string_search.values()):
        return (
            INCORRECT,
            f"Tests did not run correctly. Errors found: {error_string_search}",
            *empty_results,
        )

    # Parse test results using repo-specific parser
    test_output_parser = MAP_REPO_TO_PARSER.get(state.metadata["repo"])
    if test_output_parser is None:
        logger.warning(
            f"No parser available for repo {state.metadata['repo']}, using basic parsing"
        )
        return parse_test_output_basic(test_output, state)

    test_output_parsed = test_output_parser(test_output)

    # Check pass-to-pass and fail-to-pass tests
    pass_to_pass_results = {k: "FAILED" for k in state.metadata["PASS_TO_PASS"]}
    fail_to_pass_results = {k: "FAILED" for k in state.metadata["FAIL_TO_PASS"]}

    for k, v in test_output_parsed.items():
        if k in state.metadata["PASS_TO_PASS"]:
            pass_to_pass_results[k] = v
        elif k in state.metadata["FAIL_TO_PASS"]:
            fail_to_pass_results[k] = v

    # Check if all tests passed
    passed_all_tests = all(
        v == "PASSED" for v in pass_to_pass_results.values()
    ) and all(v == "PASSED" for v in fail_to_pass_results.values())
    value = CORRECT if passed_all_tests else INCORRECT

    # Sort results (failures first)
    pass_to_pass_results = dict(
        sorted(pass_to_pass_results.items(), key=lambda x: x[1] == "PASSED")
    )
    fail_to_pass_results = dict(
        sorted(fail_to_pass_results.items(), key=lambda x: x[1] == "PASSED")
    )

    # Create summary
    p2p_counts = Counter(pass_to_pass_results.values())
    f2p_counts = Counter(fail_to_pass_results.values())

    p2p_summary = "\n".join(f"* {k}: {v}" for k, v in p2p_counts.items())
    f2p_summary = "\n".join(f"* {k}: {v}" for k, v in f2p_counts.items())

    explanation = (
        f"Pass-to-pass tests:\n{p2p_summary}\n\nFail-to-pass tests:\n{f2p_summary}"
    )

    return value, explanation, pass_to_pass_results, fail_to_pass_results


def parse_test_output_basic(
    test_output: str, state: TaskState
) -> tuple[str, str, dict, dict]:
    """
    Basic test output parsing when swebench package is not available.
    """
    # Initialize results
    pass_to_pass_results = {k: "UNKNOWN" for k in state.metadata["PASS_TO_PASS"]}
    fail_to_pass_results = {k: "UNKNOWN" for k in state.metadata["FAIL_TO_PASS"]}

    # Simple heuristic: look for test names in output
    for test_name in state.metadata["PASS_TO_PASS"]:
        if f"{test_name} PASSED" in test_output or f"PASSED {test_name}" in test_output:
            pass_to_pass_results[test_name] = "PASSED"
        elif (
            f"{test_name} FAILED" in test_output or f"FAILED {test_name}" in test_output
        ):
            pass_to_pass_results[test_name] = "FAILED"

    for test_name in state.metadata["FAIL_TO_PASS"]:
        if f"{test_name} PASSED" in test_output or f"PASSED {test_name}" in test_output:
            fail_to_pass_results[test_name] = "PASSED"
        elif (
            f"{test_name} FAILED" in test_output or f"FAILED {test_name}" in test_output
        ):
            fail_to_pass_results[test_name] = "FAILED"

    # Count results
    p2p_counts = Counter(pass_to_pass_results.values())
    f2p_counts = Counter(fail_to_pass_results.values())

    # If we couldn't parse any results, mark as incorrect
    if p2p_counts.get("UNKNOWN", 0) == len(pass_to_pass_results) and f2p_counts.get(
        "UNKNOWN", 0
    ) == len(fail_to_pass_results):
        return (
            INCORRECT,
            "Could not parse test results (swebench package not installed)",
            pass_to_pass_results,
            fail_to_pass_results,
        )

    # Check if all known tests passed
    passed_all = all(
        v == "PASSED" for v in pass_to_pass_results.values() if v != "UNKNOWN"
    ) and all(v == "PASSED" for v in fail_to_pass_results.values() if v != "UNKNOWN")

    value = CORRECT if passed_all else INCORRECT

    p2p_summary = "\n".join(f"* {k}: {v}" for k, v in p2p_counts.items())
    f2p_summary = "\n".join(f"* {k}: {v}" for k, v in f2p_counts.items())

    explanation = (
        f"Pass-to-pass tests:\n{p2p_summary}\n\nFail-to-pass tests:\n{f2p_summary}"
    )

    return value, explanation, pass_to_pass_results, fail_to_pass_results


def create_model_patch_cmd(base_commit: str) -> str:
    """Create command to generate a patch from the model's changes."""
    return f"""cd /testbed
git add -A
git diff --cached {base_commit} > model.patch"""


def create_eval_script(
    test_patch: str, repo: str, version: str, base_commit: str
) -> str:
    """
    Creates a script that runs the tests for the given patch.
    """
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # type: ignore
        from swebench.harness.utils import get_test_directives  # type: ignore

        # Get test command for the repo
        test_command = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]

        # Find modified files in test patch
        test_patch_files = re.findall(r"--- a/(.*)", test_patch)

        # Get test directives
        test_files = get_test_directives({"repo": repo, "test_patch": test_patch})

    except (ImportError, KeyError) as e:
        logger.warning(f"Could not load swebench test specs: {e}, using fallback")
        # Fallback to pytest
        test_command = "pytest"
        test_patch_files = re.findall(r"--- a/(.*)", test_patch)
        test_files = []

    # Create evaluation script
    eval_script = f"""#!/bin/bash
set -uo pipefail -x
source ~/.bashrc

# Reset test files to base state
git checkout {base_commit} {" ".join(test_patch_files)}

# Apply test patch
echo {shlex.quote(test_patch)} > /tmp/test_patch.diff
git apply --check /tmp/test_patch.diff
git apply /tmp/test_patch.diff

# Run tests
set +x
{test_command} {" ".join(test_files)} || true
"""

    return eval_script
