# src/openbench/scorers/mbpp.py
"""
MBPP verifier for OpenBench (Inspect), aligned with the paper-style prompt:

You are an expert Python programmer, and here is your task: {prompt}
Your code should pass these tests:

{tests}
[BEGIN]
{code}
[DONE]

Scoring policy:
  1) Assemble one Python program:
       [optional setup/imports]
       [candidate code extracted from the completion]
       [unit tests (assert statements)]
       print("__MBPP_OK__")
  2) Run: python -c "<assembled>"
  3) Pass iff exit code == 0
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional
from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    accuracy,
    stderr,
    CORRECT,
    INCORRECT,
)
from inspect_ai.util import ExecResult, sandbox


# code extraction logic

_BEGIN_DONE_RE = re.compile(r"\[BEGIN\](?P<code>[\s\S]*?)\[DONE\]", re.IGNORECASE)
_FENCED_RE = re.compile(r"```(?:python|py)?\s*(?P<code>[\s\S]*?)```", re.IGNORECASE)


def _extract_code(completion: str) -> str:
    """
    Extract Python code in priority order:
      1) [BEGIN] ... [DONE]
      2) ```python fenced code```
      3) raw completion
    """
    if not completion:
        return ""
    m = _BEGIN_DONE_RE.search(completion)
    if m:
        return m.group("code").strip()
    m = _FENCED_RE.search(completion)
    if m:
        return m.group("code").strip()
    return completion.strip()


# utils


def _normalize_tests(tests: Any) -> List[str]:
    """Accept list[str] or str; return list[str] of statements (asserts)."""
    if tests is None:
        return []
    if isinstance(tests, list):
        return [s for s in (str(x).rstrip() for x in tests) if s]
    if isinstance(tests, str):
        return [ln.rstrip() for ln in tests.splitlines() if ln.strip()]
    return []


def _assemble_program(
    *,
    setup: Optional[str],
    candidate_code: str,
    tests: List[str],
) -> str:
    """Order: setup -> candidate_code -> tests"""
    parts: List[str] = []
    if setup and setup.strip():
        parts.append(setup.strip())
    parts.append(candidate_code.strip())
    if tests:
        parts.append("\n".join(tests))
    return "\n\n".join(parts)


def _truncate(s: str, n: int = 1200) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


@scorer(metrics=[accuracy(), stderr()])
def mbpp_verify(*, timeout: int = 6) -> Callable[[TaskState, Target], Score]:
    """
    Execute model-generated MBPP solutions and unit tests inside the current sandbox.

    Parameters:
        timeout : int
            Seconds allowed for python -c execution.
    """

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata or {}
        task_id = md.get("task_id")
        prompt = md.get("prompt", "")
        setup = md.get("setup")
        tests = _normalize_tests(md.get("tests"))

        # extract candidate code from completion
        raw = state.output.completion if state.output else ""
        candidate = _extract_code(raw)

        if not candidate:
            program = _assemble_program(setup=setup, candidate_code="", tests=tests)
            explanation = (
                "No candidate code found.\n\n"
                "The following verification code was executed:\n\n```python\n"
                + program
                + "\n```\n"
            )
            return Score(
                value=INCORRECT,
                answer="",
                explanation=explanation,
                metadata={"task_id": task_id},
            )

        # assemble program
        program = _assemble_program(setup=setup, candidate_code=candidate, tests=tests)

        # execute in sandbox (task defaults to 'local')
        try:
            result = await sandbox().exec(
                cmd=["python", "-c", program],
                timeout=timeout,
            )
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")
        except Exception as e:
            result = ExecResult(False, 1, "", f"{type(e).__name__}: {e}")

        # decide pass/fail
        passed = bool(result.success)
        value = CORRECT if passed else INCORRECT

        # explanation (show exactly what ran; include stderr on failure)
        parts = [
            "The following verification code was executed:\n\n```python\n",
            program,
            "\n```\n",
        ]
        if not passed:
            parts.extend(["The submission was incorrect.\n\n", (result.stderr or "")])
        explanation = "".join(parts)

        return Score(
            value=value,
            answer=candidate,
            explanation=explanation,
            metadata={
                "task_id": task_id,
                "prompt": _truncate(str(prompt), 400),
                "tests_count": len(tests),
                "timeout_s": timeout,
                "exit_success": passed,
            },
        )

    return score
