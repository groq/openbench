"""
Utility helpers for Exercism solvers.

These helpers manage the shared /workspace checkout, run optional setup
commands, execute the final test command, and format solver output so the
scorer can continue to parse `[FINAL_TEST_RESULTS]`.
"""

from __future__ import annotations

import re
from typing import List

from inspect_ai.util import sandbox


async def ensure_repo_and_task(language: str, task_name: str) -> bool:
    """Ensure the Exercism repository is available inside /workspace."""
    try:
        commands: List[str] = [
            "mkdir -p /workspace",
            "[ -d /workspace/.git ] || git clone https://github.com/RooCodeInc/Roo-Code-Evals.git /workspace",
            f"test -d /workspace/{language}/{task_name}",
            f"ls -la /workspace/{language}/{task_name}",
        ]
        result = await sandbox().exec(
            cmd=["bash", "-lc", " && ".join(commands)],
            timeout=180,
        )
        return result.returncode == 0
    except Exception:
        return False


async def run_setup_commands(setup_commands: List[str], workdir: str) -> str:
    """Execute optional setup commands inside the task workspace."""
    if not setup_commands:
        return "No setup commands"

    joined = " && ".join(setup_commands)
    try:
        result = await sandbox().exec(
            cmd=["bash", "-lc", f"cd {workdir} && ({joined})"],
            timeout=900,
        )
        parts: List[str] = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]
        if result.stdout:
            parts.extend(["", "--- STDOUT ---", result.stdout])
        if result.stderr:
            parts.extend(["", "--- STDERR ---", result.stderr])
        return "\n".join(parts)
    except Exception as exc:
        return f"ERROR: setup failed: {exc}"


async def run_final_test(test_command: str, workdir: str) -> str:
    """Run the benchmark's final test command and capture stdout/stderr."""
    try:
        fixed_test_command = test_command
        if "python" in workdir.lower():
            fixed_test_command = re.sub(
                r"([a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*)-test\.py",
                lambda m: m.group(0).replace("-", "_"),
                test_command,
            )
            fixed_test_command = re.sub(
                r"([a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+)(_test\.py)",
                lambda m: m.group(1).replace("-", "_") + m.group(2),
                fixed_test_command,
            )

        result = await sandbox().exec(
            cmd=["bash", "-lc", f"cd {workdir} && {fixed_test_command}"],
            timeout=600,
        )
        parts: List[str] = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]
        if result.stdout:
            parts.extend(["", "--- STDOUT ---", result.stdout])
        if result.stderr:
            parts.extend(["", "--- STDERR ---", result.stderr])
        return "\n".join(parts)
    except Exception as exc:
        return f"ERROR: test run failed: {exc}"


def format_solver_output(
    code_agent: str, setup_out: str, code_agent_out: str, test_out: str
) -> str:
    """Standardize solver output for downstream scoring."""
    code_agent_section_map = {
        "codex": "CODEX_AGENT_OUTPUT",
        "codex_cli": "CODEX_AGENT_OUTPUT",
        "claude_code": "CLAUDE_CODE_OUTPUT",
        "claude": "CLAUDE_CODE_OUTPUT",
    }

    code_agent_section = code_agent_section_map.get(
        code_agent, f"{code_agent.upper()}_OUTPUT"
    )

    return "\n".join(
        [
            "[SETUP_OUTPUT]",
            setup_out,
            "",
            f"[{code_agent_section}]",
            code_agent_out,
            "",
            "[FINAL_TEST_RESULTS]",
            test_out,
        ]
    )
