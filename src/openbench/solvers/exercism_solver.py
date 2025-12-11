"""
Unified CLI solver for exercism tasks that supports multiple code agents.

This solver provides a unified interface for different CLI code agents
(codex, aider, opencode, claude_code, roo)
and selects the appropriate tool based on the --code-agent flag or task arguments.

Supported code agents:
- codex: Codex CLI agent powered by inspect_swe (default)
- aider: AI-powered pair programming tool with git integration
- opencode: OpenAI-compatible code generation tool
- claude_code: Claude Code editor powered by inspect_swe
- roo: Roo extension for VS Code with interactive development

Usage:
    openbench eval exercism --code-agent codex --model openai/gpt-5
    openbench eval exercism --code-agent opencode --model openai/gpt-4o-mini
    openbench eval exercism --code-agent claude_code --model anthropic/claude-sonnet-4-5-20250929
    openbench eval exercism --code-agent aider --model groq/llama-3.1-70b
    openbench eval exercism --code-agent roo --model openrouter/anthropic/claude-sonnet-4-20250514
"""

from __future__ import annotations

from typing import Any, Dict

from inspect_ai.solver import Solver, TaskState, solver
from openbench.utils.cli_commands import (
    ensure_repo_and_task,
    run_setup_commands,
    run_final_test,
    format_solver_output,
    prepare_hidden_workspace,
    sync_agent_workspace,
)
from openbench.agents import AgentManager


@solver
def exercism_solver() -> Solver:
    """
    Unified CLI-based solver for exercism tasks.

    This solver supports multiple CLI code agents and automatically selects
    the appropriate tool based on the code agent specified in task arguments.

    The code agent can be specified via:
    - CLI flag: --code-agent codex|aider|opencode|claude_code|roo
    - Defaults to 'codex' if not specified

    Returns:
        Solver function that handles the task execution
    """

    async def solve(state: TaskState, generate) -> TaskState:  # type: ignore[override]
        # Required metadata from dataset
        language = state.metadata.get("language")
        task_name = state.metadata.get("task_name")
        test_command = state.metadata.get("test_command")
        setup_commands = state.metadata.get("setup_commands", [])

        if not all([language, task_name, test_command]):
            state.output.completion = f"ERROR: Missing required metadata - language: {language}, task_name: {task_name}, test_command: {test_command}"
            return state

        assert isinstance(language, str)
        assert isinstance(task_name, str)
        assert isinstance(test_command, str)
        if not isinstance(setup_commands, list):
            setup_commands = []

        code_agent = state.metadata.get("code_agent", "codex")

        # Validate code agent input
        if isinstance(code_agent, list) and len(code_agent) > 0:
            code_agent = code_agent[0]
        elif not isinstance(code_agent, str):
            code_agent = "codex"

        code_agent = code_agent.lower()

        # Validate code agent
        if not AgentManager.is_valid_agent(code_agent):
            valid_agents = AgentManager.get_supported_agents()
            state.output.completion = f"ERROR: Invalid code agent '{code_agent}'. Supported code agents: {', '.join(valid_agents)}"
            return state

        try:
            # Ensure repo and task directory exist under /workspace
            ok = await ensure_repo_and_task(language, task_name)
            if not ok:
                state.output.completion = (
                    f"ERROR: Failed to prepare /workspace/{language}/{task_name}"
                )
                return state

            full_workdir = f"/workspace/{language}/{task_name}"
            agent_workdir = full_workdir
            hide_tests = bool(state.metadata.get("hide_tests"))
            sync_context: Dict[str, Any] | None = None

            if hide_tests:
                prep_result = await prepare_hidden_workspace(language, task_name)
                if not prep_result.get("success"):
                    stderr = prep_result.get("stderr") or "unknown error"
                    state.output.completion = (
                        f"ERROR: Failed to prepare hidden workspace: {stderr}"
                    )
                    return state
                agent_workdir = str(prep_result.get("agent_dir", full_workdir))
                sync_context = {
                    "hidden_paths": prep_result.get("hidden_paths", {}),
                }

            prompt_text = state.input_text

            # Run any language-specific setup commands inside the task directory
            setup_out = await run_setup_commands(setup_commands, agent_workdir)

            agent = AgentManager.get_agent(code_agent)

            model = agent.resolve_model_with_fallback(str(state.model))

            code_agent_out = await agent.execute(agent_workdir, prompt_text, model)

            if hide_tests and sync_context is not None:
                sync_result = await sync_agent_workspace(
                    agent_workdir,
                    full_workdir,
                    dict(sync_context.get("hidden_paths", {})),
                )
                if not sync_result.get("success"):
                    stderr = sync_result.get("stderr") or "unknown error"
                    state.output.completion = (
                        f"ERROR: Failed to sync hidden workspace: {stderr}"
                    )
                    return state

            test_out = await run_final_test(test_command, full_workdir)

            state.output.completion = format_solver_output(
                code_agent, setup_out, code_agent_out, test_out
            )

        except Exception as e:  # pragma: no cover - defensive
            state.output.completion = (
                f"ERROR: {code_agent} code agent execution failed: {e}"
            )

        return state

    return solve
