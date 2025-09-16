"""
Unified CLI solver for exercism tasks that supports multiple code agents.

This solver provides a unified interface for different CLI code agents (aider, opencode, claude, roo)
and selects the appropriate tool based on the --code-agent flag or task arguments.

Supported code agents:
- aider: AI-powered pair programming tool with git integration
- opencode: OpenAI-compatible code generation tool
- claude: Claude-based code editor with file system access
- roo: Roo extension for VS Code with interactive development

Usage:
    openbench eval exercism --code-agent aider --model groq/llama-3.1-70b
    openbench eval exercism --code-agent opencode --model openai/gpt-4o-mini
    openbench eval exercism --code-agent claude --model anthropic/claude-sonnet-4-20250514
    openbench eval exercism --code-agent roo --model openrouter/anthropic/claude-sonnet-4-20250514
"""

from __future__ import annotations


from inspect_ai.solver import Solver, TaskState, solver
from openbench.utils.cli_commands import (
    ensure_repo_and_task,
    run_setup_commands,
    run_final_test,
    format_solver_output,
)
from openbench.agents import AgentManager


@solver
def exercism_solver() -> Solver:
    """
    Unified CLI-based solver for exercism tasks.

    This solver supports multiple CLI code agents and automatically selects
    the appropriate tool based on the code agent specified in task arguments.

    The code agent can be specified via:
    - CLI flag: --code-agent aider|opencode|claude|roo
    - Defaults to 'aider' if not specified

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

        code_agent = state.metadata.get("code_agent", "aider")

        # Validate code agent input
        if isinstance(code_agent, list) and len(code_agent) > 0:
            code_agent = code_agent[0]
        elif not isinstance(code_agent, str):
            code_agent = "aider"

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

            workdir = f"/workspace/{language}/{task_name}"
            prompt_text = state.input_text

            # Run any language-specific setup commands inside the task directory
            setup_out = await run_setup_commands(setup_commands, workdir)

            agent = AgentManager.get_agent(code_agent)

            model = agent.resolve_model_with_fallback(str(state.model))

            code_agent_out = await agent.execute(workdir, prompt_text, model)

            test_out = await run_final_test(test_command, workdir)

            state.output.completion = format_solver_output(
                code_agent, setup_out, code_agent_out, test_out
            )

        except Exception as e:  # pragma: no cover - defensive
            state.output.completion = (
                f"ERROR: {code_agent} code agent execution failed: {e}"
            )

        return state

    return solve
