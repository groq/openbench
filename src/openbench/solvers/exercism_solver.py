"""
Unified CLI solver for Roo-Code tasks that supports multiple harnesses.

This solver provides a unified interface for different CLI harnesses (aider, opencode, claude, roo)
and selects the appropriate tool based on the --harness flag or task arguments.

Supported harnesses:
- aider: AI-powered pair programming tool with git integration
- opencode: OpenAI-compatible code generation tool
- claude: Claude-based code editor with file system access
- roo: Roo extension for VS Code with interactive development

Usage:
    openbench eval exercism --harness aider --model groq/llama-3.1-70b
    openbench eval exercism --harness opencode --model openai/gpt-4o-mini
    openbench eval exercism --harness claude --model anthropic/claude-sonnet-4-20250514
    openbench eval exercism --harness roo --model openrouter/anthropic/claude-sonnet-4-20250514
"""

from __future__ import annotations


from inspect_ai.solver import Solver, TaskState, solver
from openbench.utils.cli_commands import (
    ensure_repo_and_task,
    run_setup_commands,
    run_final_test,
    build_aider_command,
    build_opencode_command,
    build_claude_code_command,
    build_roo_command,
    resolve_model_for_harness,
    format_solver_output,
)


@solver
def exercism_solver() -> Solver:
    """
    Unified CLI-based solver for Roo-Code tasks.

    This solver supports multiple CLI harnesses and automatically selects
    the appropriate tool based on the harness specified in task arguments.

    The harness can be specified via:
    - CLI flag: --harness aider|opencode|claude|roo
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

        # Get harness from task metadata
        harness = state.metadata.get("harness", "aider")

        # Validate harness input
        if isinstance(harness, list) and len(harness) > 0:
            harness = harness[0]
        elif not isinstance(harness, str):
            harness = "aider"

        harness = harness.lower()

        # Validate harness
        if harness not in ["aider", "opencode", "claude", "roo"]:
            state.output.completion = f"ERROR: Invalid harness '{harness}'. Supported harnesses: aider, opencode, claude, roo"
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

            # Resolve the appropriate model for this harness
            model = resolve_model_for_harness(harness, str(state.model), {})

            # Execute the appropriate CLI harness
            if harness == "aider":
                harness_out = await build_aider_command(workdir, prompt_text, model)
            elif harness == "opencode":
                harness_out = await build_opencode_command(workdir, prompt_text, model)
            elif harness == "claude":
                harness_out = await build_claude_code_command(
                    workdir, prompt_text, model
                )
            elif harness == "roo":
                harness_out = await build_roo_command(workdir, prompt_text, model)
            else:
                harness_out = f"ERROR: Unsupported harness: {harness}"

            # Execute the task's test command
            test_out = await run_final_test(test_command, workdir)

            # Format the final output
            state.output.completion = format_solver_output(
                harness, setup_out, harness_out, test_out
            )

        except Exception as e:  # pragma: no cover - defensive
            state.output.completion = f"ERROR: {harness} CLI execution failed: {e}"

        return state

    return solve
