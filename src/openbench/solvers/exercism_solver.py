"""
Exercism solver that runs Inspect SWE code agents inside sandboxed workspaces.
"""

from __future__ import annotations

from copy import deepcopy

from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
)
from inspect_ai.solver import Solver, TaskState, solver

from openbench.agents import AgentManager
from openbench.utils.cli_commands import (
    ensure_repo_and_task,
    format_solver_output,
    run_final_test,
    run_setup_commands,
)


@solver
def exercism_solver() -> Solver:
    """Return a solver that orchestrates Inspect SWE agents for Exercism."""

    async def solve(state: TaskState, generate) -> TaskState:  # type: ignore[override]
        language = state.metadata.get("language")
        task_name = state.metadata.get("task_name")
        test_command = state.metadata.get("test_command")
        setup_commands = state.metadata.get("setup_commands", [])

        if not all([language, task_name, test_command]):
            state.output.completion = (
                "ERROR: Missing required metadata - "
                f"language: {language}, task_name: {task_name}, test_command: {test_command}"
            )
            return state

        assert isinstance(language, str)
        assert isinstance(task_name, str)
        assert isinstance(test_command, str)
        if not isinstance(setup_commands, list):
            setup_commands = []

        code_agent = state.metadata.get("code_agent", "codex")
        if isinstance(code_agent, list) and code_agent:
            code_agent = code_agent[0]
        elif not isinstance(code_agent, str):
            code_agent = "codex"

        code_agent = code_agent.lower()
        if not AgentManager.is_valid_agent(code_agent):
            valid_agents = ", ".join(AgentManager.get_supported_agents())
            state.output.completion = f"ERROR: Invalid code agent '{code_agent}'. Supported code agents: {valid_agents}"
            return state

        workdir = f"/workspace/{language}/{task_name}"

        try:
            ok = await ensure_repo_and_task(language, task_name)
        except Exception as exc:  # pragma: no cover - defensive
            state.output.completion = f"ERROR: failed to prepare workspace: {exc}"
            return state

        if not ok:
            state.output.completion = (
                f"ERROR: Failed to prepare /workspace/{language}/{task_name}"
            )
            return state

        prompt_text = state.input_text
        base_messages: list[ChatMessage] = [
            ChatMessageUser(content=prompt_text),
        ]
        state.messages = deepcopy(base_messages)

        setup_out = await run_setup_commands(setup_commands, workdir)

        model_name = AgentManager.resolve_model(code_agent, str(state.model))

        agent_state: AgentState | None = None
        agent_error: Exception | None = None

        try:
            inspect_agent = AgentManager.create_agent(
                code_agent,
                cwd=workdir,
                model=model_name,
            )
            agent_state = await inspect_agent(
                AgentState(messages=deepcopy(base_messages))
            )
            state.messages = agent_state.messages
        except Exception as exc:  # pragma: no cover - defensive
            agent_error = exc

        code_agent_out = _summarize_agent_run(
            code_agent, model_name, workdir, agent_state, agent_error
        )

        test_out = await run_final_test(test_command, workdir)

        state.output.completion = format_solver_output(
            code_agent, setup_out, code_agent_out, test_out
        )
        return state

    return solve


def _summarize_agent_run(
    agent_name: str,
    model_name: str,
    workdir: str,
    agent_state: AgentState | None,
    error: Exception | None,
) -> str:
    """Build a human-readable summary of the Inspect SWE agent execution."""
    lines = [
        f"Agent: {agent_name}",
        f"Model: {model_name}",
        f"Working Directory: {workdir}",
    ]

    if error is not None:
        lines.extend(["", f"ERROR: {error}"])
        return "\n".join(lines)

    if agent_state is None:
        lines.extend(
            ["", "Agent did not run. See Inspect SWE logs for more information."]
        )
        return "\n".join(lines)

    completion = agent_state.output.completion.strip()
    if completion:
        lines.extend(["", "--- AGENT COMPLETION ---", completion])

    assistant_messages = [
        message.text
        for message in agent_state.messages
        if isinstance(message, ChatMessageAssistant) and message.text
    ]
    if assistant_messages:
        lines.extend(["", "--- ASSISTANT MESSAGES (last 3) ---"])
        for snippet in assistant_messages[-3:]:
            lines.append(snippet)
    else:
        lines.extend(
            [
                "",
                "No assistant messages were recorded. Full command output is available "
                "in the Inspect SWE log trace.",
            ]
        )

    return "\n".join(lines)
