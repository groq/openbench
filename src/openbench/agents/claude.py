"""
Claude Code agent backed by inspect_swe.
"""

from __future__ import annotations

from typing import List

from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageUser, ModelOutput

from openbench.utils.cli_commands import format_execution_output

from .base import BaseCodeAgent
from inspect_swe import claude_code  # type: ignore[import-not-found, import-untyped]


class ClaudeCodeAgent(BaseCodeAgent):
    """Claude Code CLI agent via inspect_swe."""

    def __init__(self) -> None:
        super().__init__("claude_code")

    async def execute(self, workdir: str, prompt_text: str, model: str) -> str:
        """Execute Claude Code."""

        try:
            claude_agent = claude_code(cwd=workdir, model=model)
            state = AgentState(messages=[ChatMessageUser(content=prompt_text)])
            completed_state = await claude_agent(state)
            stdout_text = _format_agent_output(completed_state.output)
            result = {
                "returncode": 0,
                "success": True,
                "stdout": stdout_text,
                "stderr": "",
            }
            return format_execution_output(result)
        except Exception as exc:  # pragma: no cover - defensive
            return f"ERROR: claude_code execution failed: {exc}"

    def resolve_model(self, state_model: str) -> str:
        """Resolve the appropriate model string for Claude Code."""
        stripped = (state_model or "").strip()
        return stripped if stripped else self.get_default_model()

    def get_default_model(self) -> str:
        return "anthropic/claude-sonnet-4-5-20250929"

    def get_description(self) -> str:
        return "Claude Code agent."

    def get_dockerfile_commands(self) -> List[str]:
        return []


def _format_agent_output(output: ModelOutput) -> str:
    """Render agent output as plain text."""
    if not output or not output.choices:
        return "Agent completed without emitting assistant output."

    parts: List[str] = []
    for idx, choice in enumerate(output.choices, start=1):
        message = choice.message
        text = (
            message.text.strip() if message and message.text else ""
        ) or "(no text output)"
        parts.append(f"[Choice {idx}] {text}")
    return "\n\n".join(parts)
