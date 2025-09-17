"""
Claude code agent implementation.
"""

from __future__ import annotations

import os
from typing import List

from .base import BaseCodeAgent
from openbench.utils.cli_commands import (
    generate_env_setup_script,
    write_prompt_to_file,
    write_and_execute_script,
    read_log_file,
    format_execution_output,
    get_claude_script_template,
)
from openbench.utils.docker import ClaudeCommands


class ClaudeAgent(BaseCodeAgent):
    """Claude-based code editor with file system access."""

    def __init__(self):
        super().__init__("claude")

    async def execute(self, workdir: str, prompt_text: str, model: str) -> str:
        """Execute Claude Code CLI command.

        Args:
            workdir: Working directory path for the task
            prompt_text: The prompt to send to claude code
            model: Model string to use with claude code

        Returns:
            Formatted output string with claude code execution results
        """
        try:
            # Check for required API key
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                return "ERROR: ANTHROPIC_API_KEY is not set"

            # Write prompt to avoid shell quoting issues
            if not await write_prompt_to_file(prompt_text, "claude_code_prompt.txt"):
                return "ERROR: failed to write prompt file"

            # Get environment setup script
            env_setup = generate_env_setup_script()

            # Create claude execution script
            script_content = get_claude_script_template().format(
                workdir=workdir, env_setup=env_setup, model=model
            )

            # Execute the script
            result = await write_and_execute_script(
                script_content,
                "claude_script.sh",
                timeout=1800,  # 30 minutes
            )

            # Read claude-specific log
            additional_logs = []
            claude_log = await read_log_file(
                "/tmp/claude-code-output.log", "CLAUDE CODE", tail_lines=200
            )
            if claude_log:
                additional_logs.append(claude_log)

            return format_execution_output(result, additional_logs)

        except Exception as e:
            return f"ERROR: Failed to run claude code: {str(e)}"

    def resolve_model(self, state_model: str) -> str:
        """Resolve the appropriate model string for Claude.

        Args:
            state_model: Model from TaskState.model

        Returns:
            Resolved model string for Claude (removes anthropic/ prefix)
        """
        # Claude CLI uses Anthropic models directly (remove prefix)
        if state_model.startswith("anthropic/"):
            return state_model[len("anthropic/") :]
        return state_model

    def get_setup_commands(self) -> List[str]:
        """Get setup commands required by Claude.

        Returns:
            Empty list (no special setup required)
        """
        return []

    def get_default_model(self) -> str:
        """Get the default model for Claude.

        Returns:
            Default model string
        """
        return "anthropic/claude-sonnet-4-20250514"

    def get_description(self) -> str:
        """Get description of Claude.

        Returns:
            Description string
        """
        return "Claude cli code agent"

    def get_dockerfile_commands(self) -> List[str]:
        """Get Dockerfile commands to install Claude Code CLI.

        Returns:
            List of Dockerfile RUN commands
        """
        return ClaudeCommands.DOCKERFILE_COMMANDS

    def get_base_packages(self) -> List[str]:
        """Get base packages required by Claude.

        Returns:
            List of apt package names
        """
        return ClaudeCommands.BASE_PACKAGES

    def get_env_requirements(self) -> List[str]:
        """Get environment variables required by Claude.

        Returns:
            List of environment variable names
        """
        return ["ANTHROPIC_API_KEY"]  # Claude specifically requires Anthropic API key
