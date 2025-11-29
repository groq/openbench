"""
Gemini CLI agent implementation.
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
    get_gemini_script_template,
)
from openbench.utils.docker import GeminiCommands


class GeminiAgent(BaseCodeAgent):
    """Google Gemini CLI code generation tool."""

    def __init__(self):
        super().__init__("gemini")

    async def execute(self, workdir: str, prompt_text: str, model: str) -> str:
        """Execute Gemini CLI command.

        Args:
            workdir: Working directory path for the task
            prompt_text: The prompt to send to gemini
            model: Model string to use with gemini

        Returns:
            Formatted output string with gemini execution results
        """
        try:
            if not await write_prompt_to_file(prompt_text, "gemini_prompt.txt"):
                return "ERROR: failed to write prompt file"

            # Get environment setup script
            env_setup = generate_env_setup_script()

            # Create gemini execution script
            script_content = get_gemini_script_template().format(
                workdir=workdir, env_setup=env_setup, model=model
            )

            # Execute the script
            result = await write_and_execute_script(
                script_content,
                "gemini_script.sh",
                timeout=1800,  # 30 minutes
            )

            additional_logs = []
            gemini_log = await read_log_file(
                "/tmp/gemini-output.log", "GEMINI", tail_lines=200
            )
            if gemini_log:
                additional_logs.append(gemini_log)

            return format_execution_output(result, additional_logs)

        except Exception as e:
            return f"ERROR: Failed to run gemini: {str(e)}"

    def resolve_model(self, state_model: str) -> str:
        """Resolve the appropriate model string for Gemini CLI.

        Args:
            state_model: Model from TaskState.model

        Returns:
            Resolved model string for Gemini CLI
        """
        if state_model.startswith("google/"):
            return state_model[7:]

        return state_model

    def get_setup_commands(self) -> List[str]:
        """Get setup commands required by Gemini CLI.

        Returns:
            Empty list (no special setup required)
        """
        return []

    def get_default_model(self) -> str:
        """Get the default model for Gemini CLI.

        Returns:
            Default model string
        """
        return os.getenv("BENCH_MODEL", "google/gemini-2.5-pro")

    def get_description(self) -> str:
        """Get description of Gemini CLI.

        Returns:
            Description string
        """
        return "gemini cli code agent"

    def get_dockerfile_commands(self) -> List[str]:
        """Get Dockerfile commands to install Gemini CLI.

        Returns:
            List of Dockerfile RUN commands
        """
        return GeminiCommands.DOCKERFILE_COMMANDS

    def get_base_packages(self) -> List[str]:
        """Get base packages required by Gemini CLI.

        Returns:
            List of apt package names
        """
        return GeminiCommands.BASE_PACKAGES

    def get_env_requirements(self) -> List[str]:
        """Get environment variables required by Gemini CLI.

        Returns:
            List of environment variable names
        """
        return ["GEMINI_API_KEY"]
