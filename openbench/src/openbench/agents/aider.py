"""
Aider code agent implementation.
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
    get_aider_script_template,
)
from openbench.utils.docker import AiderCommands


class AiderAgent(BaseCodeAgent):
    """Aider AI-powered pair programming tool with git integration."""

    def __init__(self):
        super().__init__("aider")

    async def execute(self, workdir: str, prompt_text: str, model: str) -> str:
        """Execute Aider with comprehensive environment setup.

        Args:
            workdir: Working directory path for the task
            prompt_text: The prompt to send to aider
            model: Model string to use with aider

        Returns:
            Formatted output string with aider execution results
        """
        try:
            # Write prompt to avoid shell quoting issues
            if not await write_prompt_to_file(prompt_text, "aider_prompt.txt"):
                return "ERROR: failed to write prompt file"

            # Get environment setup script
            env_setup = generate_env_setup_script()

            # Create aider execution script
            script_content = get_aider_script_template().format(
                workdir=workdir, env_setup=env_setup, model=model
            )

            # Execute the script
            result = await write_and_execute_script(
                script_content,
                "aider_script.sh",
                timeout=1800,  # 30 minutes
            )

            # Read aider-specific log
            additional_logs = []
            aider_log = await read_log_file("/tmp/aider-output.log", "AIDER")
            if aider_log:
                additional_logs.append(aider_log)

            return format_execution_output(result, additional_logs)

        except Exception as e:
            return f"ERROR: Failed to run aider: {str(e)}"

    def resolve_model(self, state_model: str) -> str:
        """Resolve the appropriate model string for Aider.

        Args:
            state_model: Model from TaskState.model

        Returns:
            Resolved model string for Aider
        """
        # Aider uses model strings directly
        return state_model

    def get_setup_commands(self) -> List[str]:
        """Get setup commands required by Aider.

        Returns:
            List of setup commands (git initialization)
        """
        return [
            "git init || true",  # Initialize git repo if not present
            "git config user.email 'test@example.com' || true",
            "git config user.name 'Test User' || true",
        ]

    def get_default_model(self) -> str:
        """Get the default model for Aider.

        Returns:
            Default model string
        """
        return os.getenv("BENCH_MODEL", "groq/openai/gpt-oss-20b")

    def get_description(self) -> str:
        """Get description of Aider.

        Returns:
            Description string
        """
        return "aider cli code agent"

    def get_dockerfile_commands(self) -> List[str]:
        """Get Dockerfile commands to install Aider.

        Returns:
            List of Dockerfile RUN commands
        """
        return AiderCommands.DOCKERFILE_COMMANDS

    def get_base_packages(self) -> List[str]:
        """Get base packages required by Aider.

        Returns:
            List of apt package names
        """
        return AiderCommands.BASE_PACKAGES

    def get_env_requirements(self) -> List[str]:
        """Get environment variables required by Aider.

        Returns:
            List of environment variable names
        """
        return []  # Aider uses provider env vars automatically
