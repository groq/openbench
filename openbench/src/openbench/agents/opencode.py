"""
OpenCode agent implementation.
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
    get_opencode_script_template,
)
from openbench.utils.docker import OpenCodeCommands


class OpenCodeAgent(BaseCodeAgent):
    """OpenAI-compatible code generation tool."""

    def __init__(self):
        super().__init__("opencode")

    async def execute(self, workdir: str, prompt_text: str, model: str) -> str:
        """Execute OpenCode CLI command.

        Args:
            workdir: Working directory path for the task
            prompt_text: The prompt to send to opencode
            model: Model string to use with opencode

        Returns:
            Formatted output string with opencode execution results
        """
        try:
            # Write prompt to avoid shell quoting issues
            if not await write_prompt_to_file(prompt_text, "opencode_prompt.txt"):
                return "ERROR: failed to write prompt file"

            # Get environment setup script
            env_setup = generate_env_setup_script()

            # Create opencode execution script
            script_content = get_opencode_script_template().format(
                workdir=workdir, env_setup=env_setup, model=model
            )

            # Execute the script
            result = await write_and_execute_script(
                script_content,
                "opencode_script.sh",
                timeout=1800,  # 30 minutes
            )

            # Read opencode-specific log
            additional_logs = []
            opencode_log = await read_log_file(
                "/tmp/opencode-output.log", "OPENCODE", tail_lines=200
            )
            if opencode_log:
                additional_logs.append(opencode_log)

            return format_execution_output(result, additional_logs)

        except Exception as e:
            return f"ERROR: Failed to run opencode: {str(e)}"

    def resolve_model(self, state_model: str) -> str:
        """Resolve the appropriate model string for OpenCode.

        Args:
            state_model: Model from TaskState.model

        Returns:
            Resolved model string for OpenCode
        """
        # Check for environment override
        env_model = os.getenv("OPEN_CODE_MODEL")
        if env_model:
            return env_model

        # OpenCode uses model strings directly
        return state_model

    def get_setup_commands(self) -> List[str]:
        """Get setup commands required by OpenCode.

        Returns:
            Empty list (no special setup required)
        """
        return []

    def get_default_model(self) -> str:
        """Get the default model for OpenCode.

        Returns:
            Default model string
        """
        return os.getenv("BENCH_MODEL", "groq/openai/gpt-oss-20b")

    def get_description(self) -> str:
        """Get description of OpenCode.

        Returns:
            Description string
        """
        return "opencode cli code agent"

    def get_dockerfile_commands(self) -> List[str]:
        """Get Dockerfile commands to install OpenCode.

        Returns:
            List of Dockerfile RUN commands
        """
        return OpenCodeCommands.DOCKERFILE_COMMANDS

    def get_base_packages(self) -> List[str]:
        """Get base packages required by OpenCode.

        Returns:
            List of apt package names
        """
        return OpenCodeCommands.BASE_PACKAGES

    def get_env_requirements(self) -> List[str]:
        """Get environment variables required by OpenCode.

        Returns:
            List of environment variable names
        """
        return []  # OpenCode uses provider env vars automatically
