"""
Roo code agent implementation.
"""

from __future__ import annotations

import os
from typing import Dict, List

from .base import BaseCodeAgent
from openbench.utils.cli_commands import (
    generate_env_setup_script,
    write_and_execute_script,
    read_log_file,
    format_execution_output,
    get_roo_script_template,
)
from openbench.utils.docker import RooCommands


class RooAgent(BaseCodeAgent):
    """Roo extension for VS Code with interactive development."""

    def __init__(self):
        super().__init__("roo")

    async def execute(self, workdir: str, prompt_text: str, model: str) -> str:
        """Execute Roo CLI command with VS Code headless mode.

        Args:
            workdir: Working directory path for the task
            prompt_text: The prompt to send to roo-cli
            model: Model string (used for roo-cli configuration)

        Returns:
            Formatted output string with roo execution results
        """
        try:
            # Get environment setup script
            env_setup = generate_env_setup_script()

            # Add completion instruction to the prompt
            enhanced_prompt = f"""{prompt_text}

IMPORTANT: When you have completed the task, please run this exact command to signal completion:
echo "Task completed" > /tmp/roo-finish.log

This will allow the evaluation system to know when you're done."""

            # Create roo execution script
            script_content = get_roo_script_template().format(
                workdir=workdir,
                env_setup=env_setup,
                enhanced_prompt=enhanced_prompt,
                model=model,
            )

            # Execute the script with extended environment
            env = {
                "WORKDIR": workdir,
                "TASK_PROMPT": prompt_text,
                "ROO_CODE_IPC_SOCKET_PATH": "/tmp/roo-code.sock",
                "VSCODE_EXT_DIR": "/opt/vscode-extensions",
                "VSCODE_USER_DIR": "/opt/vscode-user",
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
            }

            result = await write_and_execute_script(
                script_content,
                "roo_cli_script.sh",
                timeout=1800,  # 30 minutes
                env=env,
            )

            # Read roo-specific logs
            additional_logs = []

            # Roo CLI output
            roo_output = await read_log_file(
                "/tmp/roo-cli-output.log", "ROO-CLI OUTPUT"
            )
            if roo_output:
                additional_logs.append(roo_output)

            # VS Code logs
            vscode_log = await read_log_file("/tmp/code.log", "VSCODE LOG")
            if vscode_log:
                additional_logs.append(vscode_log)

            return format_execution_output(result, additional_logs)

        except Exception as e:
            return f"ERROR: Failed to run roo-cli: {str(e)}"

    def get_setup_commands(self) -> List[str]:
        """Get setup commands required by Roo.

        Returns:
            Empty list (setup handled in script)
        """
        return []

    def get_default_model(self) -> str:
        """Get the default model for Roo.

        Returns:
            Default model string
        """
        return "openrouter/x-ai/grok-code-fast-1"

    def get_description(self) -> str:
        """Get description of Roo.

        Returns:
            Description string
        """
        return "Roo extension for VS Code"

    def resolve_model(self, state_model: str) -> str:
        """Resolve the appropriate model string for Roo with special handling.

        Args:
            state_model: Model from TaskState.model

        Returns:
            Resolved model string for Roo
        """
        # Roo expects OpenRouter model IDs
        if state_model.startswith("openrouter/"):
            return state_model.split("/", 1)[1]

        # Otherwise, use the state model as-is (CLI should enforce OpenRouter for --model)
        return state_model

    def get_dockerfile_commands(self) -> List[str]:
        """Get Dockerfile commands to install Roo (VS Code + extension + roo-cli).

        Returns:
            List of Dockerfile RUN commands
        """
        return RooCommands.get_dockerfile_commands()

    def get_base_packages(self) -> List[str]:
        """Get base packages required by Roo.

        Returns:
            List of apt package names
        """
        return RooCommands.BASE_PACKAGES

    def get_volumes(self) -> Dict[str, str]:
        """Get Docker volumes required by Roo.

        Returns:
            Dictionary mapping volume names to mount paths
        """
        return RooCommands.VOLUMES

    def get_runtime_commands(self) -> List[str]:
        """Get commands to run when Roo container starts.

        Returns:
            List of bash commands to run at startup
        """
        return RooCommands.RUNTIME_COMMANDS

    def get_env_requirements(self) -> List[str]:
        """Get environment variables required by Roo.

        Returns:
            List of environment variable names
        """
        return RooCommands.ENV_REQUIREMENTS
