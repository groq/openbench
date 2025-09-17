"""
Base class for code agents in OpenBench evaluations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseCodeAgent(ABC):
    """Abstract base class for code agents."""

    def __init__(self, name: str):
        """Initialize the code agent.

        Args:
            name: The name of the code agent
        """
        self.name = name

    @abstractmethod
    async def execute(self, workdir: str, prompt_text: str, model: str) -> str:
        """Execute the code agent with the given parameters.

        Args:
            workdir: Working directory path for the task
            prompt_text: The prompt to send to the code agent
            model: Model string to use with the code agent

        Returns:
            Formatted output string with execution results
        """
        pass

    @abstractmethod
    def resolve_model(self, state_model: str) -> str:
        """Resolve the appropriate model string for this code agent.

        Args:
            state_model: Model from TaskState.model

        Returns:
            Resolved model string for the code agent
        """
        pass

    def get_setup_commands(self) -> List[str]:
        """Get any setup commands required by this code agent.

        Returns:
            List of setup commands
        """
        return []

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this code agent.

        Returns:
            Default model string
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this code agent.

        Returns:
            Description string
        """
        pass

    def get_timeout_seconds(self) -> int:
        """Get the timeout in seconds for this code agent.

        Returns:
            Timeout in seconds (default: 300)
        """
        return 300

    def resolve_model_with_fallback(self, requested_model: Optional[str] = None) -> str:
        """Resolve model with fallback to default.

        Args:
            requested_model: Optionally requested model

        Returns:
            Resolved model string
        """
        if requested_model:
            # Let the agent's specific resolve_model handle the logic
            return self.resolve_model(requested_model)
        else:
            return self.get_default_model()

    @abstractmethod
    def get_dockerfile_commands(self) -> List[str]:
        """Get agent-specific Dockerfile commands for installation.

        Returns:
            List of Dockerfile RUN commands to install this agent
        """
        pass

    def get_base_packages(self) -> List[str]:
        """Get base system packages required by this agent.

        Returns:
            List of apt package names
        """
        return []

    def get_volumes(self) -> Dict[str, str]:
        """Get Docker volumes required by this agent.

        Returns:
            Dictionary mapping volume names to mount paths
        """
        return {}

    def get_runtime_commands(self) -> List[str]:
        """Get commands to run at container startup.

        Returns:
            List of bash commands to run when container starts
        """
        return []

    def get_env_requirements(self) -> List[str]:
        """Get environment variables required by this agent.

        Returns:
            List of environment variable names
        """
        return []
