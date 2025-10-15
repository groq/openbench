"""
Agent manager for code agents in openbench evaluations.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .base import BaseCodeAgent
from .aider import AiderAgent
from .opencode import OpenCodeAgent
from .claude import ClaudeAgent
from .roo import RooAgent


class AgentManager:
    """Manager class for code agents with configuration and model resolution."""

    _agents: Dict[str, Callable[[], BaseCodeAgent]] = {
        "aider": AiderAgent,
        "opencode": OpenCodeAgent,
        "claude": ClaudeAgent,
        "roo": RooAgent,
    }

    @classmethod
    def get_agent(cls, agent_name: str) -> BaseCodeAgent:
        """Get an instance of the specified code agent.

        Args:
            agent_name: Name of the code agent

        Returns:
            Instance of the requested code agent

        Raises:
            ValueError: If the agent name is not supported
        """
        agent_name = agent_name.lower()
        if agent_name not in cls._agents:
            raise ValueError(f"Unsupported code agent: {agent_name}")

        agent_class = cls._agents[agent_name]
        return agent_class()

    @classmethod
    def get_supported_agents(cls) -> List[str]:
        """Get list of supported code agent names.

        Returns:
            List of supported agent names
        """
        return list(cls._agents.keys())

    @classmethod
    def is_valid_agent(cls, agent_name: str) -> bool:
        """Check if an agent name is valid.

        Args:
            agent_name: Name of the code agent

        Returns:
            True if the agent is supported, False otherwise
        """
        return agent_name.lower() in cls._agents

    @classmethod
    def validate_code_agent(cls, agent_name: str) -> bool:
        """Validate if a code agent name is supported (alias for is_valid_agent).

        Args:
            agent_name: Name of the code agent to validate

        Returns:
            True if valid, False otherwise
        """
        return cls.is_valid_agent(agent_name)

    @classmethod
    def get_valid_code_agents(cls) -> List[str]:
        """Get list of all valid code agent names (alias for get_supported_agents).

        Returns:
            List of valid code agent names
        """
        return cls.get_supported_agents()

    @classmethod
    def get_default_model(cls, agent_name: str) -> str:
        """Get the default model for a code agent.

        Args:
            agent_name: Name of the code agent

        Returns:
            Default model string for the code agent

        Raises:
            ValueError: If agent name is not supported
        """
        agent = cls.get_agent(agent_name)
        return agent.get_default_model()

    @classmethod
    def get_description(cls, agent_name: str) -> str:
        """Get the description for a code agent.

        Args:
            agent_name: Name of the code agent

        Returns:
            Description string for the code agent

        Raises:
            ValueError: If agent name is not supported
        """
        agent = cls.get_agent(agent_name)
        return agent.get_description()

    @classmethod
    def get_timeout_seconds(cls, agent_name: str) -> int:
        """Get the timeout for a code agent.

        Args:
            agent_name: Name of the code agent

        Returns:
            Timeout in seconds for the code agent

        Raises:
            ValueError: If agent name is not supported
        """
        agent = cls.get_agent(agent_name)
        return agent.get_timeout_seconds()

    @classmethod
    def resolve_model(
        cls, agent_name: str, requested_model: Optional[str] = None
    ) -> str:
        """Resolve the appropriate model for a code agent.

        Args:
            agent_name: Name of the code agent
            requested_model: Optionally requested model

        Returns:
            Resolved model string

        Raises:
            ValueError: If agent name is not supported
        """
        agent = cls.get_agent(agent_name)
        return agent.resolve_model_with_fallback(requested_model)

    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict[str, str]]:
        """Get configuration information for all code agents.

        Returns:
            Dictionary mapping agent names to their configuration
        """
        configs = {}
        for agent_name in cls._agents.keys():
            agent = cls.get_agent(agent_name)
            configs[agent_name] = {
                "name": agent.name,
                "description": agent.get_description(),
                "default_model": agent.get_default_model(),
                "timeout_seconds": str(agent.get_timeout_seconds()),
            }
        return configs

    @classmethod
    def get_help_text(cls) -> str:
        """Generate help text for code agents.

        Returns:
            Formatted help text describing all available code agents
        """
        agent_names = cls.get_supported_agents()
        default_agent = "opencode"

        help_text = f"CLI code agent to use for code evaluation. Options: {', '.join(agent_names)} (default: {default_agent})"
        return help_text
